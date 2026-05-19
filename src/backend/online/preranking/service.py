"""在线 preranking 服务编排层。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.online.preranking.gbdt import run_preranking_gbdt
from backend.online.search_query import (
    DEFAULT_INTENT_PATTERNS,
    detect_query_intents,
    normalize_query_text,
    segment_query_terms,
)

_USER_SCALAR_COLS = (
    ["gender_enc", "platform_enc", "age_enc", "location_enc", "fans_num", "follows_num"]
    + [f"dense_feat{i}" for i in range(1, 41)]
)
_LINKAGE_TAG_COLS = ["taxonomy1_id", "taxonomy2_id", "taxonomy3_id"]
_NOTE_META_COLS = [
    "accum_like_num",
    "accum_collect_num",
    "accum_comment_num",
    "image_path",
    "note_title",
    "note_content",
]

def _char_ngrams(text: str, n: int = 2) -> set[str]:
    raw = normalize_query_text(text)
    if len(raw) <= n:
        return {raw} if raw else set()
    return {raw[i : i + n] for i in range(len(raw) - n + 1)}


def _search_match_score(
    query_phrase: str,
    title: str,
    content: str,
    terms: list[str] | None = None,
    intents: list[str] | None = None,
) -> float:
    q = normalize_query_text(query_phrase)
    if not q:
        return 0.0
    t = normalize_query_text(title)
    c = normalize_query_text(content)
    query_terms = [normalize_query_text(x) for x in (terms or segment_query_terms(q)) if normalize_query_text(x)]
    query_intents = [str(x) for x in (intents or detect_query_intents(q)) if str(x)]

    score = 0.0
    if q in t:
        score += 3.2
    if q in c:
        score += 1.0
    q_grams = _char_ngrams(q, n=2)
    if q_grams:
        title_overlap = len(q_grams & _char_ngrams(t, n=2)) / max(len(q_grams), 1)
        content_overlap = len(q_grams & _char_ngrams(c[:500], n=2)) / max(len(q_grams), 1)
        score += 1.4 * title_overlap + 0.35 * content_overlap
    if query_terms:
        title_hits = sum(1 for term in query_terms if term and term in t)
        content_hits = sum(1 for term in query_terms if term and term in c)
        score += 1.7 * (title_hits / max(len(query_terms), 1))
        score += 0.55 * (content_hits / max(len(query_terms), 1))
    if query_intents:
        matched = 0
        for intent in query_intents:
            for pat in DEFAULT_INTENT_PATTERNS.get(intent, ()):
                norm_pat = normalize_query_text(pat)
                if norm_pat and (norm_pat in t or norm_pat in c):
                    matched += 1
                    break
        score += 0.30 * matched
    return float(score)


def _valid_history_value(v) -> bool:
    if v is None:
        return False
    if isinstance(v, float) and np.isnan(v):
        return False
    if isinstance(v, np.ndarray):
        return v.size > 0
    if isinstance(v, (list, tuple)):
        return len(v) > 0
    return True


def _resolve_prerank_feat_cols(predict_gbdt, feat_req: pd.DataFrame, group_key: str) -> list[str]:
    cols = {group_key, "note_idx", "user_idx", "recent_clicked_note_idxs", "y_multi", "click", *_USER_SCALAR_COLS, *_LINKAGE_TAG_COLS, *_NOTE_META_COLS}
    predictor = getattr(predict_gbdt, "__self__", None)
    for col in getattr(predictor, "feature_cols", []) or []:
        cols.add(str(col))
    return [col for col in feat_req.columns if col in cols]


def run_preranking(
    user_idx: int,
    query: str,
    scene: str,
    group_key: str,
    recall_cand: pd.DataFrame,
    feat_req: pd.DataFrame,
    gbdt_topn: int,
    fetch_notes,
    predict_gbdt,
    linkage_ctx: dict | None = None,
    query_phrase: str | None = None,
    query_terms: list[str] | None = None,
    query_intents: list[str] | None = None,
) -> pd.DataFrame:
    # 召回候选拼接特征与内容元信息，再进入 GBDT preranking
    if recall_cand.empty:
        return pd.DataFrame()

    feat_cols = _resolve_prerank_feat_cols(predict_gbdt, feat_req, group_key)
    feat_frame = feat_req[feat_cols] if feat_cols else feat_req[[group_key, "note_idx"]]
    cand = recall_cand.merge(
        feat_frame,
        on=[group_key, "note_idx"],
        how="left",
        suffixes=("", "_feat"),
    )
    cand["note_idx"] = pd.to_numeric(cand.get("note_idx"), errors="coerce")
    cand = cand[cand["note_idx"].notna()].copy()
    if cand.empty:
        return pd.DataFrame()
    cand["note_idx"] = cand["note_idx"].astype("int64")
    cand = cand[cand["note_idx"] >= 0].copy()
    if cand.empty:
        return pd.DataFrame()

    if "rank" in cand.columns:
        cand = cand.sort_values([group_key, "rank"], ascending=[True, True], kind="mergesort")
    elif "recall_score" in cand.columns:
        cand = cand.sort_values([group_key, "recall_score"], ascending=[True, False], kind="mergesort")
    cand = cand.drop_duplicates(subset=[group_key, "note_idx"], keep="first").reset_index(drop=True)
    if "user_idx" not in cand.columns:
        cand["user_idx"] = int(user_idx)
    cand["user_idx"] = cand["user_idx"].fillna(int(user_idx)).astype("int64")

    if len(feat_req) > 0:
        user_profile = feat_req.iloc[0]
        for col in _USER_SCALAR_COLS:
            val = user_profile.get(col)
            if val is None:
                continue
            try:
                fill_val = float(val) if not isinstance(val, (int, float)) else val
                if isinstance(fill_val, float) and np.isnan(fill_val):
                    continue
            except Exception:
                continue
            if col not in cand.columns:
                cand[col] = fill_val
            else:
                cand[col] = pd.to_numeric(cand[col], errors="coerce").fillna(fill_val)
        if "recent_clicked_note_idxs" in user_profile:
            hist_val = user_profile["recent_clicked_note_idxs"]
            if "recent_clicked_note_idxs" not in cand.columns:
                cand["recent_clicked_note_idxs"] = [hist_val] * len(cand)
            else:
                cand["recent_clicked_note_idxs"] = [
                    v if _valid_history_value(v) else hist_val
                    for v in cand["recent_clicked_note_idxs"]
                ]

    note_cols = _NOTE_META_COLS
    need_note_meta = any((col not in cand.columns) or cand[col].isna().all() for col in note_cols)
    if need_note_meta:
        note_meta = fetch_notes(cand["note_idx"].drop_duplicates().astype(int).tolist())
        if not note_meta.empty:
            cand = cand.merge(note_meta, on="note_idx", how="left", suffixes=("", "_note"))

    if scene == "search":
        phrase_query = normalize_query_text(query_phrase or query)
        cand["query"] = phrase_query
        if phrase_query:
            titles = cand.get("note_title", pd.Series([""] * len(cand))).fillna("").astype(str)
            contents = cand.get("note_content", pd.Series([""] * len(cand))).fillna("").astype(str)
            terms = [normalize_query_text(x) for x in (query_terms or segment_query_terms(phrase_query)) if normalize_query_text(x)]
            intents = [str(x) for x in (query_intents or detect_query_intents(phrase_query)) if str(x)]
            title_lower = titles.astype(str).map(normalize_query_text)
            content_lower = contents.astype(str).map(normalize_query_text)
            cand["query_exact_hit"] = (
                title_lower.str.contains(phrase_query, regex=False)
                | content_lower.str.contains(phrase_query, regex=False)
            ).astype(np.int8)
            cand["query_match_score"] = np.asarray(
                [_search_match_score(phrase_query, t, c, terms=terms, intents=intents) for t, c in zip(titles.tolist(), contents.tolist())],
                dtype=np.float32,
            )
            if terms:
                title_hits = np.zeros(len(cand), dtype=np.float32)
                content_hits = np.zeros(len(cand), dtype=np.float32)
                for term in terms:
                    title_hits += title_lower.str.contains(term, regex=False).astype(np.float32).to_numpy()
                    content_hits += content_lower.str.contains(term, regex=False).astype(np.float32).to_numpy()
                denom = float(max(len(terms), 1))
                cand["query_term_hits"] = (title_hits + content_hits).astype(np.float32)
                cand["query_term_cover"] = np.clip((title_hits + 0.45 * content_hits) / denom, 0.0, 1.5).astype(np.float32)
            else:
                cand["query_term_hits"] = np.zeros(len(cand), dtype=np.float32)
                cand["query_term_cover"] = np.zeros(len(cand), dtype=np.float32)
            if intents:
                intent_hits = np.zeros(len(cand), dtype=np.float32)
                for intent in intents:
                    for pat in DEFAULT_INTENT_PATTERNS.get(intent, ()):
                        norm_pat = normalize_query_text(pat)
                        intent_hits += title_lower.str.contains(norm_pat, regex=False).astype(np.float32).to_numpy() * 0.7
                        intent_hits += content_lower.str.contains(norm_pat, regex=False).astype(np.float32).to_numpy() * 0.3
                cand["query_intent_match"] = np.clip(intent_hits / max(len(intents), 1), 0.0, 1.5).astype(np.float32)
            else:
                cand["query_intent_match"] = np.zeros(len(cand), dtype=np.float32)
        else:
            cand["query_exact_hit"] = np.zeros(len(cand), dtype=np.int8)
            cand["query_match_score"] = np.zeros(len(cand), dtype=np.float32)
            cand["query_term_hits"] = np.zeros(len(cand), dtype=np.float32)
            cand["query_term_cover"] = np.zeros(len(cand), dtype=np.float32)
            cand["query_intent_match"] = np.zeros(len(cand), dtype=np.float32)

    for col in note_cols:
        note_col = f"{col}_note"
        if note_col in cand.columns:
            if col in cand.columns:
                cand[col] = cand[col].where(cand[col].notna(), cand[note_col])
            else:
                cand[col] = cand[note_col]

    ann_score = pd.to_numeric(cand.get("score_ann", 0.0), errors="coerce").fillna(0.0)
    recall_score = pd.to_numeric(cand.get("recall_score", ann_score), errors="coerce").fillna(0.0)
    cand["dssm_score"] = ann_score.where(ann_score >= recall_score, recall_score).astype("float32")

    linkage_boost = np.zeros(len(cand), dtype=np.float32)
    if linkage_ctx:
        keywords = [str(x).strip().lower() for x in linkage_ctx.get("keywords", []) if str(x).strip()]
        tag_ids = {int(x) for x in linkage_ctx.get("tag_ids", []) if pd.notna(x) and int(x) >= 0}
        if keywords:
            titles = cand.get("note_title", pd.Series([""] * len(cand))).fillna("").astype(str).str.lower()
            contents = cand.get("note_content", pd.Series([""] * len(cand))).fillna("").astype(str).str.lower()
            for kw in keywords[:4]:
                linkage_boost += (
                    titles.str.contains(kw, regex=False).astype(np.float32).to_numpy() * (0.14 if scene == "rec" else 0.05)
                    + contents.str.contains(kw, regex=False).astype(np.float32).to_numpy() * (0.06 if scene == "rec" else 0.02)
                )
        if tag_ids:
            for col in _LINKAGE_TAG_COLS:
                if col in cand.columns:
                    linkage_boost += cand[col].isin(tag_ids).astype(np.float32).to_numpy() * (0.10 if scene == "search" else 0.06)
        if scene == "search":
            quality = np.log1p(
                pd.to_numeric(cand.get("accum_like_num", 0.0), errors="coerce").fillna(0.0).to_numpy(np.float32)
                + 2.0 * pd.to_numeric(cand.get("accum_collect_num", 0.0), errors="coerce").fillna(0.0).to_numpy(np.float32)
                + 2.0 * pd.to_numeric(cand.get("accum_comment_num", 0.0), errors="coerce").fillna(0.0).to_numpy(np.float32)
            )
            if quality.size > 0 and float(np.max(quality)) > 0:
                linkage_boost += (quality / float(np.max(quality))) * 0.05
    cand["linkage_boost"] = linkage_boost.astype(np.float32)

    out = run_preranking_gbdt(cand=cand, gbdt_topn=gbdt_topn, predict_gbdt=predict_gbdt)
    if scene == "search" and str(query or "").strip():
        exact = pd.to_numeric(out.get("query_exact_hit", 0), errors="coerce").fillna(0.0)
        lexical = pd.to_numeric(out.get("query_match_score", 0.0), errors="coerce").fillna(0.0)
        cover = pd.to_numeric(out.get("query_term_cover", 0.0), errors="coerce").fillna(0.0)
        weak_penalty = ((exact <= 0) & (cover < 0.34)).astype(np.float32) * 0.72
        strong_match = ((exact > 0) | (cover >= 0.45) | (lexical >= 1.25)).astype(np.float32)
        hard_penalty = ((exact <= 0) & (cover < 0.20) & (lexical < 0.80)).astype(np.float32) * 2.20
        out["query_strong_match"] = strong_match.astype(np.int8)
        intent_match = pd.to_numeric(out.get("query_intent_match", 0.0), errors="coerce").fillna(0.0)
        out["preranking_score"] = (
            pd.to_numeric(out.get("preranking_score", 0.0), errors="coerce").fillna(0.0)
            + exact * 1.55
            + lexical * 0.32
            + cover * 1.60
            + strong_match * 0.85
            + intent_match * 0.42
            - weak_penalty
            - hard_penalty
        ).astype(np.float32)
        out = out.sort_values(
            ["query_strong_match", "query_exact_hit", "query_term_cover", "preranking_score", "gbdt_score", "rank"],
            ascending=[False, False, False, False, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        strong_mask = (
            (pd.to_numeric(out.get("query_exact_hit", 0.0), errors="coerce").fillna(0.0) > 0)
            | (pd.to_numeric(out.get("query_term_cover", 0.0), errors="coerce").fillna(0.0) >= 0.34)
            | (pd.to_numeric(out.get("query_match_score", 0.0), errors="coerce").fillna(0.0) >= 0.88)
        )
        strong_df = out[strong_mask].copy()
        weak_df = out[~strong_mask].copy()
        if len(strong_df) >= min(18, max(12, int(gbdt_topn) // 2)):
            out = strong_df.reset_index(drop=True)
        elif len(strong_df) > 0:
            weak_keep = weak_df[
                (pd.to_numeric(weak_df.get("query_term_cover", 0.0), errors="coerce").fillna(0.0) >= 0.24)
                | (pd.to_numeric(weak_df.get("query_match_score", 0.0), errors="coerce").fillna(0.0) >= 0.90)
            ].copy()
            out = pd.concat([strong_df, weak_keep.head(max(0, int(gbdt_topn) - len(strong_df)))], ignore_index=True)
    return out
