"""Current-user request replay under the active model, separate from metric totals."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


def build_current_user_example(pipeline, user_idx: int, snapshot: dict):
    state = pipeline.state
    scene = state.scene
    if scene not in {"search", "rec"}:
        raise ValueError("Unsupported replay scene")
    root = Path(state.bundle.root)
    key = "search_idx" if scene == "search" else "request_idx"
    folder = "search" if scene == "search" else "recommendation"
    nested = "search_result_details_with_idx" if scene == "search" else "rec_result_details_with_idx"
    record = None
    for split in ("test", "train"):
        files = sorted((root / "datasets" / f"{folder}_{split}").glob("*.parquet"))
        if not files:
            continue
        if split == "test":
            # Match the authoritative test source bound to the versioned snapshot.
            from backend.online.bundle import verify_asset
            for path in files:
                expected = snapshot.get("input_hashes", {}).get(str(path))
                if not expected:
                    raise ValueError("Current-user replay requires a bound original test source")
                verify_asset(path, expected)
        with duckdb.connect(config={"threads": 2}) as con:
            con.read_parquet(list(map(str, files))).create_view("requests")
            selected = con.execute(f"SELECT {key},user_idx,query,recent_clicked_note_idxs,{nested} FROM requests WHERE user_idx=? ORDER BY {key} LIMIT 1",[int(user_idx)]).fetchdf()
        if not selected.empty:
            record = selected.iloc[0]
            break
    if record is None:
        return None
    gid = int(record[key])
    query = str(record["query"] or "") if scene == "search" else ""
    from training.history import recent_history_ids
    history = recent_history_ids(record["recent_clicked_note_idxs"])
    context = pd.DataFrame([{key: gid, "user_idx": int(user_idx), "query": query,
                             "recent_clicked_note_idxs": history}])
    # Outcome labels are deliberately not passed to inference; no interaction writes.
    feed = pipeline.build_feed(int(user_idx),query,1,10,record_exposure=False,
                               request_context=context,include_stage_items=True)
    if feed.get("model_bundle_id") != state.bundle.bundle_id:
        raise ValueError("Replay predictions belong to a different model")
    positive = list(dict.fromkeys(int(item["note_idx"]) for item in record[nested] if int(item["click"])>0))
    truth = {note: i+1 for i,note in enumerate(positive)}
    stages = feed["stage_top10_ids"]
    lists = {"true_top10": positive[:10], "dssm_top10": stages["recall"],
             "preranking_top10": stages["preranking"], "ranking_top10": stages["ranking"]}
    wanted = sorted(set(history + [int(note) for values in lists.values() for note in values]))
    metadata = state._fetch_notes(wanted)
    from backend.online.pipeline import _existing_images, _to_path_list, _format_note_title, _safe_intish
    notes = {}
    for row in metadata.to_dict("records"):
        note = int(row["note_idx"])
        images = _existing_images(_to_path_list(row.get("image_path")))
        notes[note] = {"note_idx": note, "title": _format_note_title(row.get("note_title"),row.get("note_content")),
                       "content": str(row.get("note_content") or ""), "images": images,
                       "cover_image": images[0] if images else "",
                       **{name:_safe_intish(row.get(name,0)) for name in ("accum_like_num","accum_collect_num","accum_comment_num")}}
    example = {"request_id": gid, "source_user_idx":int(user_idx), "is_current_user":True,
               "scene":scene, "query":query, "scope":"current_user_request_replay", "split":split,
               "model_bundle_id":state.bundle.bundle_id, "evaluation_included":False,
               "sample_note": "当前用户测试请求回放，仅展示，不改变全量指标" if split=="test" else "当前用户训练请求回放，不能作为测试成绩",
               "result_count":len(record[nested]), "notes":list(notes.values()),
               "source_user":{"user_idx":int(user_idx)},
               "source_recent_behaviors":[{"note_idx":note,"title":notes.get(note,{}).get("title",str(note)),"scene":"unknown","ts":None,"request_id":None,"action":"provided_history"} for note in history],
               "stage_enabled": {name:bool(feed["stages"][name]["enabled"]) for name in ("preranking","ranking")},
               "overlap":{name:len(set(lists[field])&set(positive)) for name,field in (("dssm","dssm_top10"),("preranking","preranking_top10"),("ranking","ranking_top10"))}}
    for field,values in lists.items():
        example[field] = [{"note_idx":int(note),"title":notes.get(int(note),{}).get("title",str(note)),"rank_in_true":truth.get(int(note))} for note in values]
    return example
