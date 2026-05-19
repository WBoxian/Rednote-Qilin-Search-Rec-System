from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

DEFAULT_INTENT_PATTERNS: dict[str, tuple[str, ...]] = {
    "tutorial": ("怎么", "如何", "教程", "攻略", "步骤"),
    "compare": ("对比", "区别", "还是", "哪个好", "选哪"),
    "recommend": ("推荐", "种草", "买什么", "哪款", "适合"),
    "review": ("测评", "评测", "体验", "开箱", "值不值"),
    "health": ("胃药", "止吐", "头疼", "感冒", "症状", "药"),
    "fashion": ("穿搭", "搭配", "显瘦", "口红", "香水", "包"),
}

DEFAULT_SERVICE_STOPWORDS: set[str] = {
    "怎么",
    "如何",
    "一下",
    "一个",
    "一些",
    "可以",
    "有没有",
    "请问",
    "帮我",
    "给我",
    "推荐一下",
}

DEFAULT_INTENT_HINTS: dict[str, str] = {
    "tutorial": "教程",
    "compare": "对比",
    "recommend": "推荐",
    "review": "测评",
    "health": "症状",
    "fashion": "搭配",
}

DEFAULT_QUERY_ALIAS_MAP: dict[str, str] = {
    "蘋果": "苹果",
    "手機": "手机",
    "電腦": "电脑",
    "東京": "东京",
    "东jing": "东京",
    "dongjing": "东京",
    "beijing": "北京",
    "shanghai": "上海",
    "guangzhou": "广州",
    "shenzhen": "深圳",
    "hangzhou": "杭州",
    "xian": "西安",
}

TRADITIONAL_CHAR_MAP: dict[str, str] = {
    "蘋": "苹",
    "國": "国",
    "業": "业",
    "電": "电",
    "腦": "脑",
    "機": "机",
    "東": "东",
    "臺": "台",
    "灣": "湾",
    "書": "书",
    "網": "网",
    "軟": "软",
    "體": "体",
    "貓": "猫",
    "車": "车",
    "門": "门",
    "頭": "头",
    "風": "风",
    "藥": "药",
}

TIME_PHRASE_MAP: dict[str, str] = {
    "明早": "明天早上",
    "今早": "今天早上",
    "昨晚": "昨天晚上",
    "今晚": "今天晚上",
    "明晚": "明天晚上",
    "后天": "後天",
}

UNIT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?P<num>\d+(?:\.\d+)?)\s*(公里|千米)"), "{num}km"),
    (re.compile(r"(?P<num>\d+(?:\.\d+)?)\s*(米)"), "{num}m"),
    (re.compile(r"(?P<num>\d+(?:\.\d+)?)\s*(公斤|千克)"), "{num}kg"),
    (re.compile(r"(?P<num>\d+(?:\.\d+)?)\s*(克)"), "{num}g"),
    (re.compile(r"(?P<num>\d+(?:\.\d+)?)\s*(毫升|ml|ML)"), "{num}ml"),
    (re.compile(r"(?P<num>\d+(?:\.\d+)?)\s*(升|l|L)"), "{num}l"),
)

EMOJI_OR_SYMBOL_RE = re.compile(
    "["
    "\u2600-\u27BF"
    "\U0001F300-\U0001FAFF"
    "]",
    flags=re.UNICODE,
)
CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
PUNCT_RE = re.compile(r"[，。！？、；：,.!?:;~`·•…]+")
SPACE_RE = re.compile(r"\s+")
ALNUM_RE = re.compile(r"[a-z0-9]+")
TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+")

MAX_QUERY_CHARS = 100
MAX_SEGMENT_TERM_LEN = 8
MIN_TERM_FREQ = 2


@dataclass(slots=True)
class SearchQueryResources:
    query_catalog: tuple[str, ...]
    normalized_catalog: tuple[str, ...]
    normalized_map: dict[str, str]
    query_scores: dict[str, float]
    term_scores: dict[str, float]
    term_bucket_map: dict[str, tuple[str, ...]]
    dynamic_stopwords: frozenset[str]

    @property
    def norm_catalog(self) -> tuple[str, ...]:
        return self.normalized_catalog

    @property
    def norm_map(self) -> dict[str, str]:
        return self.normalized_map

    @property
    def term_score_map(self) -> dict[str, float]:
        return self.term_scores


def _strip_control_chars(text: str) -> str:
    return CONTROL_RE.sub(" ", text)


def _apply_phrase_aliases(text: str) -> str:
    for src, dst in DEFAULT_QUERY_ALIAS_MAP.items():
        if src in text:
            text = text.replace(src, dst)
    return text


def _convert_traditional_chars(text: str) -> str:
    return "".join(TRADITIONAL_CHAR_MAP.get(char, char) for char in text)


def _normalize_time_phrases(text: str) -> str:
    for src, dst in TIME_PHRASE_MAP.items():
        if src in text:
            text = text.replace(src, dst)
    return text.replace("後天", "后天")


def _normalize_numeric_forms(text: str) -> str:
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    for pattern, template in UNIT_PATTERNS:
        text = pattern.sub(lambda match: template.format(num=match.group("num")), text)
    return text


def _filter_noise_chars(text: str) -> str:
    text = EMOJI_OR_SYMBOL_RE.sub(" ", text)
    text = PUNCT_RE.sub(" ", text)
    text = text.replace("“", " ").replace("”", " ").replace('"', " ")
    text = text.replace("‘", " ").replace("’", " ").replace("'", " ")
    text = text.replace("/", " ").replace("\\", " ").replace("|", " ")
    text = text.replace("(", " ").replace(")", " ").replace("[", " ").replace("]", " ")
    return text


def normalize_query_text(query: str | None) -> str:
    raw = unicodedata.normalize("NFKC", str(query or ""))
    raw = _strip_control_chars(raw).strip().lower()
    raw = _apply_phrase_aliases(raw)
    raw = _convert_traditional_chars(raw)
    raw = _normalize_time_phrases(raw)
    raw = _normalize_numeric_forms(raw)
    raw = _filter_noise_chars(raw)
    raw = SPACE_RE.sub(" ", raw).strip()
    if len(raw) > MAX_QUERY_CHARS:
        raw = raw[:MAX_QUERY_CHARS].rstrip()
    return raw


def bounded_edit_distance(left: str, right: str, max_cost: int = 2) -> int:
    if left == right:
        return 0
    if abs(len(left) - len(right)) > max_cost:
        return max_cost + 1
    if not left:
        return min(len(right), max_cost + 1)
    if not right:
        return min(len(left), max_cost + 1)
    prev = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        curr = [i]
        row_min = curr[0]
        for j, right_char in enumerate(right, start=1):
            replace = prev[j - 1] + (left_char != right_char)
            delete = prev[j] + 1
            insert = curr[j - 1] + 1
            cost = min(replace, delete, insert)
            curr.append(cost)
            row_min = min(row_min, cost)
        if row_min > max_cost:
            return max_cost + 1
        prev = curr
    return prev[-1]


def split_query_chunks(query: str) -> list[str]:
    normalized = normalize_query_text(query)
    return [match.group(0) for match in TOKEN_RE.finditer(normalized)]


def seed_query_terms(query: str) -> list[str]:
    normalized = normalize_query_text(query)
    if not normalized:
        return []
    terms: list[str] = []
    compact = normalized.replace(" ", "")
    chunks = split_query_chunks(normalized)
    for chunk in chunks:
        if chunk not in terms:
            terms.append(chunk)
        if ALNUM_RE.fullmatch(chunk):
            continue
        if len(chunk) >= 4:
            prefix = chunk[:2]
            suffix = chunk[-2:]
            if prefix not in terms:
                terms.append(prefix)
            if suffix not in terms:
                terms.append(suffix)
    if compact and compact not in terms:
        terms.append(compact)
    return terms


def _term_weight(term: str, term_scores: dict[str, float]) -> float:
    base = term_scores.get(term, 0.0)
    return math.log1p(base) + min(len(term), 6) * 0.15


def _best_chinese_segmentation(chunk: str, term_scores: dict[str, float]) -> list[str]:
    if len(chunk) <= 2:
        return [chunk]
    best: list[tuple[float, list[str]] | None] = [None] * (len(chunk) + 1)
    best[0] = (0.0, [])
    for idx in range(len(chunk)):
        state = best[idx]
        if state is None:
            continue
        score, path = state
        max_len = min(MAX_SEGMENT_TERM_LEN, len(chunk) - idx)
        for size in range(1, max_len + 1):
            term = chunk[idx : idx + size]
            known = term in term_scores or size == len(chunk) or (size >= 2 and idx + size == len(chunk))
            if not known:
                continue
            term_score = score + _term_weight(term, term_scores)
            next_idx = idx + size
            prev_state = best[next_idx]
            if prev_state is None or term_score > prev_state[0]:
                best[next_idx] = (term_score, path + [term])
    final_state = best[len(chunk)]
    if final_state is None:
        return [chunk]
    return final_state[1]


def segment_query_terms(query: str, resources: SearchQueryResources | None = None) -> list[str]:
    normalized = normalize_query_text(query)
    if not normalized:
        return []
    term_scores = resources.term_scores if resources else {}
    stopwords = resources.dynamic_stopwords if resources else frozenset(DEFAULT_SERVICE_STOPWORDS)
    ordered_terms: list[str] = []
    seen: set[str] = set()
    compact = normalized.replace(" ", "")

    for chunk in split_query_chunks(normalized):
        if ALNUM_RE.fullmatch(chunk):
            if chunk not in stopwords and chunk not in seen:
                ordered_terms.append(chunk)
                seen.add(chunk)
            continue
        for term in _best_chinese_segmentation(chunk, term_scores):
            if len(term) == 1 and term not in term_scores:
                continue
            if term in stopwords or term in seen:
                continue
            ordered_terms.append(term)
            seen.add(term)

    if compact and compact not in seen:
        ordered_terms.append(compact)
        seen.add(compact)

    return ordered_terms


def detect_query_intents(query: str, patterns: dict[str, tuple[str, ...]] | None = None) -> list[str]:
    normalized = normalize_query_text(query)
    if not normalized:
        return []
    pattern_map = patterns or DEFAULT_INTENT_PATTERNS
    intents = [intent for intent, phrases in pattern_map.items() if any(phrase in normalized for phrase in phrases)]
    return intents[:3]


def _bucket_key(term: str) -> str:
    if not term:
        return ""
    if ALNUM_RE.fullmatch(term):
        return term[:2]
    return term[:1]


def build_search_query_resources(queries: Iterable[str]) -> SearchQueryResources:
    query_counter: Counter[str] = Counter()
    normalized_map: dict[str, str] = {}
    term_counter: Counter[str] = Counter()

    for query in queries:
        normalized = normalize_query_text(query)
        if not normalized:
            continue
        query_counter[normalized] += 1
        normalized_map.setdefault(normalized, normalized)
        for term in seed_query_terms(normalized):
            if len(term) >= 2:
                term_counter[term] += 1

    dynamic_stopwords = {
        term
        for term, freq in term_counter.items()
        if freq >= 200 and len(term) <= 2 and term not in {"jk", "jk裙", "jk制服"}
    }

    term_scores = {
        term: float(freq)
        for term, freq in term_counter.items()
        if freq >= MIN_TERM_FREQ and term not in dynamic_stopwords
    }

    bucket_map: dict[str, list[str]] = defaultdict(list)
    for term in term_scores:
        bucket_map[_bucket_key(term)].append(term)
    for terms in bucket_map.values():
        terms.sort(key=lambda value: (-term_scores.get(value, 0.0), -len(value), value))

    normalized_catalog = tuple(sorted(query_counter, key=lambda value: (-query_counter[value], -len(value), value)))
    query_scores = {query: float(freq) for query, freq in query_counter.items()}

    return SearchQueryResources(
        query_catalog=normalized_catalog,
        normalized_catalog=normalized_catalog,
        normalized_map=normalized_map,
        query_scores=query_scores,
        term_scores=term_scores,
        term_bucket_map={key: tuple(values) for key, values in bucket_map.items()},
        dynamic_stopwords=frozenset(DEFAULT_SERVICE_STOPWORDS | dynamic_stopwords),
    )


def correct_query_term(
    term: str,
    resources: SearchQueryResources | None = None,
    recent_queries: Iterable[str] | None = None,
) -> str:
    normalized = normalize_query_text(term)
    if not normalized or resources is None:
        return normalized
    if normalized in resources.term_scores:
        return normalized
    if normalized.isdigit() or ALNUM_RE.fullmatch(normalized):
        return normalized
    if len(normalized) <= 2:
        return normalized

    bucket = resources.term_bucket_map.get(_bucket_key(normalized), ())
    best_term = normalized
    best_score = 0.0
    for candidate in bucket[:80]:
        if abs(len(candidate) - len(normalized)) > 1:
            continue
        edit_cost = bounded_edit_distance(normalized, candidate, max_cost=1)
        if edit_cost > 1:
            continue
        overlap = len(set(normalized) & set(candidate)) / max(len(set(normalized) | set(candidate)), 1)
        score = overlap + math.log1p(resources.term_scores.get(candidate, 0.0)) * 0.05 - edit_cost * 0.35
        if score > best_score:
            best_score = score
            best_term = candidate
    return best_term if best_score >= 0.75 else normalized


def _query_candidate_score(
    normalized_query: str,
    candidate: str,
    query_scores: dict[str, float],
    recent_queries: set[str],
) -> float:
    if not candidate:
        return float("-inf")
    length_gap = abs(len(candidate) - len(normalized_query))
    if length_gap > 2:
        return float("-inf")
    edit_limit = 1 if len(normalized_query) <= 6 else 2
    edit_cost = bounded_edit_distance(normalized_query, candidate, max_cost=edit_limit)
    if edit_cost > edit_limit:
        return float("-inf")
    left = set(seed_query_terms(normalized_query))
    right = set(seed_query_terms(candidate))
    token_overlap = len(left & right) / max(len(left | right), 1)
    char_overlap = len(set(normalized_query) & set(candidate)) / max(len(set(normalized_query) | set(candidate)), 1)
    exact_prefix = 1.0 if candidate.startswith(normalized_query) or normalized_query.startswith(candidate) else 0.0
    recent_bonus = 0.25 if candidate in recent_queries else 0.0
    hot_bonus = math.log1p(query_scores.get(candidate, 0.0)) * 0.05
    return token_overlap * 1.1 + char_overlap * 0.6 + exact_prefix * 0.1 + recent_bonus + hot_bonus - edit_cost * 0.45


def preprocess_search_query(
    query: str | None,
    *,
    resources: SearchQueryResources | None = None,
    recent_queries: Iterable[str] | None = None,
) -> dict[str, object]:
    normalized_query = normalize_query_text(query)
    recent_norm_queries = {normalize_query_text(item) for item in (recent_queries or []) if normalize_query_text(item)}

    corrected_query = normalized_query
    if normalized_query and resources is not None:
        direct_hit = resources.normalized_map.get(normalized_query)
        if direct_hit:
            corrected_query = direct_hit
        elif len(normalized_query) >= 3:
            bucket = resources.term_bucket_map.get(_bucket_key(normalized_query), ())
            catalog_candidates = list(resources.normalized_catalog[:120])
            candidate_pool = []
            seen_candidates: set[str] = set()
            for candidate in list(bucket) + catalog_candidates:
                normalized_candidate = normalize_query_text(candidate)
                if not normalized_candidate or normalized_candidate in seen_candidates:
                    continue
                if normalized_query.isascii() and normalized_query.isalnum():
                    if not normalized_candidate.isascii():
                        continue
                seen_candidates.add(normalized_candidate)
                candidate_pool.append(normalized_candidate)
            best_query = normalized_query
            best_score = 0.0
            for candidate in candidate_pool[:180]:
                score = _query_candidate_score(
                    normalized_query,
                    candidate,
                    resources.query_scores,
                    recent_norm_queries,
                )
                if score > best_score:
                    best_query = candidate
                    best_score = score
            if best_score >= 1.1:
                corrected_query = best_query

    query_terms = segment_query_terms(corrected_query or normalized_query, resources=resources)
    corrected_terms: list[str] = []
    seen_terms: set[str] = set()
    for term in query_terms:
        corrected_term = correct_query_term(term, resources=resources, recent_queries=recent_norm_queries)
        if corrected_term and corrected_term not in seen_terms:
            corrected_terms.append(corrected_term)
            seen_terms.add(corrected_term)

    compact = (corrected_query or normalized_query).replace(" ", "")
    if compact and compact not in seen_terms:
        corrected_terms.append(compact)
        seen_terms.add(compact)

    intents = detect_query_intents(corrected_query or normalized_query)
    stopwords = resources.dynamic_stopwords if resources else frozenset(DEFAULT_SERVICE_STOPWORDS)
    high_value_terms = [term for term in corrected_terms if term not in stopwords]
    if not high_value_terms:
        high_value_terms = corrected_terms[:]

    must_terms = high_value_terms[:3]
    core_terms = high_value_terms[:4]
    optional_terms = high_value_terms[4:8]
    service_terms = high_value_terms[:]
    for intent in intents:
        hint = DEFAULT_INTENT_HINTS.get(intent)
        if hint and hint not in service_terms:
            service_terms.append(hint)

    service_query = " ".join(service_terms[:8]).strip()
    return {
        "input_query": str(query or "").strip(),
        "normalized_query": normalized_query,
        "corrected_query": corrected_query if corrected_query != normalized_query else "",
        "resolved_query": corrected_query or normalized_query,
        "terms": corrected_terms,
        "core_terms": core_terms,
        "optional_terms": optional_terms,
        "must_terms": must_terms,
        "intents": intents,
        "service_query": service_query,
    }
