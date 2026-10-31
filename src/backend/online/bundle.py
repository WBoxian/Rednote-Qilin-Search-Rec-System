"""Load an explicitly promoted, content-bound serving bundle for one scene."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock

from backend.online.item_features import STATIC_ITEM_FEATURES
from backend.online.ranking.stack import RankStackPredictor
from training.history import HISTORY_ORDER
from training.rank_stack import SEMANTIC_FEATURES


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_VERIFIED_ASSETS = set()
_ASSET_LOCK = RLock()


def verify_asset(path: Path, expected: str) -> None:
    """Process-local only: first load hashes bytes, subsequent loads bind full file stat."""
    path = path.resolve()
    if not isinstance(expected, str) or len(expected) != 64 or not path.is_file():
        raise ValueError(f"Serving bundle asset mismatch: {path}")
    before = path.stat()
    identity = (str(path), before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino, before.st_dev, expected)
    with _ASSET_LOCK:
        if identity in _VERIFIED_ASSETS:
            return
        actual = file_sha256(path)
        after = path.stat()
        if (after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino, after.st_dev) != identity[1:6] or actual != expected:
            raise ValueError(f"Serving bundle asset mismatch: {path}")
        _VERIFIED_ASSETS.add(identity)


def bundle_identity(manifest: dict) -> str:
    payload = {key: value for key, value in manifest.items() if key != "bundle_id"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def resolve_asset(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Bundle asset path must be a nonempty string")
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Bundle asset escapes the project root")
    return path


def required_asset_names(manifest: dict, root: Path) -> set[str]:
    """Every file consumed by current serving, not an arbitrary caller-provided subset."""
    root = Path(root).resolve()
    scene, tag = manifest["scene"], manifest["tag"]
    required = set()
    def require(path):
        required.add(resolve_asset(root, str(path)).relative_to(root).as_posix())
    rank = resolve_asset(root, manifest["rank_dir"])
    recall = resolve_asset(root, manifest["recall_dir"])
    for name in ("manifest.json", "gbdt.txt", "dien.pt"):
        require(rank / name)
    prefix = f"dssm_{scene}_{tag}"
    for suffix in ("user_tower.pt", "item_tower.pt"):
        require(recall / "models" / f"{prefix}_{suffix}")
    for suffix in ("item_map.json", "item_meta.json", "item_emb.bin", "row2note.npy", "ivfpq.faiss", "ivfpq_meta.json"):
        require(recall / "index" / f"{prefix}_{suffix}")
    for modality in ("text", "img"):
        for suffix in ("map.json", "bin"):
            require(root / "embeddings" / f"note_{modality}_emb" / f"note_{modality}.{suffix}".replace(".map.json", "_map.json"))
    for name in ("gender", "platform", "age", "location", "taxonomy1_id", "taxonomy2_id", "taxonomy3_id"):
        require(root / "features" / "vocab_dict" / f"{name}.pkl")
    for folder in ("notes", "user_feat", "search_train" if scene == "search" else "recommendation_train"):
        sources = sorted((root / "datasets" / folder).glob("*.parquet"))
        if not sources:
            raise ValueError(f"Missing bound serving source: {folder}")
        for source in sources:
            require(source)
    if scene == "search":
        lexical = resolve_asset(root, manifest["bm25_index"])
        require(lexical)
        require(lexical.with_suffix(lexical.suffix + ".manifest.json"))
        encoder = resolve_asset(root, manifest["query_encoder_dir"])
        files = [path for path in encoder.rglob("*") if path.is_file()]
        if not files:
            raise ValueError("Pinned query encoder has no files")
        for path in files:
            require(path)
    else:
        graph = resolve_asset(root, manifest["recall_policy"]["cf_graph_dir"])
        for name in ("graph.npz", "users.npy", "items.npy", "swing.npz", "meta.json"):
            require(graph / name)
    return required


def validate_manifest(manifest: dict, scene: str, root: Path, *, tag: str | None = None) -> None:
    if manifest.get("version") != 1 or manifest.get("scene") != scene:
        raise ValueError("Unsupported serving bundle version or scene")
    if manifest.get("tag") not in {"easy", "hard"} or (tag is not None and manifest["tag"] != tag):
        raise ValueError("Serving bundle tag mismatch")
    if manifest.get("bundle_id") != bundle_identity(manifest):
        raise ValueError("Serving bundle identity mismatch")
    if manifest.get("history_order") != HISTORY_ORDER:
        raise ValueError("Serving history order differs from the trained bundle")
    if manifest.get("selection_split") != "train_internal_validation":
        raise ValueError("Serving bundle must bind training-internal validation selection")
    if manifest.get("allow_dpr", True):
        raise ValueError("DPR is outside the serving contract")
    cutoff = manifest.get("training_cutoff_exclusive")
    if isinstance(cutoff, bool) or not isinstance(cutoff, int) or cutoff <= 0:
        raise ValueError("A positive complete-request training cutoff must be bound")
    policy = manifest.get("recall_policy", {})
    strategy = policy.get("strategy")
    if strategy not in ({"weighted_rrf", "protected_headfill"} if scene == "search" else {"weighted_rrf"}):
        raise ValueError("Unsupported frozen recall policy")
    weights = policy.get("weights", {})
    routes = {"ann", "bm25"} if scene == "search" else {"ann", "swing", "usercf"}
    if set(weights) != routes or any(not math.isfinite(float(value)) or float(value) < 0 for value in weights.values()) or not sum(weights.values()):
        raise ValueError("Invalid frozen recall route weights")
    final_topk, route_topk = int(policy.get("topk", 500)), int(policy.get("route_topk", 1000))
    if not (1 <= int(manifest["coarse_cap"]) <= int(manifest["input_cap"]) <= final_topk <= 1000 and 1 <= route_topk <= 1000):
        raise ValueError("Invalid serving candidate budgets")
    if int(policy.get("rrf_k", 60)) < 1:
        raise ValueError("Invalid reciprocal-rank smoothing")
    if strategy == "protected_headfill" and not 0 <= int(policy.get("head_cap", 100)) <= final_topk:
        raise ValueError("Invalid protected lexical head budget")
    if scene == "rec":
        config = policy.get("cf_config", {})
        if not isinstance(config.get("fold_in"), bool) or not 1 <= int(config.get("neighbors", 0)) <= 1000 or not math.isfinite(float(config.get("history_weight", -1))) or float(config["history_weight"]) < 0:
            raise ValueError("Invalid frozen collaborative-filtering configuration")
    if scene == "search":
        revision = manifest.get("query_encoder_revision")
        if not isinstance(revision, str) or not revision or not isinstance(manifest.get("query_encoder_dir"), str):
            raise ValueError("Search requires a pinned project-local query encoder")
        encoder = resolve_asset(root, manifest["query_encoder_dir"])
        if not (encoder / "config.json").is_file() or not (encoder / "tokenizer_config.json").is_file():
            raise ValueError("Pinned query encoder config/tokenizer files are missing")
        if not any((encoder / name).is_file() for name in ("tokenizer.json", "vocab.txt")) or not any((encoder / name).is_file() for name in ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json")):
            raise ValueError("Pinned query encoder weights/vocabulary are missing")
    required = required_asset_names(manifest, root)
    assets = manifest.get("assets", {})
    if not required.issubset(assets):
        raise ValueError(f"Serving bundle omits required asset bindings: {sorted(required-set(assets))}")
    # Check the small schema manifest before hashing gigabytes or constructing models.
    rank = resolve_asset(root, manifest["rank_dir"])
    metadata = json.loads((rank / "manifest.json").read_text())
    allowed = {*STATIC_ITEM_FEATURES, *SEMANTIC_FEATURES, *({"query_item_cosine"} if scene == "search" else set())}
    features = metadata.get("features", [])
    if metadata.get("scene") != scene or metadata.get("key") != ("search_idx" if scene == "search" else "request_idx") or not features or len(features) != len(set(features)) or set(features) - allowed:
        raise ValueError("Unsafe or incompatible ranking feature schema")
    dien = metadata.get("dien", {})
    if dien.get("numeric_features") != [*features, "coarse_score"] or dien.get("architecture", {}).get("inference_precision") != "fp32":
        raise ValueError("Fine ranking input/precision contract mismatch")
    for name, expected in assets.items():
        path = resolve_asset(root, name)
        verify_asset(path, expected)


@dataclass
class ServingBundle:
    manifest: dict
    ranker: RankStackPredictor
    root: Path
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    @property
    def bundle_id(self):
        return self.manifest["bundle_id"]

    @property
    def recall_dir(self):
        return resolve_asset(self.root, self.manifest["recall_dir"])

    @property
    def input_cap(self):
        return int(self.manifest["input_cap"])

    @property
    def coarse_cap(self):
        return int(self.manifest["coarse_cap"])

    def cf_candidates(self, user_idx, history, topk):
        from recall.cf_retrieval import CFGraphRecaller
        with self._lock:
            policy = self.manifest["recall_policy"]
            recaller = getattr(self, "_cf", None)
            if recaller is None:
                recaller = CFGraphRecaller(resolve_asset(self.root, policy["cf_graph_dir"]), policy["cf_config"])
                self._cf = recaller
            return recaller.score(user_idx, history, topk)

    def bm25_candidates(self, query, topk):
        from recall.lexical import BM25Index
        with self._lock:
            index = getattr(self, "_lexical", None)
            if index is None:
                index = BM25Index(resolve_asset(self.root, self.manifest["bm25_index"]))
                self._lexical = index
            return index.search(query, topk)


def get_active_bundle(scene: str, tag: str, root: Path) -> ServingBundle | None:
    root = Path(root)
    path = root / "outputs" / "deploy" / scene / tag / "active_bundle.json"
    if not path.exists():
        return None
    manifest = json.loads(path.read_text())
    validate_manifest(manifest, scene, root, tag=tag)
    ranker = RankStackPredictor(scene, resolve_asset(root, manifest["rank_dir"]), root=root,
        query_encoder_dir=resolve_asset(root, manifest["query_encoder_dir"]) if scene == "search" else None,
        query_encoder_revision=manifest.get("query_encoder_revision"))
    return ServingBundle(manifest, ranker, root)