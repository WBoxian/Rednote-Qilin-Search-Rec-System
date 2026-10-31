"""Load an explicitly promoted, content-bound serving bundle for one scene."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock

from backend.online.item_features import STATIC_ITEM_FEATURES
from backend.online.ranking.stack import RankStackPredictor, validate_rank_prior
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
    diagnostic = manifest.get("evaluation", {}).get("validation_snapshot")
    if diagnostic is not None:
        if not isinstance(diagnostic, dict) or set(diagnostic) != {"path", "sha256"}:
            raise ValueError("Invalid bound diagnostic snapshot")
        require(diagnostic["path"])
    if manifest.get("note_metadata_index"):
        metadata_index = resolve_asset(root, manifest["note_metadata_index"])
        require(metadata_index)
        require(metadata_index.with_suffix(metadata_index.suffix+'.manifest.json'))
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
    ranking_strategy = manifest.get("ranking_strategy", "learned")
    if ranking_strategy not in {"learned", "recall_only"}:
        raise ValueError("Unsupported ranking strategy")
    if ranking_strategy == "recall_only" and (int(manifest["input_cap"]) != final_topk or int(manifest["coarse_cap"]) != final_topk or manifest.get("schema_enabled") is not False):
        raise ValueError("Recall-only serving must retain the full recall pool and explicitly disable learned schema")
    if int(policy.get("rrf_k", 60)) < 1:
        raise ValueError("Invalid reciprocal-rank smoothing")
    context_policy = policy.get("context_policy")
    if context_policy is not None and (scene != "rec" or context_policy != "filter_history" or strategy != "weighted_rrf" or
            weights != {"ann": 1., "swing": 2., "usercf": 1.} or int(policy.get("rrf_k", 60)) != 60 or route_topk != 1000):
        raise ValueError("Unsupported frozen request-context fusion contract")
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
    diagnostic = manifest.get("evaluation", {}).get("validation_snapshot")
    if diagnostic is not None and assets.get(resolve_asset(root, diagnostic["path"]).relative_to(root.resolve()).as_posix()) != diagnostic["sha256"]:
        raise ValueError("Diagnostic snapshot hash differs from the required asset binding")
    if manifest.get('note_metadata_index'):
        path = resolve_asset(root, manifest['note_metadata_index'])
        metadata = json.loads(path.with_suffix(path.suffix+'.manifest.json').read_text())
        source_hashes = {entry['name']: entry['sha256'] for entry in metadata.get('sources', [])}
        bound_sources = {Path(name).name: expected for name, expected in assets.items() if name.startswith('datasets/notes/') and name.endswith('.parquet')}
        if source_hashes != bound_sources or metadata.get('index_sha256') != assets.get(path.relative_to(root.resolve()).as_posix()):
            raise ValueError('Note metadata index does not bind the same static catalog')
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
    score_contract = dien.get("score_contract", "request_minmax_v1")
    from backend.online.ranking.inference import validate_inference_contract
    validate_inference_contract(dien.get("inference_contract"))
    if score_contract not in {"request_minmax_v1", "probability_v2"}:
        raise ValueError("Unsupported ranking score contract")
    if score_contract == "probability_v2" and metadata.get("trees", {}).get("chosen", {}).get("config", {}).get("kind") != "classifier":
        raise ValueError("Probability fusion requires a binary-classifier teacher")
    validate_rank_prior(dien.get("rank_prior"))
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
        return self._get_bm25_index().search(query, topk)

    def _get_bm25_index(self):
        from recall.lexical import BM25Index
        with self._lock:
            path = resolve_asset(self.root, self.manifest["bm25_index"])
            sidecar = path.with_suffix(path.suffix + ".manifest.json")
            bound_hash = self.manifest["assets"][path.relative_to(self.root.resolve()).as_posix()]
            verify_asset(path, bound_hash)
            verify_asset(sidecar, self.manifest["assets"][sidecar.relative_to(self.root.resolve()).as_posix()])
            if json.loads(sidecar.read_text()).get("index_sha256") != bound_hash:
                raise ValueError("BM25 sidecar differs from the bundle's bound index hash")
            index = getattr(self, "_lexical", None)
            if index is None:
                # Bound-byte validation above precedes reader schema/size/SQLite metadata checks.
                index = BM25Index(path, verify_hash=False)
                self._lexical = index
            return index

    def _get_note_metadata_index(self):
        from backend.online.note_metadata import NoteMetadataIndex
        with self._lock:
            index = getattr(self, '_note_metadata', None)
            if index is None:
                path = resolve_asset(self.root, self.manifest['note_metadata_index'])
                sidecar = path.with_suffix(path.suffix+'.manifest.json')
                expected = self.manifest['assets'][path.relative_to(self.root.resolve()).as_posix()]
                verify_asset(path, expected)
                verify_asset(sidecar, self.manifest['assets'][sidecar.relative_to(self.root.resolve()).as_posix()])
                if json.loads(sidecar.read_text()).get('index_sha256') != expected:
                    raise ValueError('Note metadata index differs from its bound hash')
                index = NoteMetadataIndex(path)
                self._note_metadata = index
            return index


def get_active_bundle(scene: str, tag: str, root: Path) -> ServingBundle | None:
    root = Path(root)
    path = root / "outputs" / "deploy" / scene / tag / "active_bundle.json"
    if not path.exists():
        return None
    manifest = json.loads(path.read_text())
    validate_manifest(manifest, scene, root, tag=tag)
    if scene == "search":
        from backend.online.shared_query_encoder import get_shared_query_encoder
        tokenizer, model = get_shared_query_encoder(resolve_asset(root, manifest["query_encoder_dir"]), manifest["query_encoder_revision"])
        if tokenizer is None or model is None:
            raise RuntimeError("Pinned query encoder failed before serving readiness")
    ranker = RankStackPredictor(scene, resolve_asset(root, manifest["rank_dir"]), root=root,
        query_encoder_dir=resolve_asset(root, manifest["query_encoder_dir"]) if scene == "search" else None,
        query_encoder_revision=manifest.get("query_encoder_revision"))
    bundle = ServingBundle(manifest, ranker, root)
    if scene == "search":
        bundle._get_bm25_index()
    if manifest.get('note_metadata_index'):
        bundle._get_note_metadata_index()
    return bundle
