"""Package actual fit-trained query/item heads into a frozen residual checkpoint.

This performs no optimization, vector/index export, evaluation or activation.
Every input is validated before a temporary output stage is created. The final
directory appears only after full/partial round-trip and source-hash checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
import tempfile

import numpy as np
import torch

from recall.residual_dssm import (
    AdaptedResidualDSSMModel, CONTENT_MODEL_FAMILY, PCA_KEYS, MODEL_FAMILY,
    model_from_checkpoint, save_residual_checkpoints,
)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha_value(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdefABCDEF" for char in value)):
        raise ValueError(f"Invalid {name} SHA256")
    return value.lower()


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _state(payload, name):
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must contain a state dictionary")
    state = payload.get("state_dict", payload)
    if (not isinstance(state, Mapping) or not state
            or any(not isinstance(key, str) or not torch.is_tensor(value)
                   for key, value in state.items())):
        raise ValueError(f"{name} has invalid state keys or tensors")
    return dict(state)


def _same(left, right, name):
    if set(left) != set(right):
        raise ValueError(f"{name}: state key sets differ")
    for key, value in left.items():
        other = right[key]
        if (value.dtype != other.dtype or value.shape != other.shape
                or not torch.equal(value, other)):
            raise ValueError(f"{name}: tensor differs: {key}")


def _finite_state(state, name):
    for key, tensor in state.items():
        if tensor.is_complex() or (tensor.is_floating_point()
                                   and not torch.isfinite(tensor).all()):
            raise ValueError(f"{name}: nonfinite or complex tensor: {key}")


def _identity(value, name):
    names = [value[key] for key in ("variant", "arm") if key in value]
    if (not names or any(not isinstance(item, str) or not item for item in names)
            or len(set(names)) != 1):
        raise ValueError(f"{name} requires an unambiguous variant or arm")
    return names[0]


def _no_test_read(value):
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {"official_test_read", "test_read", "official_test_seen", "test_accessed"}:
                if item is not False:
                    raise ValueError("Training lineage declares or ambiguously records test access")
            _no_test_read(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _no_test_read(item)


def _training_steps(heads, summary, plan, hashes):
    plan_fields = [heads[key] for key in ("training_plan_sha256", "plan_sha256")
                   if key in heads]
    if not plan_fields or any(_sha_value(v, "training plan") != hashes["training_plan"]
                              for v in plan_fields):
        raise ValueError("Actual heads do not bind the supplied training plan")
    if _sha_value(heads.get("data_contract_sha256"), "data contract") != hashes["data_contract"]:
        raise ValueError("Actual heads do not bind the supplied data contract")
    if summary.get("official_test_read") is not False:
        raise ValueError("Training summary must explicitly record official_test_read=false")
    _no_test_read(summary)
    _no_test_read(heads)
    identity = _identity(heads, "Actual heads")
    epoch = _positive(heads.get("epoch"), "Selected epoch")
    training = plan.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("Training plan requires a training object")
    if "beta" in training and (isinstance(training["beta"], bool) or training["beta"] != .5):
        raise ValueError("Training content beta must be .5")
    if "epochs" in training and epoch > _positive(training["epochs"], "Planned epochs"):
        raise ValueError("Selected epoch exceeds the training plan")
    for key in ("training_plan_sha256", "plan_sha256", "data_contract_sha256"):
        if key in summary:
            expected = hashes["data_contract"] if key == "data_contract_sha256" else hashes["training_plan"]
            if _sha_value(summary[key], key) != expected:
                raise ValueError("Training summary source hash mismatch")
    trials = summary.get("trials")
    if not isinstance(trials, list) or any(not isinstance(row, Mapping) for row in trials):
        raise ValueError("Training summary requires trials records")
    selected = {}
    for row in trials:
        row_identity = _identity(row, "Training record")
        row_epoch = _positive(row.get("epoch"), "Training record epoch")
        if row_identity != identity or row_epoch > epoch:
            continue
        if row_epoch in selected:
            raise ValueError("Duplicate selected-arm epoch in training summary")
        selected[row_epoch] = row
    if set(selected) != set(range(1, epoch + 1)):
        raise ValueError("Selected arm must have continuous training epochs 1..selected epoch")
    steps = []
    for current in range(1, epoch + 1):
        row = selected[current]
        values = [_positive(row[key], key) for key in ("updates", "batches") if key in row]
        if not values or len(set(values)) != 1:
            raise ValueError("Each epoch must bind positive, consistent optimizer updates/batches")
        if "updates_per_epoch" in training:
            if values[0] != _positive(training["updates_per_epoch"], "Planned updates per epoch"):
                raise ValueError("Training step count differs from the frozen plan")
        _sha_value(row.get("weights_sha256"), "Epoch weights")
        steps.append(values[0])
    if _sha_value(selected[epoch]["weights_sha256"], "Selected weights") != hashes["towers_checkpoint"]:
        raise ValueError("Selected training summary weights do not match actual towers")
    return identity, epoch, steps


def _pretraining_steps(heads, summary, plan, identity, fine_tuning_steps, plan_path, register):
    """Bind optional pretraining blocks without relabeling them as fine-tuning epochs."""
    fields = ("pretrained_checkpoint_path", "pretrained_checkpoint_sha256",
              "pretraining_successful_updates", "total_successful_updates")
    config = plan.get("pretraining")
    if config is None:
        if any(key in heads for key in fields) or summary.get("pretraining_trials"):
            raise ValueError("Pretrained heads require a bound pretraining plan")
        return [], {}
    if not isinstance(config, Mapping):
        raise ValueError("Pretraining plan must be an object")
    _no_test_read(config)
    expected = _positive(config.get("steps"), "Planned pretraining steps")
    blocks = _positive(config.get("logical_blocks"), "Pretraining logical blocks")
    arm = heads.get("experiment_arm")
    if not isinstance(arm, str) or not arm:
        raise ValueError("Pretrained heads require an experiment_arm")
    if summary.get("optimizer_reset_between_phases") is not True:
        raise ValueError("Pretraining must bind the optimizer reset before fine-tuning")
    records = summary.get("pretraining_trials")
    if not isinstance(records, list) or any(not isinstance(row, Mapping) for row in records):
        raise ValueError("Pretraining requires logical-block records")
    selected = {}
    for row in records:
        if (_identity(row, "Pretraining record") != identity
                or row.get("experiment_arm") != arm):
            continue
        block = _positive(row.get("block"), "Pretraining block")
        if block in selected:
            raise ValueError("Duplicate pretraining block")
        selected[block] = row
    if set(selected) != set(range(1, blocks + 1)):
        raise ValueError("Pretraining blocks must be continuous and complete")
    steps = []
    for block in range(1, blocks + 1):
        row = selected[block]
        counts = [_positive(row[key], key) for key in ("updates", "batches") if key in row]
        if not counts or len(set(counts)) != 1:
            raise ValueError("Pretraining block updates/batches disagree")
        _sha_value(row.get("weights_sha256"), "Pretraining weights")
        steps.append(counts[0])
    total = sum(steps)
    if total != expected or _positive(heads.get("pretraining_successful_updates"),
                                     "Recorded pretraining updates") != total:
        raise ValueError("Pretraining updates differ from the frozen plan")
    if _positive(heads.get("total_successful_updates"), "Total updates") != total + sum(fine_tuning_steps):
        raise ValueError("Total updates omit or double-count a training stage")
    paths = config.get("checkpoint_paths")
    if not isinstance(paths, Mapping) or not isinstance(paths.get(arm), str):
        raise ValueError("Pretraining checkpoint path must be declared in the plan")
    declared = heads.get("pretrained_checkpoint_path")
    if not isinstance(declared, str) or not declared:
        raise ValueError("Actual heads must bind their pretraining checkpoint path")
    def resolve(value):
        path = Path(value)
        return (path if path.is_absolute() else plan_path.parent / path).resolve()
    checkpoint = resolve(declared)
    if checkpoint != resolve(paths[arm]):
        raise ValueError("Actual pretraining path differs from the frozen plan")
    actual_sha = register("pretraining_checkpoint", checkpoint)
    if (_sha_value(heads.get("pretrained_checkpoint_sha256"), "Pretraining checkpoint") != actual_sha
            or _sha_value(selected[blocks]["weights_sha256"], "Last pretraining block") != actual_sha):
        raise ValueError("Pretraining checkpoint and training records disagree")
    return steps, {
        "content_pretraining_checkpoint_sha256": actual_sha,
        "content_pretraining_steps_by_block": steps,
        "content_pretraining_gradient_updates": total,
        "content_finetuning_gradient_updates": sum(fine_tuning_steps),
    }


def _heads(payload):
    expected = {"net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias"}
    result = {}
    input_dim = None
    for role in ("query", "item"):
        state = payload.get(role)
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError(f"Actual {role} head must contain all four net.0/2 tensors")
        if any(not torch.is_tensor(v) or v.dtype != torch.float32
               or not torch.isfinite(v).all() for v in state.values()):
            raise ValueError(f"Actual {role} head must contain finite float32 tensors")
        weight = state["net.0.weight"]
        if weight.ndim != 2 or weight.shape[0] != 256 or weight.shape[1] not in (128, 768):
            raise ValueError("Unsupported actual content-head input dimension")
        current_dim = int(weight.shape[1])
        if input_dim is not None and current_dim != input_dim:
            raise ValueError("Query/item heads must use the same input dimension")
        input_dim = current_dim
        shapes = {"net.0.weight": (256, input_dim), "net.0.bias": (256,),
                  "net.2.weight": (128, 256), "net.2.bias": (128,)}
        if any(tuple(state[key].shape) != shape for key, shape in shapes.items()):
            raise ValueError(f"Invalid actual {role} head tensor shape")
        prefix = "content_query_adapter." if role == "query" else "content_item_adapter."
        result.update({prefix + key.removeprefix("net."): value for key, value in state.items()})
    return input_dim, result


def _pca_sources(plan, plan_path, base, register):
    sources = {}
    for name in ("source_sha256", "input_sha256"):
        values = plan.get(name, {})
        if not isinstance(values, Mapping):
            raise ValueError(f"{name} must be a path-to-SHA mapping")
        for path, digest in values.items():
            if not isinstance(path, str):
                raise ValueError("Plan input paths must be strings")
            if Path(path).name not in {"mean.npy", "components.npy"}:
                continue
            resolved = (Path(path) if Path(path).is_absolute() else plan_path.parent / path).resolve()
            expected = _sha_value(digest, "PCA source")
            if resolved in sources and sources[resolved] != expected:
                raise ValueError("Conflicting PCA source hashes")
            sources[resolved] = expected
    pairs = {}
    for name, buffer in (("mean.npy", base.residual_pca_mean),
                         ("components.npy", base.residual_pca_components)):
        matches = [path for path in sources if path.name == name]
        if len(matches) != 1:
            raise ValueError("Plan must uniquely bind mean.npy and components.npy")
        path = matches[0]
        digest = register("pca_mean" if name == "mean.npy" else "pca_components", path)
        if digest != sources[path]:
            raise ValueError("Frozen PCA source SHA mismatch")
        values = np.load(path, allow_pickle=False)
        expected = buffer.detach().cpu().numpy()
        if (not isinstance(values, np.ndarray) or values.dtype != np.float32
                or values.shape != expected.shape or not np.isfinite(values).all()
                or not np.array_equal(values, expected)):
            raise ValueError("Plan PCA values differ from the base residual buffers")
        pairs[name] = path
    if pairs["mean.npy"].parent != pairs["components.npy"].parent:
        raise ValueError("PCA mean/components must be bound from the same source directory")


def export_content_adapter_release(*, base_checkpoint, original_learned_checkpoint,
                                   towers_checkpoint, training_plan, data_contract,
                                   training_summary, destination) -> Path:
    """Export one actual trained pair of heads; return its final release manifest."""
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    sources = {}

    def register(role, path):
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = _sha(path)
        sources[role] = {"path": str(path), "sha256": digest}
        return digest

    paths = {name: Path(path).resolve() for name, path in {
        "base_checkpoint": base_checkpoint,
        "original_learned_checkpoint": original_learned_checkpoint,
        "towers_checkpoint": towers_checkpoint, "training_plan": training_plan,
        "data_contract": data_contract, "training_summary": training_summary,
    }.items()}
    hashes = {name: register(name, path) for name, path in paths.items()}
    payload = torch.load(paths["base_checkpoint"], map_location="cpu", weights_only=False)
    state = _state(payload, "Base residual")
    meta = payload.get("meta") if isinstance(payload, Mapping) else None
    if not meta:
        sidecar = paths["base_checkpoint"].with_suffix(".meta.json")
        register("base_metadata", sidecar)
        meta = _json(sidecar)
    if (not isinstance(meta, Mapping) or meta.get("model_family") != MODEL_FAMILY
            or meta.get("tower") is not None or meta.get("residual_weight") != .25):
        raise ValueError("Base must be a complete legacy residual_v1 checkpoint with weight .25")
    if _sha_value(meta.get("source_model_sha256"), "Frozen learned source") != hashes["original_learned_checkpoint"]:
        raise ValueError("Base does not bind the supplied original learned checkpoint")
    learned = _state(torch.load(paths["original_learned_checkpoint"], map_location="cpu",
                                weights_only=False), "Original learned")
    _finite_state(state, "Base residual")
    _finite_state(learned, "Original learned")
    _same({key: value for key, value in state.items() if key not in PCA_KEYS},
          learned, "Frozen learned/base identity")
    base = model_from_checkpoint({"state_dict": state, "meta": dict(meta)})
    if base.checkpoint_is_partial or float(base.residual_weight) != .25:
        raise ValueError("A partial or non-.25 source cannot produce a complete release")
    # The shared factory must not silently change any source dtype or value.
    _same(base.state_dict(), state, "Base factory round-trip")
    heads = torch.load(paths["towers_checkpoint"], map_location="cpu", weights_only=False)
    if not isinstance(heads, Mapping):
        raise ValueError("Actual towers must be a query/item checkpoint container")
    plan = _json(paths["training_plan"])
    contract = _json(paths["data_contract"])
    summary = _json(paths["training_summary"])
    _no_test_read(contract)
    identity, epoch, steps = _training_steps(heads, summary, plan, hashes)
    pretraining_steps, pretraining_meta = _pretraining_steps(
        heads, summary, plan, identity, steps, paths["training_plan"], register)
    input_dim, mapped = _heads(heads)
    _pca_sources(plan, paths["training_plan"], base, register)
    model = AdaptedResidualDSSMModel(
        meta.get("scene", "rec"), meta["cat_vocabs"], int(meta["item_dense_dim"]),
        int(meta["user_dense_dim"]), preprocessing=meta.get("numeric_preprocessing"),
        pca_mean=base.residual_pca_mean, pca_components=base.residual_pca_components,
        residual_weight=.25, content_input_dim=input_dim, content_beta=.5,
        mask_sequence_padding=bool(base.seq_mask_padding),
        mask_user_profile=bool(base.user_mask_profile),
    )
    merged = dict(state)
    merged.update({name: model.state_dict()[name] for name in ("content_beta", "content_version")})
    merged.update(mapped)
    model.load_state_dict(merged, strict=True)
    model.eval()
    _same({key: model.state_dict()[key] for key in state}, state, "Preserved base tensors")
    _same({key: model.state_dict()[key] for key in mapped}, mapped, "Actual adapter tensors")
    # Each optimizer update affects both towers and is counted once.
    total_steps = sum(steps) + sum(pretraining_steps)
    update_scope = ("content_adapter_pretraining_and_finetuning" if pretraining_steps
                    else "content_adapter_stage_only")
    release_meta = {
        **dict(meta), "gradient_updates": total_steps,
        "gradient_update_scope": update_scope, "packaging_gradient_updates": 0,
        "base_recorded_gradient_updates": meta.get("gradient_updates"),
        "frozen_learned_checkpoint_sha256": hashes["original_learned_checkpoint"],
        "actual_training_checkpoint_sha256": hashes["towers_checkpoint"],
        "content_training_plan_sha256": hashes["training_plan"],
        "content_training_data_contract_sha256": hashes["data_contract"],
        "content_training_summary_sha256": hashes["training_summary"],
        "content_training_identity": identity, "content_training_epoch": epoch,
        "content_training_steps_by_epoch": steps,
        **pretraining_meta,
    }
    # Validate the same complete family contract the shared factory will load.
    model_from_checkpoint({"state_dict": model.state_dict(),
        "meta": {**release_meta, "model_family": CONTENT_MODEL_FAMILY,
                 **model.content_metadata()}})
    json.dumps(release_meta, allow_nan=False)

    def verify_sources():
        for entry in sources.values():
            if _sha(entry["path"]) != entry["sha256"]:
                raise ValueError("A bound release input changed while packaging")

    verify_sources()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="." + destination.name + "-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "release"
        saved_meta = save_residual_checkpoints(model, release_meta, staged)
        full = staged / "dssm_rec_residual.pt"
        loaded = model_from_checkpoint(full)
        _same(loaded.state_dict(), model.state_dict(), "Exported full round-trip")
        expected_full_sha = _sha(full)
        for role in ("user", "item"):
            path = staged / f"dssm_rec_residual_{role}_tower.pt"
            part = torch.load(path, map_location="cpu", weights_only=False)
            partial = model_from_checkpoint(path, allow_partial=True)
            if (not partial.checkpoint_is_partial or part["meta"].get("tower") != role
                    or part["meta"].get("exported_full_checkpoint_sha256") != expected_full_sha):
                raise ValueError("Exported partial role/full-checkpoint binding mismatch")
            _same(part["state_dict"],
                  {key: loaded.state_dict()[key] for key in part["state_dict"]},
                  "Serialized partial/full identity")
            _same(part["state_dict"],
                  {key: partial.state_dict()[key] for key in part["state_dict"]},
                  "Exported active partial round-trip")
        verify_sources()
        exports = {path.name: {"sha256": _sha(path),
                   "role": "metadata" if path.suffix == ".json" else
                           ("user" if "_user_tower" in path.name else
                            "item" if "_item_tower" in path.name else "full")}
                   for path in sorted(staged.iterdir())}
        manifest = {
            "version": 1, "status": "packaged_not_deployed", "model_family": CONTENT_MODEL_FAMILY,
            "architecture": model.content_metadata(), "retrieval_dim": 256, "learned_weight": .25,
            "training_identity": identity, "epoch": epoch,
            "gradient_updates": total_steps, "gradient_update_scope": update_scope,
            "optimizer_steps_by_epoch": steps, "packaging_gradient_updates": 0,
            **pretraining_meta,
            "sources": sources, "exported_files": exports,
            "exported_full_checkpoint_sha256": saved_meta["exported_full_checkpoint_sha256"],
            "input_hashes_verified_unchanged": True, "training_performed_by_exporter": False,
            "index_created": False, "active_modified": False, "deployed": False,
        }
        (staged / "release_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        if destination.exists():
            raise FileExistsError(destination)
        staged.rename(destination)
    return destination / "release_manifest.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-checkpoint", "original-learned-checkpoint", "towers-checkpoint",
                 "training-plan", "data-contract", "training-summary", "destination"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    path = export_content_adapter_release(**vars(args))
    manifest = _json(path)
    print(json.dumps({"manifest": str(path), "model_family": manifest["model_family"],
                      "gradient_updates": manifest["gradient_updates"],
                      "packaging_gradient_updates": 0, "activated": False}))


if __name__ == "__main__":
    main()
