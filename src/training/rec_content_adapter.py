"""Fit-only PCA128 content adaptation with explicit, checksum-bound data inputs.

Training follows the frozen dual-content recipe. It never evaluates retrieval,
reads validation/test feedback, changes the frozen learned tower or builds indexes.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from recall.residual_dssm import project_content, raw_history_text_mean
from training.history import HISTORY_ORDER, recent_history_ids

RECIPE = dict(epochs=3, batch_requests=512, seed=42, lr=.001,
              weight_decay=.0001, temperature=.07, beta=.5)
INPUTS = {"requests", "labels", "partitions", "split_manifest", "text_vectors",
          "text_map", "pca_mean", "pca_components", "pca_vectors", "item_ids"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def finite(name, value):
    valid = bool(torch.isfinite(value).all()) if torch.is_tensor(value) else np.isfinite(value).all()
    if not valid:
        raise ValueError("Nonfinite " + name)


def configure(seed=42):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ContentAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(128, 256), torch.nn.ReLU(),
                                       torch.nn.Linear(256, 128))
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, values):
        delta = self.net(values)
        result = torch.nn.functional.normalize(values + .5 * delta, dim=1)
        result = torch.where(values.norm(dim=1, keepdim=True) > 0, result,
                             torch.zeros_like(result))
        if not self.training:
            result = torch.where((delta == 0).all(dim=1, keepdim=True), values, result)
        return result


class TextStore:
    def __init__(self, vectors, mapping):
        self.positions = json.loads(Path(mapping).read_text())
        positions = list(self.positions.values())
        if (not positions or any(type(p) is not int for p in positions)
                or sorted(positions) != list(range(len(positions)))):
            raise ValueError("Text map must map IDs bijectively to vector rows")
        if Path(vectors).stat().st_size != len(positions) * 768 * 2:
            raise ValueError("Text vectors must be fp16 [map rows,768]")
        self.values = np.memmap(vectors, np.float16, "r", shape=(len(positions), 768))

    def get_batch_vecs(self, ids):
        result = np.zeros((len(ids), 768), np.float32)
        for row, item in enumerate(ids):
            position = self.positions.get(str(item))
            if position is not None:
                result[row] = self.values[position]
        return torch.from_numpy(result)


def checked_inputs(config):
    config = Path(config).resolve()
    value = json.loads(config.read_text())
    if value.get("version") != 1 or set(value.get("inputs", {})) != INPUTS:
        raise ValueError("Expected version1 and all ten explicit input assets")
    result = {}
    for name, spec in value["inputs"].items():
        path = Path(spec["path"])
        path = (config.parent / path).resolve() if not path.is_absolute() else path.resolve()
        digest = spec["sha256"]
        if (not isinstance(digest, str) or len(digest) != 64
                or sha(path) != digest):
            raise ValueError("Input SHA256 mismatch: " + name)
        result[name] = {"path": str(path), "sha256": digest}
    cutoff = value.get("training_cutoff_exclusive")
    if type(cutoff) is not int or cutoff <= 0:
        raise ValueError("Explicit integer training cutoff required")
    return value, result


def make_training_rows(requests, labels, queries, vectors):
    """Dedup target IDs; own positives override fit-known/history negative masks."""
    positive = labels[labels.positive > 0]
    by_gid = {int(g): sorted(set(a.note_idx.astype(int))) for g, a in positive.groupby("gid")}
    by_user = {int(u): set(a.note_idx.astype(int)) for u, a in positive.groupby("user_idx")}
    zero = {int(g): a.note_idx.to_numpy(np.int64) for g, a in labels[labels.positive == 0].groupby("gid")}
    entries = []
    with threadpool_limits(2):
        for row, request in enumerate(requests.itertuples()):
            own = sorted(set(map(int, request.positives)))
            if not own or own != by_gid.get(int(request.gid), []):
                raise ValueError("Fit positive lists disagree with raw fit labels")
            blocked = (by_user[int(request.user_idx)] |
                       set(recent_history_ids(request.history, 20))) - set(own)
            negative = np.array([int(i) for i in zero.get(int(request.gid), [])
                                 if int(i) not in blocked], np.int64)
            if len(negative):
                scores = np.einsum("ij,j->i", np.array(vectors[negative]), queries[row], optimize=False)
                finite("hard negative scores", scores)
                negative = negative[np.lexsort((negative, -scores))[:4]]
            entries.append(dict(gid=int(request.gid), user_idx=int(request.user_idx),
                                positives=own, hardneg=negative.tolist(), blocked=sorted(blocked)))
    return pd.DataFrame(entries)


def batch_masks(entries, selected):
    candidates = np.unique(np.concatenate([
        np.concatenate([entries.positives.iloc[k], entries.hardneg.iloc[k]]) for k in selected
    ])).astype(np.int64)
    positive = np.zeros((len(selected), len(candidates)), bool)
    allowed = np.ones(positive.shape, bool)
    for row, k in enumerate(selected):
        own = np.asarray(entries.positives.iloc[k], np.int64)
        block = np.asarray(entries.blocked.iloc[k], np.int64)
        positive[row, np.searchsorted(candidates, own)] = True
        where = np.searchsorted(candidates, block)
        valid = where < len(candidates)
        where = where[valid]
        where = where[candidates[where] == block[valid]]
        allowed[row, where] = False
    if not positive.any(1).all() or not allowed[positive].all():
        raise ValueError("Every request must retain all its positive candidates")
    return candidates, positive, allowed


def contrastive_loss(query, target, positive, allowed):
    logits = query @ target.T / .07
    finite("logits", logits)
    positive = torch.as_tensor(positive, dtype=torch.bool, device=logits.device)
    allowed = torch.as_tensor(allowed, dtype=torch.bool, device=logits.device)
    if (positive.shape != logits.shape or allowed.shape != logits.shape
            or not positive.any(1).all() or not allowed[positive].all()):
        raise ValueError("Invalid multi-positive/false-negative mask")
    log_probability = torch.log_softmax(logits.masked_fill(~allowed, -torch.inf), dim=1)
    loss = -(log_probability.masked_fill(~positive, 0).sum(1) / positive.sum(1)).mean()
    finite("loss", loss)
    return loss


def prepare(config, output):
    output = Path(output)
    value, inputs = checked_inputs(config)
    if output.exists():
        raise FileExistsError(output)
    configure()
    paths = {k: Path(v["path"]) for k, v in inputs.items()}
    manifest = json.loads(paths["split_manifest"].read_text())
    cutoff = value["training_cutoff_exclusive"]
    if manifest.get("fit_end_exclusive") != cutoff:
        raise ValueError("Cutoff does not match the original temporal split")
    partitions = pd.read_parquet(paths["partitions"], columns=["gid", "lo", "hi", "partition"])
    if not partitions.gid.is_unique:
        raise ValueError("Duplicate split request IDs")
    eligible = partitions[partitions.partition.eq("fit")]
    if eligible.empty or not eligible.hi.lt(cutoff).all():
        raise ValueError("Invalid original fit partition")
    # Only fit-only labels/requests are accepted; no validation/test feature or feedback input.
    requests = pd.read_parquet(paths["requests"]).sort_values("gid").reset_index(drop=True)
    labels = pd.read_parquet(paths["labels"])
    expected = set(map(int, eligible.gid))
    if (not requests.gid.is_unique or set(requests.gid) != expected
            or set(labels.gid) != expected or labels[["gid", "note_idx"]].duplicated().any()
            or not labels.positive.isin([0, 1]).all() or not labels.hi.lt(cutoff).all()
            or not labels.lo.le(labels.hi).all() or labels.groupby("gid").user_idx.nunique().ne(1).any()
            or not np.array_equal(requests.user_idx, labels.groupby("gid").user_idx.first().sort_index())):
        raise ValueError("Fit roster/labels/time/user contract mismatch")
    ids = np.load(paths["item_ids"], allow_pickle=False)
    if ids.dtype != np.int64 or not np.array_equal(ids, np.arange(len(ids))) or not len(ids):
        raise ValueError("This recipe requires explicit contiguous catalog IDs, including valid item0")
    if labels.note_idx.lt(0).any() or labels.note_idx.ge(len(ids)).any():
        raise ValueError("Label item outside frozen catalog")
    if paths["pca_vectors"].stat().st_size != len(ids) * 128 * 4:
        raise ValueError("PCA vector byte size mismatch")
    vectors = np.memmap(paths["pca_vectors"], np.float32, "r", shape=(len(ids), 128))
    for start in range(0, len(ids), 65536):
        finite("PCA items", vectors[start:start+65536])
    mean = np.load(paths["pca_mean"], allow_pickle=False)
    components = np.load(paths["pca_components"], allow_pickle=False)
    if mean.dtype != np.float32 or components.dtype != np.float32:
        raise ValueError("PCA must be float32")
    finite("PCA mean", mean)
    finite("PCA components", components)
    store = TextStore(paths["text_vectors"], paths["text_map"])
    queries = np.zeros((len(requests), 128), np.float32)
    with torch.no_grad(), threadpool_limits(2):
        for row, history in enumerate(requests.history):
            raw = raw_history_text_mean(history, store)
            queries[row] = project_content(torch.from_numpy(raw[None]), torch.from_numpy(mean),
                                           torch.from_numpy(components)).numpy()[0]
    finite("fit queries", queries)
    entries = make_training_rows(requests, labels, queries, vectors)
    used = np.linalg.norm(queries, axis=1) > 0
    if not used.any():
        raise ValueError("No usable fit history queries")
    output.mkdir(parents=True)
    source = Path(__file__).resolve()
    plan = dict(version=1, training=RECIPE, variant="dual_tower", history_contract=HISTORY_ORDER,
                training_cutoff_exclusive=cutoff, source_sha256={v["path"]: v["sha256"] for v in inputs.values()},
                inputs=inputs, code_sha256=sha(source),
                entrypoint_sha256=sha(source.parents[2]/"scripts/train_rec_content_adapter.py"), official_test_read=False,
                semantics="Own exposure0 hard4; other-request positives are implicit retrieval contrasts, not CTR negatives; no target injected into history",
                precision="FP32/TF32false/deterministic; prepCPU2 single-request; training device explicitly selected",
                learned_branch="Frozen .25 learned + .75 content outside content-only training loss")
    write(output / "plan.json", plan)
    np.save(output / "fit_queries.npy", queries)
    np.save(output / "fit_used.npy", used)
    np.save(output / "fit_request_ids.npy", requests.gid.to_numpy(np.int64))
    entries.to_parquet(output / "training_requests.parquet", index=False)
    files = ["fit_queries.npy", "fit_used.npy", "fit_request_ids.npy", "training_requests.parquet"]
    write(output / "data_contract.json", dict(fit_requests=len(requests), fit_clicked_pairs=int(labels.positive.sum()),
          fit_usable_requests=int(used.sum()), fit_empty_excluded_SGD_only=int((~used).sum()),
          actual_hardnegative_pairs=int(entries.hardneg.map(len).sum()),
          training_updates_per_epoch=int(np.ceil(used.sum()/512)),
          derived_sha256={f: sha(output/f) for f in files}, official_test_read=False))
    verify(output)


def verify(output):
    output = Path(output)
    plan = json.loads((output/"plan.json").read_text())
    if (plan["training"] != RECIPE or plan["code_sha256"] != sha(Path(__file__))
            or plan["entrypoint_sha256"] != sha(Path(__file__).resolve().parents[2]/"scripts/train_rec_content_adapter.py")):
        raise ValueError("Frozen recipe or training source changed")
    for path, digest in plan["source_sha256"].items():
        if sha(path) != digest:
            raise ValueError("Frozen source changed: " + path)
    data = json.loads((output/"data_contract.json").read_text())
    for path, digest in data["derived_sha256"].items():
        if sha(output/path) != digest:
            raise ValueError("Prepared data changed: " + path)
    return plan, data


def train(output, device):
    output = Path(output)
    plan, data = verify(output)
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Explicit available cpu/cuda device required; no silent fallback")
    destination = output/"dual_tower"
    destination.mkdir(exist_ok=False)
    configure()
    queries = np.load(output/"fit_queries.npy")
    used = np.load(output/"fit_used.npy")
    entries = pd.read_parquet(output/"training_requests.parquet")
    n = len(np.load(plan["inputs"]["item_ids"]["path"]))
    items = np.memmap(plan["inputs"]["pca_vectors"]["path"], np.float32, "r", shape=(n, 128))
    x = torch.from_numpy(queries).to(device)
    query, item = ContentAdapter().to(device), ContentAdapter().to(device)
    optimizer = torch.optim.AdamW(list(query.parameters())+list(item.parameters()), lr=.001, weight_decay=.0001)
    rng = np.random.default_rng(42)
    records, schedule = [], []
    for epoch in range(1, 4):
        start = time.perf_counter()
        order = rng.permutation(np.flatnonzero(used))
        schedule.append(order)
        batches, weighted, anchors = [], 0., 0
        for step, start_row in enumerate(range(0, len(order), 512)):
            selected = order[start_row:start_row+512]
            candidates, positive, allowed = batch_masks(entries, selected)
            targets = torch.from_numpy(np.array(items[candidates], copy=True)).to(device)
            loss = contrastive_loss(query(x[selected]), item(targets), positive, allowed)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for name, parameter in list(query.named_parameters())+list(item.named_parameters()):
                finite(name+" gradient", parameter.grad)
            optimizer.step()
            for name, parameter in list(query.named_parameters())+list(item.named_parameters()):
                finite(name+" parameter", parameter)
            weighted += float(loss.detach().cpu()) * len(selected)
            anchors += len(selected)
            batches.append(dict(batch=step, anchors=len(selected), candidate_items=len(candidates),
                                positive_pairs=int(positive.sum()), blocked_comparisons=int((~allowed).sum()),
                                loss=float(loss.detach().cpu())))
        target = destination/f"epoch_{epoch}"
        target.mkdir()
        checkpoint = dict(query=query.cpu().state_dict(), item=item.cpu().state_dict(),
                          variant="dual_tower", epoch=epoch, architecture="128-256-ReLU-128, beta.5",
                          training_plan_sha256=sha(output/"plan.json"), data_contract_sha256=sha(output/"data_contract.json"),
                          official_test_read=False)
        torch.save(checkpoint, target/"towers.pt")
        query.to(device)
        item.to(device)
        record = dict(variant="dual_tower", epoch=epoch, batches=len(batches), updates=len(batches),
                      training_anchors=anchors, request_weighted_loss=weighted/anchors,
                      train_seconds=time.perf_counter()-start, weights_sha256=sha(target/"towers.pt"))
        records.append(record)
        write(target/"training.json", record)
        pd.DataFrame(batches).to_parquet(target/"training_batches.parquet", index=False)
        print(json.dumps(record), flush=True)
    np.save(destination/"request_schedule.npy", np.asarray(schedule))
    write(output/"training_summary.json", dict(trials=records, official_test_read=False, device=device,
          training_plan_sha256=sha(output/"plan.json"), data_contract_sha256=sha(output/"data_contract.json"),
          optimizer_updates=sum(r["updates"] for r in records)))
    verify(output)


def export(output, epoch, destination):
    """Copy a trained native head plus exact lineage; no model/index promotion."""
    import shutil
    output, destination = Path(output), Path(destination)
    verify(output)
    if epoch not in (1, 2, 3):
        raise ValueError("Epoch must be an already trained predeclared checkpoint")
    heads = output/"dual_tower"/f"epoch_{epoch}"/"towers.pt"
    payload = torch.load(heads, map_location="cpu", weights_only=False)
    summary = json.loads((output/"training_summary.json").read_text())
    records = [r for r in summary["trials"] if r["epoch"] == epoch]
    if len(records) != 1 or sha(heads) != records[0]["weights_sha256"]:
        raise ValueError("Selected heads do not match actual training summary")
    for role in ("query", "item"):
        adapter = ContentAdapter()
        adapter.load_state_dict(payload[role], strict=True)
        for parameter in adapter.parameters():
            finite("export "+role, parameter)
    destination.mkdir(parents=True, exist_ok=False)
    for source, name in [(heads, "towers.pt"), (output/"plan.json", "plan.json"),
                         (output/"data_contract.json", "data_contract.json"),
                         (output/"training_summary.json", "training_summary.json")]:
        shutil.copyfile(source, destination/name)
    write(destination/"export_manifest.json", dict(version=1, epoch=epoch,
          files={p.name: sha(p) for p in destination.iterdir()}, official_test_read=False,
          scope="Native heads/lineage compatible with training.content_adapter_release; no full model/index export"))
