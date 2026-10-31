"""Request-local numerical execution shared by serving and frozen replay."""
from __future__ import annotations

import numpy as np
import torch

REQUEST_FP32 = "request_fp32_v1"
CPU_THREADS = 4
FINE_BATCH_SIZE = 1024


def validate_inference_contract(contract):
    if contract is not None and (not isinstance(contract, str) or contract != REQUEST_FP32):
        raise ValueError("Unsupported request inference contract")


def configure_request_inference(contract):
    """Configure once at model startup; forwards reject subsequent global drift."""
    validate_inference_contract(contract)
    if contract == REQUEST_FP32:
        torch.set_num_threads(CPU_THREADS)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False


def check_request_inference(contract):
    validate_inference_contract(contract)
    if contract == REQUEST_FP32 and (
        torch.get_num_threads() != CPU_THREADS
        or torch.backends.cuda.matmul.allow_tf32
        or torch.backends.cudnn.allow_tf32
        or torch.backends.cudnn.benchmark
    ):
        raise RuntimeError("Request FP32 contract requires CPU threads4, TF32 disabled and cuDNN benchmark disabled")


def canonical_request_logits(model, context, frame, numeric, device, group_key):
    """Score each request in recall order with a fixed1024 tensor batch.

    Padding repeats the final candidate. Evaluation logits are row independent;
    padding creates no candidates, labels or scores in the returned result.
    """
    check_request_inference(REQUEST_FP32)
    if model.training:
        raise ValueError("Canonical request inference requires an evaluation model")
    result = np.empty(len(frame), dtype=np.float32)
    with torch.inference_mode(), torch.autocast(device_type=torch.device(device).type, enabled=False):
        for positions in frame.groupby(group_key, sort=False).indices.values():
            request = frame.iloc[positions]
            ranks = request["rank"].to_numpy(dtype=np.float64)
            if (not np.isfinite(ranks).all() or (ranks <= 0).any()
                    or (ranks != np.floor(ranks)).any() or len(np.unique(ranks)) != len(ranks)):
                raise ValueError("Canonical inference needs unique positive original recall ranks")
            if len(request) > FINE_BATCH_SIZE:
                raise ValueError("Request exceeds the canonical fine batch size")
            order = np.argsort(ranks, kind="stable")
            features = context.batch(request.iloc[order], numeric, device)
            count = len(request)
            if count < FINE_BATCH_SIZE:
                features = {name: torch.cat((value, value[-1:].expand(FINE_BATCH_SIZE-count, *value.shape[1:])), dim=0)
                            for name, value in features.items()}
            raw, _ = model(features)
            values = raw.detach().float().cpu().numpy().reshape(-1)[:count]
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite canonical fine prediction")
            result[np.asarray(positions)[order]] = values
    return result
