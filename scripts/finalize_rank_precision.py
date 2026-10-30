"""Finalize frozen Search/Recommendation ranking checkpoints under mandatory FP32 inference.

Only the five original validation blend weights are reselected. No gradients,
preprocessing fitting, teacher changes, or test-driven precision selection.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
import platform
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
BLEND_GRID=(0.,.1,.25,.5,1.)
CANONICAL_BATCH=1024
NATIVE_REQUESTS=64
SEED=42


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(16*1024*1024),b""):h.update(block)
    return h.hexdigest()


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")


def make_fp32_manifest(parent,scene):
    if parent["scene"]!=scene:raise ValueError("Parent scene mismatch")
    child=deepcopy(parent)
    child.pop("final_test",None)
    child["dien"]["architecture"]["inference_precision"]="fp32"
    # These flags identify actual frozen training, rather than silently relying on new defaults.
    contract=parent["dien"].get("auxiliary_contract")
    if contract is None:
        contract="v1: original same-request unclicked auxiliary sampling; no-valid-negative fallback used batch column0"
    is_v2=contract.startswith("v2")
    child["dien"]["architecture"]["mask_missing_aux_negatives"]=is_v2
    child["dien"]["auxiliary_contract"]=contract
    child["dien"]["objective"]=parent["dien"].get("objective",
        "original exposure BCE + .25 batch-mean intra-request pair softplus + .01 auxiliary loss")
    child["precision_finalization"]={"precision":"fp32","precision_selection":"mandatory accuracy protocol, not selected on test",
        "canonical_batch_size":CANONICAL_BATCH,"gradient_updates":0,"normalization_refitted":False,
        "teacher_changed":False,"model_weights_changed":False,"tf32_matmul":False,"tf32_cudnn":False,
        "validation_blend_grid":list(BLEND_GRID),"parent_training_contract":contract,
        "score_interpretation":"Raw neural ranking logits; per-request minmax blend global AUC is not calibrated pCTR",
        "embedding_storage":"Existing frozen embedding snapshots retained; FP32 describes model arithmetic"}
    return child


def top_sets(score,groups,notes,k=10):
    score=np.asarray(score);groups=np.asarray(groups);notes=np.asarray(notes)
    result={}
    for gid in np.unique(groups):
        rows=np.flatnonzero(groups==gid)
        order=np.lexsort((rows,notes[rows],-score[rows]))[:k]
        result[int(gid)]=set(map(int,rows[order]))
    return result


def compare_profiles(reference,actual,groups,notes):
    reference=np.asarray(reference,dtype=float);actual=np.asarray(actual,dtype=float)
    if reference.shape!=actual.shape or not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError("Profile score arrays must be aligned and finite")
    delta=np.abs(reference-actual)
    a=top_sets(reference,groups,notes);b=top_sets(actual,groups,notes)
    overlaps=[len(a[gid]&b[gid])/max(len(a[gid]|b[gid]),1) for gid in a]
    old=pd.Series(reference).groupby(np.asarray(groups),sort=False).rank(method="min",ascending=False).to_numpy()
    new=pd.Series(actual).groupby(np.asarray(groups),sort=False).rank(method="min",ascending=False).to_numpy()
    return {"rows":len(reference),"requests":len(a),"max_absolute_score_difference":float(delta.max(initial=0.)),
        "mean_absolute_score_difference":float(delta.mean()) if len(delta) else 0.,
        "changed_rank_rows":int((old!=new).sum()),"observed_equal_scores":bool(np.array_equal(reference,actual)),
        "top10_exact_set_requests":int(sum(a[gid]==b[gid] for gid in a)),
        "top10_mean_set_jaccard":float(np.mean(overlaps)) if overlaps else None,
        "top10_min_set_jaccard":float(np.min(overlaps)) if overlaps else None,
        "top10_set_identity":"Original exposure row; score ties resolved by note_idx then source row",
        "claim_scope":"Observed profile comparison; no guarantee of bitwise equality across batch shapes"}


def choose_blend(choices):
    if tuple(row["weight"] for row in choices)!=BLEND_GRID:
        raise ValueError("Only the predefined five validation weights are allowed")
    return max(choices,key=lambda row:(row["validation"]["NDCG@10"],-row["weight"]))


def load_model(manifest,path,device):
    import torch
    from training.dien_ranker import DIENRanker
    model=DIENRanker(manifest["scene"],manifest["dien"]["numeric_features"],
        **manifest["dien"]["architecture"]).to(device)
    model.load_state_dict(torch.load(path,map_location=device,weights_only=True))
    model.eval()
    if getattr(model,"inference_precision",None)!="fp32":
        raise ValueError("Root FP32 inference interface is required")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise ValueError("FP32 inference must disable both TF32 paths")
    return model


def main():
    import torch
    import lightgbm as lgb
    from train_rank_stack import contexts,score_dien,metrics,group_minmax,tree_predict,evaluate
    from backend.online.item_features import get_item_feature_store
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene",choices=["search","rec"],required=True)
    parser.add_argument("--parent-run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--train-query-cache",type=Path)
    parser.add_argument("--test-query-cache",type=Path)
    args=parser.parse_args();started=time.monotonic()
    parent=args.parent_run.resolve();out=args.output.resolve()
    if out.exists():raise FileExistsError("Use a new precision-finalization output directory")
    out.mkdir(parents=True)
    torch.set_num_threads(4);torch.manual_seed(SEED);np.random.seed(SEED)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type=="cuda" and torch.cuda.mem_get_info()[0]<3*1024**3:
        raise RuntimeError("Insufficient free GPU memory for FP32 inference")
    parent_manifest=json.loads((parent/"manifest.json").read_text())
    child=make_fp32_manifest(parent_manifest,args.scene);key=child["key"]
    teacher=ROOT/f"outputs/evaluation/optimization_20261031/{args.scene}_rank"
    validation=teacher/"validation.parquet"
    if digest(parent/"gbdt.txt")!=digest(teacher/"gbdt.txt"):
        raise ValueError("Frozen parent teacher differs from the source validation teacher")
    parent_hashes={name:digest(parent/name) for name in ("manifest.json","gbdt.txt","dien.pt")}
    for name in ("gbdt.txt","dien.pt"):shutil.copyfile(parent/name,out/name)
    for path in parent.glob("*.txt"):
        if path.name!="gbdt.txt":shutil.copyfile(path,out/path.name)
    child["precision_finalization"].update({"created_utc":datetime.now(timezone.utc).isoformat(),
        "parent_run":str(parent),"parent_sha256":parent_hashes,
        "validation_source":str(validation),"validation_source_sha256":digest(validation),
        "teacher_sha256":digest(teacher/"gbdt.txt"),"source_script_sha256":digest(Path(__file__)),
        "limitations":["Official test previously diagnosed; regression benchmark, not a new untouched holdout",
            "Provider history/catalog/vector snapshots lack event-time provenance",
            "Original checkpoint training objective retained; FP32 finalization does not retrain or repair historical auxiliary targets"]})
    save(out/"protocol.json",child["precision_finalization"])
    save(out/"parent_manifest.json",parent_manifest)
    save(out/"manifest.json",child)
    val=pd.read_parquet(validation)
    context=contexts(args.scene,"train",args.train_query_cache)
    coarse=lgb.Booster(model_file=str(out/"gbdt.txt"))
    parity=float(np.max(np.abs(tree_predict(coarse,val,child["features"])-val.coarse_score.to_numpy())))
    if parity>1e-7:raise ValueError("Validation coarse_score is not the unchanged frozen teacher output")
    model=load_model(child,out/"dien.pt",device)
    raw=score_dien(model,val,context,child["dien"]["numeric_features"],device)
    val["dien_raw_score"]=raw;val.to_parquet(out/"validation.parquet",index=False)
    groups=val[key].to_numpy();base_norm=group_minmax(val.coarse_score.to_numpy(),groups)
    fine_norm=group_minmax(raw,groups);choices=[]
    for weight in BLEND_GRID:
        result,_,_=metrics(val,(1-weight)*base_norm+weight*fine_norm,key)
        choices.append({"weight":weight,"validation":result})
    chosen=choose_blend(choices)
    child["dien"]["blend_candidates"]=choices;child["dien"]["chosen"]=chosen
    child["precision_finalization"]["validation_teacher_max_abs_error"]=parity
    child["precision_finalization"]["validation_fp32_scores_sha256"]=digest(out/"validation.parquet")
    child["dien"]["model_sha256"]=digest(out/"dien.pt")
    selection={"precision":"fp32","precision_mandatory_before_test":True,"blend_grid":list(BLEND_GRID),
        "selected":chosen,"parent_sha256":parent_hashes,"protocol_sha256":digest(out/"protocol.json"),
        "validation_scores_sha256":digest(out/"validation.parquet"),"frozen_before_test":True}
    save(out/"selection.json",selection)
    save(out/"manifest.json",child)
    print(args.scene,"FP32 validation selection",json.dumps(chosen),flush=True)
    del model,context,val
    if device.type=="cuda":torch.cuda.empty_cache()
    # Root evaluator preserves the original complete test exposure/label multiset.
    evaluate(SimpleNamespace(output=out,scene=args.scene,test_query_cache=args.test_query_cache))
    child=json.loads((out/"manifest.json").read_text())
    frame=pd.read_parquet(ROOT/f"features/{args.scene}_test_features.parquet",
        columns=[key,"user_idx","note_idx","event_timestamp","click"])
    context=contexts(args.scene,"test",args.test_query_cache)
    frame=context.feature_frame(frame,get_item_feature_store(ROOT))
    base=tree_predict(coarse,frame,child["features"]);frame["coarse_score"]=base
    model=load_model(child,out/"dien.pt",device)
    raw_again=score_dien(model,frame,context,child["dien"]["numeric_features"],device)
    weight=chosen["weight"];groups=frame[key].to_numpy();notes=frame.note_idx.to_numpy()
    combined=(1-weight)*group_minmax(base,groups)+weight*group_minmax(raw_again,groups)
    canonical={}
    for stage,values in (("gbdt",base),("dien_raw",raw_again),("selected",combined)):
        saved=pd.read_parquet(out/f"test_{stage}_predictions.parquet")
        if not np.array_equal(saved.gid.to_numpy(),groups) or not np.array_equal(saved.note_idx.to_numpy(),notes):
            raise ValueError("Canonical full-test exposure order changed")
        comparison=compare_profiles(saved.score.to_numpy(),values,groups,notes)
        canonical[stage]=comparison
        if comparison["max_absolute_score_difference"]>1e-6 or comparison["changed_rank_rows"]:
            raise ValueError(f"Canonical1024 repeat failed for {stage}: {comparison}")
    rng=np.random.default_rng(SEED)
    request_ids=np.sort(rng.choice(np.unique(groups),min(NATIVE_REQUESTS,len(np.unique(groups))),replace=False))
    sample=np.flatnonzero(np.isin(groups,request_ids));native=np.empty(len(sample),dtype=np.float32)
    # Reorder by source row after computing each request with its native number of exposures.
    position={int(row):i for i,row in enumerate(sample)}
    native_shapes=[]
    for gid in request_ids:
        rows=np.flatnonzero(groups==gid);part=frame.iloc[rows]
        values=score_dien(model,part,context,child["dien"]["numeric_features"],device)
        native[[position[int(row)] for row in rows]]=values
        native_shapes.append(len(rows))
    native_raw=compare_profiles(raw_again[sample],native,groups[sample],notes[sample])
    native_blend=(1-weight)*group_minmax(base[sample],groups[sample])+weight*group_minmax(native,groups[sample])
    native_selected=compare_profiles(combined[sample],native_blend,groups[sample],notes[sample])
    profile={"scene":args.scene,"precision":"fp32","canonical_batch_size":CANONICAL_BATCH,
        "environment":{"python":platform.python_version(),"torch":torch.__version__,
            "cuda_runtime":torch.version.cuda,"cudnn":torch.backends.cudnn.version(),
            "device":str(device),"device_name":torch.cuda.get_device_name(device) if device.type=="cuda" else "cpu",
            "torch_num_threads":torch.get_num_threads()},
        "tf32_matmul":bool(torch.backends.cuda.matmul.allow_tf32),"tf32_cudnn":bool(torch.backends.cudnn.allow_tf32),
        "canonical_full_test_repeat":canonical,"native_request_comparison":{"request_ids":request_ids.tolist(),
        "seed":SEED,"native_request_row_counts":native_shapes,"raw_dien":native_raw,"selected":native_selected},
        "source_validation_teacher_max_abs_error":parity,"model_sha256":{name:digest(out/name) for name in ("gbdt.txt","dien.pt")},
        "weights_identical_to_parent":all(digest(out/name)==parent_hashes[name] for name in ("gbdt.txt","dien.pt")),
        "gradient_updates":0}
    save(out/"precision_reproduction.json",profile)
    # Historical BF16 is retained. Compare saved scores; do not rescore/select precision on test.
    history={}
    for stage in ("gbdt","dien_raw","selected"):
        path=parent/f"test_{stage}_predictions.parquet"
        if path.exists():
            old=pd.read_parquet(path);new=pd.read_parquet(out/f"test_{stage}_predictions.parquet")
            if not np.array_equal(old.gid.to_numpy(),new.gid.to_numpy()) or not np.array_equal(old.note_idx.to_numpy(),new.note_idx.to_numpy()):
                raise ValueError("Parent BF16 prediction exposure alignment changed")
            history[stage]=compare_profiles(old.score.to_numpy(),new.score.to_numpy(),groups,notes)
    parent_metrics=json.loads((parent/"test_metrics.json").read_text())
    new_metrics=json.loads((out/"test_metrics.json").read_text())
    save(out/"precision_ablation.json",{"scope":"Historical BF16 parent versus mandatory FP32 protocol; no test selection",
        "same_checkpoint_weights":profile["weights_identical_to_parent"],"parent_blend_weight":parent_manifest["dien"]["chosen"]["weight"],
        "fp32_validation_selected_weight":weight,"note":"Selected-score changes can include validation blend reselection; raw DIEN comparison isolates inference arithmetic",
        "parent_test_metrics":parent_metrics,"fp32_test_metrics":new_metrics,"saved_score_comparison":history})
    child["precision_finalization"].update({"canonical_repeat_pass":True,"native_request_count":len(request_ids),
        "profile_sha256":digest(out/"precision_reproduction.json"),"actual_gradient_updates":0,
        "total_seconds":time.monotonic()-started})
    save(out/"manifest.json",child)
    save(out/"source_manifest.json",{"script_sha256":digest(Path(__file__)),"parent_sha256":parent_hashes,
        "root_score_helper_sha256":digest(ROOT/"scripts/train_rank_stack.py"),
        "model_source_sha256":digest(ROOT/"src/training/dien_ranker.py"),
        "selection_sha256":digest(out/"selection.json"),"precision_profile_sha256":digest(out/"precision_reproduction.json")})
    print(args.scene,"FP32 finalized",json.dumps({"selected":new_metrics["selected"],
        "canonical_raw":canonical["dien_raw"],"native_raw":native_raw,"seconds":time.monotonic()-started}),flush=True)


if __name__=="__main__":main()
