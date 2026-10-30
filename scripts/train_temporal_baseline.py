import argparse, hashlib, json, time, importlib.util, importlib.metadata, platform
import duckdb
from pathlib import Path
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import roc_auc_score
from training.utils import RANKING_STATIC_FEATURES, split_groups_temporally
ROOT = Path(__file__).resolve().parents[1]


def temporal_training_indices(train, test, key, valid_ratio=.2):
    for frame in (train, test):
        if "event_timestamp" not in frame or key not in frame or frame[key].isna().any():
            raise ValueError("Missing request IDs or event timestamps")
        values = pd.to_numeric(frame.event_timestamp, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite event timestamps")
    test_start = float(test.event_timestamp.min())
    if not np.isfinite(test_start):
        raise ValueError("Invalid official test timestamp")
    intervals = train.groupby(key).event_timestamp.agg(["min", "max"])
    eligible = intervals.index[intervals["max"] < test_start]
    historical = train[train[key].isin(eligible)].reset_index(drop=True)
    fit, val = split_groups_temporally(historical, key, valid_ratio)
    return historical, fit, val, test_start


def main():
    parser = argparse.ArgumentParser(description="Content-only temporal classification baseline; never deploys models.")
    parser.add_argument("--scene", choices=["search", "rec"], default="rec")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = ROOT
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    key = "request_idx" if args.scene == "rec" else "search_idx"
    folder = "recommendation" if args.scene == "rec" else "search"
    nested = "rec_result_details_with_idx" if args.scene == "rec" else "search_result_details_with_idx"
    source = root / f"features/{args.scene}_train_features.parquet"
    target = root / f"features/{args.scene}_test_features.parquet"
    cols = [key, "user_idx", "note_idx", "event_timestamp", "click", *RANKING_STATIC_FEATURES]
    train = pd.read_parquet(source, columns=cols)
    test = pd.read_parquet(target, columns=cols)
    assert set(train[key]).isdisjoint(test[key])
    raw_path = str(root / f"datasets/{folder}_test/*.parquet")
    with duckdb.connect() as con:
        con.register("features", test)
        mismatch = con.execute(f"""
            WITH raw AS (
                SELECT r.{key} gid,d.note_idx,d.click,count(*) n
                FROM read_parquet(?) r,unnest(r.{nested}) u(d)
                GROUP BY r.{key},d.note_idx,d.click
            ), feat AS (
                SELECT {key} gid,note_idx,click,count(*) n FROM features
                GROUP BY {key},note_idx,click
            )
            SELECT count(*) FROM raw FULL JOIN feat USING(gid,note_idx,click)
            WHERE raw.n IS DISTINCT FROM feat.n
        """, [raw_path]).fetchone()[0]
        if mismatch:
            raise ValueError("Features do not cover every original test exposure/label")
    historical, fit, val, test_start = temporal_training_indices(train, test, key)
    features=list(RANKING_STATIC_FEATURES)
    X=historical[features].replace([np.inf,-np.inf],np.nan).fillna(0)
    Xt=test[features].replace([np.inf,-np.inf],np.nan).fillna(0)
    y=historical.click.to_numpy(np.int8)
    model=LGBMClassifier(n_estimators=500,learning_rate=.05,num_leaves=15,min_child_samples=100,
                        reg_lambda=1.,random_state=42,n_jobs=4,verbosity=-1)
    start=time.monotonic()
    model.fit(X.iloc[fit],y[fit],eval_set=[(X.iloc[val],y[val])],eval_metric="auc",callbacks=[early_stopping(30,verbose=False)])
    scores=model.predict_proba(Xt)[:,1]
    module_path=root/"src/backend/offline/evaluate_full_test.py"
    spec=importlib.util.spec_from_file_location("local_eval",module_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    frame=test.rename(columns={key:"gid","click":"positive"})
    metrics,requests,predictions=module.ranking_metrics(frame,scores)
    model.booster_.save_model(str(out/"model.txt"))
    predictions.to_parquet(out/"test_predictions.parquet",index=False)
    requests.to_parquet(out/"test_requests.parquet",index=False)
    def digest(path):
     h=hashlib.sha256()
     with path.open("rb") as f:
      for chunk in iter(lambda:f.read(16*1024*1024),b""):h.update(chunk)
     return h.hexdigest()
    manifest={"model":"content-only LGBMClassifier; NOT deployed",
              "scene":args.scene,"features":features,"label":"raw binary click","seed":42,"best_iteration":model.best_iteration_,
              "train_input_rows":len(train),"historical_rows":len(historical),"fit_rows":len(fit),"validation_rows":len(val),
              "test_rows":len(test),"test_requests":int(test[key].nunique()),
              "fit_last_event":float(historical.iloc[fit].event_timestamp.max()),
              "validation_first_event":float(historical.iloc[val].event_timestamp.min()),
              "validation_last_event":float(historical.iloc[val].event_timestamp.max()),"test_first_event":test_start,
              "validation_auc":float(roc_auc_score(y[val],model.predict_proba(X.iloc[val])[:,1])),
              "test":metrics,"seconds":time.monotonic()-start,
              "environment":{"python":platform.python_version(), **{name:importlib.metadata.version(name) for name in ("lightgbm","numpy","pandas","scikit-learn")}},
              "missing_numeric_policy":"nonfinite content features filled with zero",
              "inputs":{str(p.relative_to(root)):digest(p) for p in [source,target,Path(__file__).resolve(),root/"src/training/utils.py",module_path,*sorted((root/f"datasets/{folder}_test").glob("*.parquet"))]},
              "outputs":{"test_predictions_sha256":digest(out/"test_predictions.parquet"),"test_requests_sha256":digest(out/"test_requests.parquet")},
              "model_sha256":digest(out/"model.txt"),
              "limitations":["Only labels split in time; content snapshot and category vocabulary point-in-time provenance remain unverified",
                             "Conditional logged-exposure classifier baseline; no claim about online pipeline uplift",
                             "Test used once after validation-only early stopping"]}
    assert manifest["fit_last_event"]<manifest["validation_first_event"]<=manifest["validation_last_event"]<test_start
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)

if __name__ == '__main__':
    main()
