from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb


class GBDTPredictor:
    def __init__(self, scene: str, tag: str, deploy_tag_dir: Path, group_key: str, feat_columns: list[str]):
        self.scene = scene
        self.tag = tag
        self.deploy_tag_dir = deploy_tag_dir
        self.group_key = group_key
        self.lgb_model = None
        self.xgb_model = None
        self.feature_cols: list[str] = []
        self._load_models(feat_columns)

    def _artifact_path(self, rel_path: str) -> Path:
        return self.deploy_tag_dir / rel_path

    def _load_models(self, feat_columns: list[str]) -> None:
        lgb_path = self._artifact_path(f"models/lgb_{self.scene}_{self.tag}.pkl")
        xgb_path = self._artifact_path(f"models/xgb_{self.scene}_{self.tag}.pkl")
        if lgb_path.exists():
            self.lgb_model = joblib.load(lgb_path)
        if xgb_path.exists():
            self.xgb_model = joblib.load(xgb_path)

        if self.lgb_model is not None:
            self.feature_cols = list(self.lgb_model.feature_name())
        else:
            drop_cols = ["click", "y_multi", "note_idx", self.group_key, "recent_clicked_note_idxs", "first_route"]
            self.feature_cols = [c for c in feat_columns if c not in drop_cols]

    def _prepare_scoring_frame(self, cand: pd.DataFrame) -> pd.DataFrame:
        df = cand.copy()
        if "first_route" in df.columns and "first_route_id" not in df.columns:
            route_map = {"ann": 1, "swing": 2, "usercf": 3, "hot": 4}
            df["first_route_id"] = (
                df["first_route"].astype(str).str.lower().map(route_map).fillna(0).astype(np.int16)
            )

        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        X = df[self.feature_cols].copy()
        for c in X.columns:
            if not pd.api.types.is_numeric_dtype(X[c]):
                X[c] = pd.to_numeric(X[c], errors="coerce")
        return X.fillna(0.0)

    def predict(self, cand: pd.DataFrame) -> np.ndarray:
        if cand.empty:
            return np.zeros(0, dtype=np.float32)
        X = self._prepare_scoring_frame(cand)

        lgb_pred = np.zeros(len(X), dtype=np.float32)
        if self.lgb_model is not None:
            best_iter = getattr(self.lgb_model, "best_iteration", None)
            if best_iter is None or best_iter <= 0:
                lgb_pred = np.asarray(self.lgb_model.predict(X), dtype=np.float32)
            else:
                lgb_pred = np.asarray(self.lgb_model.predict(X, num_iteration=best_iter), dtype=np.float32)

        xgb_pred = np.zeros(len(X), dtype=np.float32)
        if self.xgb_model is not None:
            dtest = xgb.DMatrix(X.to_numpy())
            best_iter = getattr(self.xgb_model, "best_iteration", None)
            if best_iter is None or best_iter <= 0:
                xgb_pred = np.asarray(self.xgb_model.predict(dtest), dtype=np.float32)
            else:
                xgb_pred = np.asarray(self.xgb_model.predict(dtest, iteration_range=(0, int(best_iter) + 1)), dtype=np.float32)

        if self.lgb_model is not None and self.xgb_model is not None:
            return (0.5 * lgb_pred + 0.5 * xgb_pred).astype(np.float32)
        if self.lgb_model is not None:
            return lgb_pred
        if self.xgb_model is not None:
            return xgb_pred
        return np.zeros(len(X), dtype=np.float32)


def run_preranking_gbdt(cand: pd.DataFrame, gbdt_topn: int, predict_gbdt) -> pd.DataFrame:
    out = cand.copy()
    out["gbdt_score"] = predict_gbdt(out)
    out["preranking_score"] = (0.8 * out["gbdt_score"] + 0.2 * out["dssm_score"]).astype(np.float32)
    out = out.sort_values(["preranking_score", "rank"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    if int(gbdt_topn) > 0:
        out = out.head(int(gbdt_topn)).reset_index(drop=True)
    return out
