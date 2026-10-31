"""Live request features from original training history and explicit current inputs.

Official test exposures/labels are deliberately not accepted by this module.
"""
from __future__ import annotations

import pickle
import math
import uuid
from pathlib import Path

import duckdb
import pandas as pd


class LiveContextBuilder:
    def __init__(self, root: Path, scene: str, profiles: dict[int, dict], *, cutoff_exclusive: int | None = None):
        self.scene = scene
        if cutoff_exclusive is not None and (not math.isfinite(float(cutoff_exclusive)) or int(cutoff_exclusive) <= 0):
            raise ValueError("Training cutoff must be a positive finite timestamp")
        self.cutoff_exclusive = int(cutoff_exclusive) if cutoff_exclusive is not None else None
        self.group_key = "search_idx" if scene == "search" else "request_idx"
        self.profiles = profiles
        self.vocabularies = {}
        for name in ("gender", "platform", "age", "location"):
            with (root / "features" / "vocab_dict" / f"{name}.pkl").open("rb") as stream:
                self.vocabularies[name] = pickle.load(stream)
        folder = "search_train" if scene == "search" else "recommendation_train"
        source = str(root / "datasets" / folder / "*.parquet")
        nested = "search_result_details_with_idx" if scene == "search" else "rec_result_details_with_idx"
        timestamp = "search_timestamp" if scene == "search" else "request_timestamp"
        with duckdb.connect(config={"threads": "2", "memory_limit": "1GB"}) as connection:
            relation = connection.read_parquet(source)
            if self.cutoff_exclusive is not None:
                # Keep complete requests only; never retain earlier rows of a spanning request.
                relation = relation.filter(f"list_max(list_transform({nested}, x -> x.{timestamp})) < {self.cutoff_exclusive}")
            relation.create_view("past_train")
            self.training_request_count = int(connection.execute("SELECT count(*) FROM past_train").fetchone()[0])
            latest = connection.execute(f"""
                SELECT user_idx, query, recent_clicked_note_idxs
                FROM past_train
                QUALIFY row_number() OVER (
                    PARTITION BY user_idx
                    ORDER BY list_max(list_transform({nested}, x -> x.{timestamp})) DESC,
                             {self.group_key} DESC
                ) = 1
            """).fetchdf()
            self.hot_note_ids = [int(row[0]) for row in connection.execute(f"""
                SELECT detail.note_idx FROM past_train request,
                    unnest(request.{nested}) details(detail)
                WHERE detail.click > 0
                GROUP BY detail.note_idx ORDER BY count(*) DESC, detail.note_idx ASC LIMIT 20000
            """).fetchall()]
            self.query_catalog = [(str(query), int(count)) for query, count in connection.execute(
                "SELECT query,count(*) FROM past_train WHERE query IS NOT NULL AND trim(query)<>'' "
                "GROUP BY query ORDER BY count(*) DESC,query").fetchall()] if scene == "search" else []
        self.training_contexts = {int(row["user_idx"]): row for row in latest.to_dict("records")}

    def seed_history(self, user_idx: int) -> list[int]:
        row = self.training_contexts.get(int(user_idx), {})
        return [int(note) for note in row.get("recent_clicked_note_idxs", []) if int(note) >= 0][:20]

    def build(self, user_idx: int, query: str, history: list[int]) -> pd.DataFrame:
        uid = int(user_idx)
        if uid < 0:
            raise ValueError("user_idx must be nonnegative")
        profile = self.profiles.get(uid, {})
        # JSON numeric identifiers remain exact in the existing JavaScript frontend.
        # The separate 128-bit trace ID supplies global execution identity.
        row = {self.group_key: uuid.uuid4().int & ((1 << 53) - 1), "user_idx": uid,
               "note_idx": -1, "query": str(query or ""),
               "recent_clicked_note_idxs": [int(note) for note in history if int(note) >= 0][:20],
               "fans_num": 0.0, "follows_num": 0.0}
        for name, vocabulary in self.vocabularies.items():
            raw = profile.get(name)
            row[name + "_enc"] = int(vocabulary.get(str(raw) if pd.notna(raw) else "UNK", 0))
        return pd.DataFrame([row])