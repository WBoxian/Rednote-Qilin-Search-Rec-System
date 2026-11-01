"""Regression tests at the raw-log -> staged sample -> feature and embedding seams."""
import importlib
import json
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from preprocess.build_samples import build_samples
from preprocess.build_features import build_features
from training.history import recent_history_ids, history_ids_oldest_first
from preprocess.release import partition_requests, fit_vocabularies
from training.dien_ranker import EmbeddingStore, EMB_DIM, SEQ_LEN


class FeaturePreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / "datasets"
        self.samples = self.root / "samples"
        self.features = self.root / "features"
        self.vocab = self.root / "vocab"
        self.notes = pd.DataFrame({
            "note_idx": [0, 1], "note_title": ["first", "second"], "note_content": ["body", "body"],
            "note_type": [1., 2.], "taxonomy1_id": ["a", "b"], "taxonomy2_id": ["a1", "b1"], "taxonomy3_id": ["a2", "b2"],
            "video_duration": [0., 12.], "video_height": [0., 1080.], "video_width": [0., 1920.],
            "image_num": [3., 1.], "content_length": [4., 4.], "commercial_flag": [0., 1.],
            "imp_num": [100., 50.], "click_num": [20., 5.],
        })
        self._save(self.notes, "notes")
        user = {"user_idx": [7], "gender": ["m"], "platform": ["ios"], "age": ["25"], "location": ["city"], "fans_num": [1.], "follows_num": [2.]}
        user.update({f"dense_feat{i}": [float(i)] for i in range(1, 41)})
        self._save(pd.DataFrame(user), "user_feat")
        for scene, folder, key, detail, timestamp in [
            ("search", "search_train", "search_idx", "search_result_details_with_idx", "search_timestamp"),
            ("rec", "recommendation_train", "request_idx", "rec_result_details_with_idx", "request_timestamp"),
        ]:
            def item(note, click, page):
                return {"note_idx": note, "click": click, "like": 0, "collect": 0, "comment": 0, "share": 0,
                        "position": note + 1, "page_time": page, timestamp: 100}
            rows = [{"user_idx": 7, "session_idx": 1, "recent_clicked_note_idxs": [0, 1], "query": "query", "query_from_type": 1,
                     key: 100, detail: [item(0, 1, np.nan), item(1, 0, -1.)]},
                    {"user_idx": 7, "session_idx": 2, "recent_clicked_note_idxs": [], "query": "query", "query_from_type": 1,
                     key: 101, detail: [item(1, 1, 2.)]}]
            self._save(pd.DataFrame(rows), folder)
            build_samples(scene, "train", dataset_dir=self.dataset, output_dir=self.samples)

    def _save(self, frame, folder):
        path = self.dataset / folder / "train-00000-of-00001.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)

    def _features(self, scene="search"):
        return build_features(scene, "train", sample_dir=self.samples, output_dir=self.features, vocab_dir=self.vocab)

    def test_note_embedding_text_handles_parquet_nulls_without_loading_encoder(self):
        with patch("transformers.AutoTokenizer.from_pretrained") as tokenizer, patch("transformers.AutoModel.from_pretrained") as model:
            module = importlib.import_module("preprocess.build_note_text_emb")
            module = importlib.reload(module)
            tokenizer.assert_not_called()
            model.assert_not_called()
        for missing in (None, np.nan, pd.NA, pd.NaT):
            with self.subTest(missing=repr(missing)):
                self.assertEqual(module.build_text(missing, " body "), "content：body")
                self.assertEqual(module.build_text(" title ", missing), "title：title")
                self.assertEqual(module.build_text(missing, missing), "content：")
        self.assertEqual(module.build_text(" title ", " body "), "title：title\ncontent：body")
        self.assertEqual(module.build_text("0", None), "title：0")

    def test_fractional_raw_click_is_not_rounded_into_a_positive(self):
        path = self.dataset / "recommendation_train/train-00000-of-00001.parquet"
        frame = pd.read_parquet(path)
        frame.at[0, "rec_result_details_with_idx"][0]["click"] = .5
        frame.to_parquet(path, index=False)
        with self.assertRaisesRegex(ValueError, "click"):
            build_samples("rec", "train", dataset_dir=self.dataset, output_dir=self.root / "bad")

    def test_nonfinite_event_time_is_rejected_before_feature_output(self):
        path = self.dataset / "search_train/train-00000-of-00001.parquet"
        frame = pd.read_parquet(path)
        frame.at[0, "search_result_details_with_idx"][0]["search_timestamp"] = float("nan")
        frame.to_parquet(path, index=False)
        with self.assertRaisesRegex(ValueError, "timestamp"):
            build_samples("search", "train", dataset_dir=self.dataset, output_dir=self.root / "bad")

    def test_source_exposure_identity_survives_repeated_notes(self):
        frame = pd.read_parquet(self.samples / "rec_train_samples.parquet")
        self.assertIn("exposure_index", frame)
        self.assertEqual(frame.sort_values(["request_idx", "exposure_index"]).exposure_index.tolist(), [0, 1, 0])
        features = pd.read_parquet(self._features("rec"))
        self.assertEqual(set(zip(frame.request_idx, frame.exposure_index)),
                         set(zip(features.request_idx, features.exposure_index)))

    def test_fit_only_vocabulary_distinguishes_holdout_unknown_and_serving_matches(self):
        from preprocess.release import fit_vocabularies
        from backend.online.item_features import encode_static_items
        import pickle
        path = self.samples / "search_train_samples.parquet"
        frame = pd.read_parquet(path)
        frame.loc[frame.search_idx == 101, "location"] = "holdout_city"
        frame.to_parquet(path,index=False)
        fit_vocabularies(path,"search_idx",[100],self.vocab)
        output = pd.read_parquet(self._features())
        self.assertTrue((output.loc[output.search_idx==100,"location_enc"]>0).all())
        self.assertEqual(output.loc[output.search_idx==101,"location_enc"].tolist(),[0])
        self.assertEqual(output.loc[output.search_idx==101,"location_is_unknown"].tolist(),[1])
        self.assertEqual(output.loc[output.search_idx==101,"location_is_missing"].tolist(),[0])
        self.assertEqual(output.loc[output.search_idx==101,"location_is_unseen"].tolist(),[1])
        mappings = {name:pickle.loads((self.vocab/f"{name}.pkl").read_bytes()) for name in ["taxonomy1_id","taxonomy2_id","taxonomy3_id"]}
        served = encode_static_items(self.notes,mappings).set_index("note_idx")
        for name in mappings:
            np.testing.assert_array_equal(output[name+"_enc"],output.note_idx.map(served[name+"_enc"]))

    def test_missing_static_value_is_distinct_from_real_zero_and_history_repeats_survive(self):
        path = self.samples / "search_train_samples.parquet"
        frame = pd.read_parquet(path)
        frame.loc[frame.note_idx==0,"image_num"] = np.nan
        frame.loc[frame.note_idx==1,"image_num"] = 0.
        for idx in frame.index:
            frame.at[idx,"recent_clicked_note_idxs"] = [0,0,1]
        frame.to_parquet(path,index=False)
        output = pd.read_parquet(self._features())
        self.assertTrue((output.image_num==0).all())
        self.assertEqual(output.loc[output.note_idx==0,"image_num_is_missing"].tolist(),[1])
        self.assertTrue((output.loc[output.note_idx==1,"image_num_is_missing"]==0).all())
        self.assertTrue((output.history_length==3).all())
        self.assertTrue((output.history_unique_length==2).all())
        np.testing.assert_allclose(output.history_repeat_ratio,1/3)

    def test_catalog_static_fields_survive_actual_join_and_feature_build(self):
        for scene in ["search", "rec"]:
            sample = pd.read_parquet(self.samples / f"{scene}_train_samples.parquet")
            output = pd.read_parquet(self._features(scene))
            self.assertEqual(len(output), 3)
            for field in ["video_height", "video_width", "image_num"]:
                expected = sample.note_idx.map(self.notes.set_index("note_idx")[field]).to_numpy()
                np.testing.assert_array_equal(sample[field], expected)
                np.testing.assert_array_equal(output[field], expected)
            self.assertEqual(int(output.loc[output.note_idx == 0, "image_num"].iloc[0]), 3)

    def test_nan_page_time_preserves_real_clicks(self):
        frame = pd.read_parquet(self._features())
        self.assertTrue(np.isfinite(frame.y_multi).all())
        np.testing.assert_array_equal(frame.y_multi > 0, frame.click == 1)
        self.assertEqual(float(frame.loc[frame.note_idx == 0, "y_multi"].iloc[0]), 1.)

    def test_missing_static_catalog_field_fails_before_writing_output(self):
        self._save(self.notes.drop(columns=["image_num"]), "notes")
        destination = self.root / "bad_samples"
        with self.assertRaisesRegex(ValueError, "image_num"):
            build_samples("search", "train", dataset_dir=self.dataset, output_dir=destination)
        self.assertFalse((destination / "search_train_samples.parquet").exists())

    def test_legacy_sample_missing_static_field_is_rejected(self):
        path = self.samples / "search_train_samples.parquet"
        pd.read_parquet(path).drop(columns=["video_width"]).to_parquet(path, index=False)
        with self.assertRaisesRegex(ValueError, "video_width"):
            self._features()

    def test_optional_scene_counts_do_not_alias_global_counts(self):
        frame = pd.read_parquet(self._features())
        self.assertGreater(frame.imp_num.max(), 0)
        self.assertTrue((frame.imp_search_num == 0).all())
        self.assertTrue((frame.search_ctr == 0).all())

    def test_duplicate_catalog_ids_do_not_multiply_exposures(self):
        self._save(pd.concat([self.notes, self.notes.iloc[:1]], ignore_index=True), "notes")
        with self.assertRaisesRegex(ValueError, "unique nonnull"):
            build_samples("rec", "train", dataset_dir=self.dataset, output_dir=self.root / "bad_samples")

    def test_staging_does_not_replace_existing_output(self):
        frozen = self.samples / "search_train_samples.parquet"
        before = frozen.read_bytes()
        build_samples("search", "train", dataset_dir=self.dataset, output_dir=self.root / "new_samples")
        self._features()
        self.assertEqual(frozen.read_bytes(), before)
        self.assertTrue((self.features / "search_train_features.parquet").exists())


class FeatureHistoryTests(unittest.TestCase):
    def test_recent_prefix_preserves_zero_duplicates_and_array_input(self):
        np.testing.assert_array_equal(recent_history_ids(np.array([0, 1, 1, 2]), 3), [0, 1, 1])
        self.assertEqual(history_ids_oldest_first([0, 1, 1, 2], 3), [1, 1, 0])
        self.assertEqual(recent_history_ids(np.array([])), [])
        self.assertEqual(recent_history_ids([None, np.nan, -1, 0]), [0])
        with self.assertRaises(ValueError):
            recent_history_ids([1.5])

    def test_query_builder_import_and_missing_path_do_not_load_encoder(self):
        with patch("transformers.AutoModel.from_pretrained") as loader:
            module = importlib.import_module("preprocess.build_query_text_emb")
            self.assertEqual(module.DATA_DIR.name, "samples")
            with tempfile.TemporaryDirectory() as folder:
                with self.assertRaises(FileNotFoundError):
                    module.main("search", sample_dir=Path(folder), output_dir=Path(folder)/"out")
            loader.assert_not_called()

    def test_query_builder_stages_train_and_test_with_bound_history_contract(self):
        module = importlib.import_module("preprocess.build_query_text_emb")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            samples = root / "samples"
            vectors = root / "vectors"
            samples.mkdir(); vectors.mkdir()
            pd.DataFrame({"note_idx": [0, 1], "note_text_emb": [np.ones(EMB_DIM), np.full(EMB_DIM, 2.)]}).to_parquet(vectors / "part.parquet", index=False)
            for split in ["train", "test"]:
                pd.DataFrame({"request_idx": [7], "recent_clicked_note_idxs": [[0, 1]]}).to_parquet(samples / f"rec_{split}_samples.parquet", index=False)
                manifest_path = module.main("rec", split=split, sample_dir=samples, output_dir=root / "out", note_emb_dir=vectors)
                manifest = json.loads(manifest_path.read_text())
                self.assertTrue(manifest["completed"])
                self.assertEqual(manifest["sequence_order"], "oldest_first_v1")
                self.assertEqual(manifest["history_order"], "most_recent_first_v1")
                self.assertTrue(manifest["embedding_inputs"])
                self.assertTrue(manifest["outputs_sha256"])
                output = pd.read_parquet(root / "out" / f"rec_{split}-part-00000.parquet")
                self.assertEqual(output.request_idx.tolist(), [7])
                np.testing.assert_array_equal(np.stack(output.seq_embs.iloc[0])[:2, 0], [2., 1.])
            # Same immutable inputs resume; a changed source cannot silently reuse a shard.
            module.main("rec", sample_dir=samples, output_dir=root / "out", note_emb_dir=vectors)
            pd.DataFrame({"request_idx": [7], "recent_clicked_note_idxs": [[1, 0]]}).to_parquet(samples / "rec_train_samples.parquet", index=False)
            with self.assertRaisesRegex(ValueError, "manifest changed"):
                module.main("rec", sample_dir=samples, output_dir=root / "out", note_emb_dir=vectors)

    def test_query_builder_reverses_once_and_keeps_note_zero(self):
        module = importlib.import_module("preprocess.build_query_text_emb")
        store = {0: np.ones(EMB_DIM), 1: np.full(EMB_DIM, 2.)}
        output = module.build_sequence_embeddings([0, 1, 0], store, seq_len=2)
        np.testing.assert_array_equal(output[:, 0], [2., 1.])
        with self.assertRaisesRegex(ValueError, "Missing required history"):
            module.build_sequence_embeddings([3], store)


class FeatureEmbeddingFallbackTests(unittest.TestCase):
    def test_valid_parquet_and_missing_id_return_expected_tensors(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            pd.DataFrame({"note_idx": [0], "note_text_emb": [np.ones(EMB_DIM)]}).to_parquet(path/"part.parquet", index=False)
            store = EmbeddingStore(path, "note_idx", "note_text_emb", name="note_text")
            output = store.get_tensor([0, 999]).cpu().numpy()
            self.assertEqual(output.shape, (2, EMB_DIM))
            np.testing.assert_array_equal(output[0], np.ones(EMB_DIM))
            np.testing.assert_array_equal(output[1], np.zeros(EMB_DIM))

    def test_explicit_parquet_backend_ignores_available_mmap(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            pd.DataFrame({"note_idx": [0], "note_text_emb": [np.ones(EMB_DIM)]}).to_parquet(path/"part.parquet", index=False)
            (path/"note_text.bin").write_bytes(b"invalid")
            (path/"note_text_map.json").write_text(json.dumps({"0": 0}))
            store = EmbeddingStore(path, "note_idx", "note_text_emb", name="note_text", use_mmap=False)
            np.testing.assert_array_equal(store.get_tensor([0]).cpu().numpy()[0], np.ones(EMB_DIM))

    def test_sequence_parquet_fallback_pads_and_validates_dimensions(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            pd.DataFrame({"request_idx": [1], "seq_embs": [np.ones((2, EMB_DIM)).tolist()]}).to_parquet(path/"rec-part.parquet", index=False)
            store = EmbeddingStore(path, "request_idx", "seq_embs", name="rec_seq", is_sequence=True, prefix="rec")
            output = store.get_tensor([1, 2]).cpu().numpy()
            self.assertEqual(output.shape, (2, SEQ_LEN, EMB_DIM))
            np.testing.assert_array_equal(output[0, :2], np.ones((2, EMB_DIM)))
            self.assertEqual(float(output[:, 2:].sum()), 0.)
            pd.DataFrame({"request_idx": [1], "seq_embs": [[[1., 2.]]]}).to_parquet(path/"rec-part.parquet", index=False)
            with self.assertRaisesRegex(ValueError, "dimensions"):
                EmbeddingStore(path, "request_idx", "seq_embs", name="rec_seq", is_sequence=True, prefix="rec")


class PreprocessReleaseTests(unittest.TestCase):
    def test_spanning_requests_are_excluded_not_split_or_moved(self):
        frame = pd.DataFrame({"gid":[1,2,3,4,5],"first":[1,2,4,5,9],"last":[1,4,4,5,10]})
        got = partition_requests(frame, 10, .5)
        self.assertEqual(got["fit_ids"], [1])
        self.assertEqual(got["validation_ids"], [3,4])
        self.assertEqual(got["excluded_at_test_boundary"], 1)
        self.assertEqual(got["excluded_at_validation_boundary"], 1)
        self.assertEqual(got["validation_start"], 4)

    def test_empty_or_tied_intervals_fail_instead_of_random_split(self):
        frame = pd.DataFrame({"gid":[1,2],"first":[1,1],"last":[1,1]})
        with self.assertRaisesRegex(ValueError,"strict forward"):
            partition_requests(frame, 2)
        with self.assertRaises(ValueError):
            partition_requests(frame, float("nan"))

    def test_vocab_fit_excludes_validation_and_reserves_unknown_without_renumbering_old(self):
        from preprocess.build_features import USER_CAT, NOTE_CAT, prepare_category_mappings
        import duckdb
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sample = root/"train.parquet"
            data = {"request_idx":[1,2,3]}
            for name in USER_CAT+NOTE_CAT:
                data[name] = ["fit_only","validation_only",None]
            pd.DataFrame(data).to_parquet(sample,index=False)
            old = root/"old.pkl"
            old.write_bytes(pickle.dumps({"fit_only":0}))
            before = old.read_bytes()
            meta = fit_vocabularies(sample,"request_idx",[1],root/"new")
            for name in USER_CAT+NOTE_CAT:
                with (root/"new"/f"{name}.pkl").open("rb") as stream:
                    mapping = pickle.load(stream)
                self.assertEqual(mapping,{"fit_only":1})
                self.assertEqual(meta["categories"][name]["embedding_size"],2)
            self.assertEqual(old.read_bytes(),before)
            with duckdb.connect() as con:
                prepare_category_mappings(con,vocab_dir=root/"new")
                known = con.execute("SELECT platform_enc FROM map_platform WHERE raw_val='fit_only'").fetchone()[0]
                self.assertEqual(known,1)
            with self.assertRaises(FileExistsError):
                fit_vocabularies(sample,"request_idx",[1],root/"new")


if __name__ == "__main__":
    unittest.main()