import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from backend.online.note_metadata import COLUMNS, NoteMetadataIndex, build_note_metadata_index
from backend.online.pipeline import SceneServingState
from backend.online.recall.dssm import _NoteRows, _SharedNoteStore
from recall import dssm_trainer


class NoteMetadataTests(unittest.TestCase):
    def test_actual_state_metadata_matches_parquet_scan_without_label_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            notes = root/'notes';notes.mkdir()
            rows = [{name: None for name in COLUMNS} for _ in range(2)]
            rows[0].update(note_idx=0, note_title='可爱猫', note_content='内容', image_path=['a.jpg'], accum_like_num=7, taxonomy1_id='cat')
            rows[1].update(note_idx=7, note_title='', note_content='', image_path=[])
            pd.DataFrame(rows).assign(click=999, y_multi=999).to_parquet(notes/'notes.parquet', index=False)
            path = root/'metadata.sqlite'
            manifest = build_note_metadata_index(notes, path)
            self.assertFalse(manifest['labels_used'])
            reader = NoteMetadataIndex(path)
            try:
                state = SceneServingState.__new__(SceneServingState)
                state._note_cache_lock = threading.Lock();state._note_meta_cache = {};state.notes_glob = str(notes/'*.parquet')
                state.bundle_active = False
                before = state._fetch_notes([7, 0, 0, 999])
                state._note_meta_cache.clear();state.bundle_active = True
                state.bundle = SimpleNamespace(manifest={'note_metadata_index': str(path)}, _get_note_metadata_index=lambda: reader)
                with patch('backend.online.pipeline.duckdb.connect', side_effect=AssertionError('Indexed metadata must not scan parquet')):
                    after = state._fetch_notes([7, 0, 0, 999])
                pd.testing.assert_frame_equal(before, after, check_dtype=False)
                self.assertNotIn('click', after)
                self.assertNotIn('y_multi', after)
                self.assertEqual(reader.fetch([]).columns.tolist(), list(COLUMNS))
                self.assertTrue(reader.fetch([999]).empty)
                with self.assertRaises(FileExistsError):
                    build_note_metadata_index(notes, path)
            finally:
                reader.close()

    def test_compact_ann_mapping_preserves_zero_unknown_and_arbitrary_row_order(self):
        values = np.asarray([9, 0, 5], dtype=np.int64)
        lookup = _NoteRows(values)
        self.assertEqual([lookup.get(note) for note in values], [0, 1, 2])
        self.assertIsNone(lookup.get(999))
        self.assertEqual(lookup.get(-1, -7), -7)
        with self.assertRaises(ValueError):
            _NoteRows(np.asarray([1, 1]))

    def test_shared_note_adapter_matches_original_mmap_for_history_and_missing_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root/'embeddings/note_text_emb';folder.mkdir(parents=True)
            (folder/'note_text_map.json').write_text(json.dumps({'0': 1, '7': 0}))
            values = np.arange(2*768, dtype=np.float16).reshape(2, 768)
            values.tofile(folder/'note_text.bin')
            with patch.object(dssm_trainer, 'EMB_DIR', root/'embeddings'), patch('backend.online.recall.dssm.BASE_DIR', root):
                original = dssm_trainer.MMapStore('note_text_emb', 'note_text')
                shared = _SharedNoteStore('text')
                ids = [7, 0, 99, -1]
                np.testing.assert_array_equal(original.get_batch_vecs(ids).numpy(), shared.get_batch_vecs(ids).numpy())
                np.testing.assert_array_equal(original.get_vec(0).numpy(), shared.get_vec(0).numpy())


if __name__ == '__main__':
    unittest.main()
