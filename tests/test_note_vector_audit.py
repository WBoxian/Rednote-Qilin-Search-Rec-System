import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from preprocess.audit_note_vectors import audit_note_vectors


class NoteVectorAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.binary = self.folder / "note.bin"
        self.mapping = self.folder / "note_map.json"
        np.array([[0, 0], [1, 0], [0, 1]], np.float16).tofile(self.binary)
        self.mapping.write_text(json.dumps({"2": 0, "0": 1, "1": 2}))

    def audit(self):
        return audit_note_vectors(self.folder, "note", catalog_size=3,
                                  dimension=2, batch_rows=2)

    def test_missing_modality_mask_follows_ids_including_note_zero(self):
        report, mask = self.audit()
        self.assertEqual(mask.tolist(), [False, False, True])
        self.assertEqual(report["zero_vectors"], 1)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(self.binary.stat().st_size, 12)

    def test_nonfinite_is_failure_even_when_geometry_is_valid(self):
        np.array([[np.nan, 0], [1, 0], [0, 1]], np.float16).tofile(self.binary)
        report, _ = self.audit()
        self.assertEqual(report["status"], "FAIL_NONFINITE")
        self.assertEqual(report["nonfinite_components"], 1)

    def test_wrong_binary_geometry_fails(self):
        self.binary.write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "geometry"):
            self.audit()

    def test_aliased_or_missing_catalog_ids_fail(self):
        for mapping in [{"0": 0, "1": 0, "2": 2}, {"0": 0, "1": 1, "3": 2},
                        {"00": 0, "1": 1, "2": 2}, {"0": False, "1": 1, "2": 2}]:
            with self.subTest(mapping=mapping):
                self.mapping.write_text(json.dumps(mapping))
                with self.assertRaises(ValueError):
                    self.audit()

    def test_duplicate_json_ids_do_not_silently_choose_last(self):
        self.mapping.write_text('{"0":0,"0":1,"1":1,"2":2}')
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.audit()


if __name__ == "__main__":
    unittest.main()
