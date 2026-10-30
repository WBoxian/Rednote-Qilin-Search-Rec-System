import hashlib,importlib.util,os,tempfile,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('reproduction',Path(__file__).resolve().parents[1]/'scripts/reproduce_rank_predictions.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class ReproductionInputTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.path=self.root/'vectors.bin';self.path.write_bytes(b'fixed-input')
        self.record={'vectors.bin':{'bytes':self.path.stat().st_size,'sha256':hashlib.sha256(self.path.read_bytes()).hexdigest()}}
    def test_same_bytes_with_different_mtime_are_reproducible(self):
        os.utime(self.path,ns=(1000000000,1000000000))
        module.verify_input_files(self.root,self.record)
    def test_changed_bytes_fail_even_if_size_and_mtime_are_preserved(self):
        before=self.path.stat();self.path.write_bytes(b'other-input')
        os.utime(self.path,ns=(before.st_atime_ns,before.st_mtime_ns))
        with self.assertRaises(ValueError):module.verify_input_files(self.root,self.record)
    def test_missing_input_fails_closed(self):
        self.path.unlink()
        with self.assertRaises(ValueError):module.verify_input_files(self.root,self.record)

if __name__=='__main__':unittest.main()