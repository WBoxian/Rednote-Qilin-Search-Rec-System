"""Local BM25 tests exercise corpus construction and actual FTS5 queries."""
from concurrent.futures import ThreadPoolExecutor
import json
import sqlite3
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from recall.lexical import BM25Index, build_index, tokenize


class LexicalRecallTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.notes=self.root/'notes';self.notes.mkdir()
        self.source=self.notes/'notes.parquet';self.path=self.root/'index.sqlite'
        self.frame=pd.DataFrame({'note_idx':[0,1,2],
            'note_title':['北京旅游攻略','旅游出行','厨房烘焙'],
            'note_content':['故宫 长城','北京周末旅行','烤箱蛋糕']})
        self.frame.to_parquet(self.source,index=False)

    def build(self,**kwargs):
        return build_index(self.notes,self.path,batch_size=2,**kwargs)

    def test_chinese_query_preserves_note_zero_and_score_direction(self):
        manifest=self.build()
        self.assertEqual(manifest['documents'],3)
        self.assertFalse(manifest['recipe']['labels_used'])
        self.assertEqual(manifest['recipe']['point_in_time_catalog_availability'],'UNKNOWN')
        with BM25Index(self.path) as index:
            result=index.search('北京旅游攻略',2)
            self.assertEqual(int(result.note_idx.iloc[0]),0)
            self.assertTrue((result.score>0).all())
            self.assertTrue(result.score.is_monotonic_decreasing)
            self.assertEqual(index.search('故宫',5).note_idx.tolist(),[0])
            self.assertIn(0,index.search('京',5).note_idx.tolist())

    def test_query_escaping_empty_and_unmatched_inputs(self):
        self.build()
        with BM25Index(self.path) as index:
            for query in ['"北京" OR NEAR(*) --','北京:攻略 NOT "','!!!','',None,'unmatchedword']:
                result=index.search(query,5)
                self.assertEqual(list(result.columns),['note_idx','score'])
                self.assertTrue(np.isfinite(result.score).all())
            for topk in [0,-1,1.5]:
                with self.assertRaises(ValueError):index.search('北京',topk)

    def test_title_weight_and_truncation_are_applied(self):
        pd.DataFrame({'note_idx':[0,1], 'note_title':['咖啡','饮品'], 'note_content':['饮品','咖啡']}).to_parquet(self.source,index=False)
        self.build(title_weight=2.,body_weight=1.)
        with BM25Index(self.path) as index:
            result=index.search('咖啡',2)
            self.assertEqual(result.note_idx.tolist(),[0,1])
            self.assertGreater(result.score.iloc[0],result.score.iloc[1])
        second=self.root/'truncated.sqlite'
        pd.DataFrame({'note_idx':[0], 'note_title':['标题'], 'note_content':['甲乙目标']}).to_parquet(self.source,index=False)
        build_index(self.notes,second,body_chars=2)
        with BM25Index(second) as index:self.assertTrue(index.search('目标',5).empty)

    def test_same_fingerprint_reuses_and_changed_corpus_cannot_overwrite(self):
        first=self.build();before=self.path.stat().st_mtime_ns
        again=self.build()
        self.assertEqual(first['index_sha256'],again['index_sha256'])
        self.assertEqual(before,self.path.stat().st_mtime_ns)
        self.frame.loc[0,'note_title']='上海旅行';self.frame.to_parquet(self.source,index=False)
        with self.assertRaises(FileExistsError):self.build()
        with BM25Index(self.path) as index:self.assertEqual(index.search('故宫',1).note_idx.tolist(),[0])

    def test_manifest_and_index_tampering_are_rejected(self):
        self.build();manifest_path=self.path.with_suffix('.sqlite.manifest.json')
        manifest=json.loads(manifest_path.read_text());manifest['recipe']['tokenizer']['version']='other'
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'incompatible'):BM25Index(self.path)
        manifest['recipe']['tokenizer']['version']='qilin_cjk_unigram_bigram_v1';manifest_path.write_text(json.dumps(manifest))
        with self.path.open('ab') as stream:stream.write(b'tamper')
        with self.assertRaisesRegex(ValueError,'content'):BM25Index(self.path)

    def test_duplicate_note_ids_fail_without_publishing_partial_index(self):
        self.frame.loc[1,'note_idx']=0;self.frame.to_parquet(self.source,index=False)
        with self.assertRaises(sqlite3.IntegrityError):self.build()
        self.assertFalse(self.path.exists())
        self.assertFalse(self.path.with_suffix('.sqlite.manifest.json').exists())
        self.assertFalse(self.path.with_suffix('.sqlite.lock').exists())

    def test_parallel_readers_and_repeat_results(self):
        self.build();index=BM25Index(self.path)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results=list(pool.map(lambda _:index.search('北京旅游',2).to_dict('list'),range(12)))
        self.assertTrue(all(result==results[0] for result in results))
        index.close()
        with self.assertRaises(RuntimeError):index.search('北京',1)

    def test_tokenizer_normalization_and_query_terms(self):
        self.assertEqual(tokenize('ＡＢＣ 北京',query=True),['abc','北京'])
        self.assertEqual(tokenize('北京'),['北','京','北京'])
        self.assertEqual(tokenize('北京北京',query=True),['北京','京北'])


if __name__=='__main__':unittest.main()