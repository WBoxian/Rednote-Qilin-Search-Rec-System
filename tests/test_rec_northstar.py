import tempfile,unittest
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.optimize_rec_northstar import evaluate_recipes,write_candidates
from scripts.optimize_rec_retrieval import metrics
from recall.fusion import weighted_rrf

class NorthstarRecallGridTests(unittest.TestCase):
    def test_vectorized_grid_matches_real_rrf_and_independent_metrics(self):
        requests=pd.DataFrame({'gid':[1,2,3,4],'user_idx':[1,1,2,3],'positives':[[1,4],[6],[88],[]]})
        routes=[{1:{1:.9,2:.9,3:.5},2:{5:.4}},{1:{4:.8,2:.7},2:{6:.6}},{1:{2:.8,4:.7},2:{5:.5,6:.4}}]
        recipes=[{'weights':[1,1,1],'rrf_k':60},{'weights':[2,8,1],'rrf_k':10},{'weights':[1,8,1],'rrf_k':60}]
        with tempfile.TemporaryDirectory() as folder:
            paths=[]
            for route,data in enumerate(routes):
                rows=[]
                for gid,items in data.items():
                    for rank,(item,score) in enumerate(sorted(items.items(),key=lambda pair:(-pair[1],pair[0])),1):rows.append((gid,item,score,rank))
                path=Path(folder)/f'{route}.parquet'
                pd.DataFrame(rows,columns=['gid','note_idx','score','rank']).to_parquet(path,index=False);paths.append(path)
            evaluated,_=evaluate_recipes(requests,paths,recipes)
            for number,recipe in enumerate(recipes):
                expected=[]
                for gid in requests.gid:
                    result=weighted_rrf(*(route.get(gid,{}) for route in routes),topk=500,
                        route_weights=dict(zip(('ann','swing','usercf'),recipe['weights'])),rrf_k=recipe['rrf_k'])
                    expected.extend((gid,row[0],row[1],rank) for rank,row in enumerate(result,1))
                expected=pd.DataFrame(expected,columns=['gid','note_idx','score','rank'])
                independent,_=metrics(requests,expected)
                for metric in ('HitRate@50','Recall@100','known_positive_NDCG@10','MRR@100','Recall@500','HitRate@500'):self.assertAlmostEqual(evaluated[number][metric],independent[metric],places=13)
                destination=Path(folder)/f'selected{number}.parquet';write_candidates(requests,paths,recipe,destination)
                actual=pd.read_parquet(destination)
                self.assertEqual(actual[['gid','note_idx','rank']].to_dict('records'),expected[['gid','note_idx','rank']].to_dict('records'))
                self.assertTrue(np.allclose(actual.score,expected.score,rtol=0,atol=1e-15))
        self.assertEqual(evaluated[0]['all_requests'],4);self.assertEqual(evaluated[0]['positive_pairs'],4)

    def test_cf_latest_positive_timestamp_is_independent_of_input_row_order(self):
        from scripts.optimize_rec_retrieval import build_graph
        clicks=pd.DataFrame({'user_idx':[1,1,1],'note_idx':[10,20,10],
            'positive':[1,1,1],'hi':[100.,50.,10.]})
        requests=pd.DataFrame({'user_idx':[],'history':[],'hi':[]})
        forward=build_graph(clicks,requests,0.,max_items=1)
        reverse=build_graph(clicks.iloc[::-1],requests,0.,max_items=1)
        np.testing.assert_array_equal(forward[2],[10])
        np.testing.assert_array_equal(forward[2],reverse[2])
        np.testing.assert_array_equal(forward[0].toarray(),reverse[0].toarray())
if __name__=='__main__':unittest.main()