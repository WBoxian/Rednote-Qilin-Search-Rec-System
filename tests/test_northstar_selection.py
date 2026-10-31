import unittest
from training.selection import choose_rank_blend, choose_rank_model


def choice(weight, ndcg=.6, auc=.7, gauc=.72):
    return {'weight':weight,'validation':{'NDCG@10':ndcg,'AUC':auc,'GAUC':gauc}}


class NorthstarSelectionTests(unittest.TestCase):
    def test_ndcg_gain_does_not_hide_auc_regression(self):
        baseline=choice(0)
        self.assertIs(choose_rank_blend([baseline,choice(.5,ndcg=.65,auc=.69)]),baseline)

    def test_ndcg_gain_does_not_hide_gauc_regression(self):
        baseline=choice(0)
        self.assertIs(choose_rank_blend([baseline,choice(.5,ndcg=.65,gauc=.71)]),baseline)

    def test_valid_multimetric_gain_is_selected(self):
        baseline=choice(0);winner=choice(.5,ndcg=.63,auc=.71,gauc=.74)
        self.assertIs(choose_rank_blend([baseline,winner]),winner)

    def test_equal_quality_prefers_bypass(self):
        baseline=choice(0)
        self.assertIs(choose_rank_blend([choice(1),baseline]),baseline)

    def test_missing_or_undefined_baseline_rejected(self):
        with self.assertRaises(ValueError):choose_rank_blend([choice(.5)])
        with self.assertRaises(ValueError):choose_rank_blend([choice(0),choice(.5,auc=float('nan'))])

    def test_tree_selection_has_same_guardrails(self):
        baseline=choice(0);bad=choice(1,ndcg=.68,auc=.6);good=choice(2,ndcg=.62,auc=.71,gauc=.73)
        self.assertIs(choose_rank_model([baseline,bad,good]),good)

    def test_tree_no_candidates_rejected(self):
        with self.assertRaises(ValueError):choose_rank_model([])


if __name__=='__main__':unittest.main()