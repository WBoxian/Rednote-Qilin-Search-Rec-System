from contextlib import nullcontext
from dataclasses import FrozenInstanceError
from fractions import Fraction
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import faiss
import numpy as np

from recall import exact_ip_filter as module
from recall.exact_ip_filter import ExactIPFilter


class ExactIPFilterTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.vectors = rng.normal(size=(2500, 256)).astype(np.float32)
        self.vectors /= np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.query = self.vectors[17:18].copy()
        self.index = faiss.IndexFlatIP(256)
        self.index.add(self.vectors)

    def assert_raw_equal(self, first, second):
        np.testing.assert_array_equal(first[1], second[1])
        np.testing.assert_array_equal(first[0].view(np.uint32), second[0].view(np.uint32))

    def engine_for_refinement(self, index=None):
        engine = ExactIPFilter(index or self.index)
        engine._certificate = engine._certify()
        return engine

    def test_cpu_default_is_original_raw_search_and_restores_threads(self):
        previous = faiss.omp_get_max_threads()
        try:
            faiss.omp_set_num_threads(1)
            engine = ExactIPFilter(self.index)
            with patch.object(engine, '_prepare_cuda', side_effect=AssertionError('no CUDA')):
                actual = engine.search(self.query, 1000)
            faiss.omp_set_num_threads(4)
            expected = self.index.search(self.query, 1000)
            self.assert_raw_equal(actual, expected)
            faiss.omp_set_num_threads(1)
            self.assert_raw_equal(engine.search(self.query[0], 17), self.index.search(self.query, 17))
            self.assertEqual(faiss.omp_get_max_threads(), 1)
            self.assertEqual(engine.diagnostics.fallback_reason, 'cuda_disabled')
        finally:
            faiss.omp_set_num_threads(previous)

    def test_invalid_query_index_and_k_reject(self):
        for index in [faiss.IndexFlatL2(256), faiss.IndexFlatIP(128), faiss.IndexIDMap(faiss.IndexFlatIP(256))]:
            with self.assertRaises(ValueError):
                ExactIPFilter(index)
        engine = ExactIPFilter(self.index)
        for q in [self.query.astype(np.float64), np.zeros((3, 256), np.float32),
                  np.full((1, 256), np.nan, np.float32), np.full(256, np.inf, np.float32), self.query.tolist()]:
            with self.assertRaises(ValueError):
                engine.search(q)
        for k in [True, 1.5, 0, 2501]:
            with self.assertRaises(ValueError):
                engine.search(self.query, k)
        self.index.add(self.query)
        with self.assertRaisesRegex(ValueError, 'modified'):
            engine.search(self.query)

    def test_norm_certificate_covers_exact_fp32_row_sum_and_is_immutable(self):
        engine = self.engine_for_refinement()
        row = self.vectors[np.argmax(np.sum(self.vectors.astype(np.float64)**2, axis=1))]
        exact_sum = sum((Fraction.from_float(float(x))**2 for x in row), Fraction(0))
        self.assertGreaterEqual(engine.norm_certificate.upper**2, exact_sum)
        self.assertEqual(engine.norm_certificate.rows, 2500)
        with self.assertRaises(FrozenInstanceError):
            engine.norm_certificate.rows = 1
        for bad in [np.nan, np.float32(1e-40), 3.]:
            x = np.zeros((1, 256), np.float32)
            x[0, 0] = bad
            index = faiss.IndexFlatIP(256)
            index.add(x)
            with self.assertRaises(ValueError):
                ExactIPFilter(index)._certify()

    def test_cutoff_is_downward_for_nonrepresentable_boundary(self):
        delta = Fraction(1, 2**80)
        cutoff = module._cutoff(np.float32(1), delta)
        self.assertLessEqual(Fraction.from_float(float(cutoff)), Fraction(1) - 2*delta)
        self.assertEqual(cutoff, np.nextafter(np.float32(1), np.float32(-np.inf)))

    def test_perturbed_gpu_order_refines_to_original_raw_cpu_output(self):
        engine = self.engine_for_refinement()
        scores = engine._subset(self.query, np.arange(2500, dtype=np.int64))
        # Simulated GPU arithmetic, not model scores: both signs change numeric order.
        perturbation = np.random.default_rng(7).choice([-1., 1.], 2500).astype(np.float32)*np.float32(1e-7)
        approximate = (scores + perturbation).astype(np.float32)
        upper = module._norm_upper(np.einsum('ij,ij->', self.query.astype(np.float64), self.query.astype(np.float64)))
        with module._cpu_threads():
            self.assert_raw_equal(engine._refine(self.query, approximate, upper), self.index.search(self.query, 1000))
        self.assertEqual(engine.diagnostics.path, 'cuda_filter_cpu_refine')
        self.assertGreaterEqual(engine.diagnostics.candidate_count, 1000)

    def test_boundary_tie_falls_back_but_internal_ties_preserve_raw_order(self):
        query = np.zeros((1, 256), np.float32)
        query[0, 0] = 1
        for top_count in [1000, 1001, 2500]:
            vectors = np.zeros((2500, 256), np.float32)
            vectors[:top_count, 0] = 1
            index = faiss.IndexFlatIP(256)
            index.add(vectors)
            engine = self.engine_for_refinement(index)
            with module._cpu_threads():
                actual = engine._refine(query, vectors[:, 0].copy(), Fraction(1))
                self.assert_raw_equal(actual, index.search(query, 1000))
            self.assertEqual(engine.diagnostics.fallback_reason, 'cpu_boundary_tie' if top_count > 1000 else None)

    def test_excess_candidates_or_numeric_bound_violation_keep_full_cpu(self):
        engine = self.engine_for_refinement()
        scores = engine._subset(self.query, np.arange(2500, dtype=np.int64))
        with module._cpu_threads():
            for approximate, expected in [(scores + np.float32(.1), 'arithmetic_bound_violation'),
                                          (np.full(2500, np.nan, np.float32), 'unsupported_gpu_scores')]:
                self.assert_raw_equal(engine._refine(self.query, approximate, Fraction(1)), self.index.search(self.query, 1000))
                self.assertEqual(engine.diagnostics.fallback_reason, expected)
            with patch.object(module, '_CAP', 999):
                self.assert_raw_equal(engine._refine(self.query, scores, Fraction(1)), self.index.search(self.query, 1000))
            self.assertEqual(engine.diagnostics.fallback_reason, 'candidate_cap')

    def test_zero_subnormal_norm_and_other_k_do_not_initialize_cuda(self):
        engine = ExactIPFilter(self.index, enable_cuda=True)
        zero = np.zeros((1, 256), np.float32)
        subnormal = zero.copy()
        subnormal[0, 0] = np.float32(1e-40)
        large = zero.copy()
        large[0, 0] = 3
        with patch.object(engine, '_prepare_cuda', side_effect=AssertionError('no CUDA')):
            for query, k, reason in [(zero, 1000, 'zero_query'), (subnormal, 1000, 'query_subnormal'),
                                     (large, 1000, 'query_norm_domain'), (self.query, 7, 'unsupported_k')]:
                with module._cpu_threads():
                    self.assert_raw_equal(engine.search(query, k), self.index.search(query, k))
                self.assertEqual(engine.diagnostics.fallback_reason, reason)

    def test_changed_faiss_dispatch_or_rounding_mode_use_original_cpu(self):
        engine = ExactIPFilter(self.index, enable_cuda=True)
        previous = faiss.cvar.distance_compute_blas_threshold
        try:
            faiss.cvar.distance_compute_blas_threshold = 21
            with patch.object(engine, '_prepare_cuda', side_effect=AssertionError('no CUDA')):
                engine.search(self.query)
            self.assertEqual(engine.diagnostics.fallback_reason, 'unsupported_faiss_dispatch')
        finally:
            faiss.cvar.distance_compute_blas_threshold = previous
        with patch.object(module, '_round_to_nearest', return_value=False):
            engine.search(self.query)
        self.assertEqual(engine.diagnostics.fallback_reason, 'unsupported_rounding_mode')

    def test_cuda_flags_restore_when_operation_raises(self):
        fake = SimpleNamespace(precision='high', backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True))))
        fake.get_float32_matmul_precision = lambda: fake.precision
        fake.set_float32_matmul_precision = lambda value: setattr(fake, 'precision', value)
        fake.deterministic = False
        fake.warn_only = True
        fake.are_deterministic_algorithms_enabled = lambda: fake.deterministic
        fake.is_deterministic_algorithms_warn_only_enabled = lambda: fake.warn_only
        def set_deterministic(value, warn_only=False):
            fake.deterministic, fake.warn_only = value, warn_only
        fake.use_deterministic_algorithms = set_deterministic
        fake.no_grad = lambda: nullcontext()
        fake.autocast = lambda *args, **kwargs: nullcontext()
        with self.assertRaises(RuntimeError):
            with module._fp32_cuda(fake):
                self.assertEqual(fake.precision, 'highest')
                self.assertFalse(fake.backends.cuda.matmul.allow_tf32)
                raise RuntimeError('inference failure')
        self.assertEqual(fake.precision, 'high')
        self.assertTrue(fake.backends.cuda.matmul.allow_tf32)
        self.assertFalse(fake.deterministic)
        self.assertTrue(fake.warn_only)


if __name__ == '__main__':
    unittest.main()
