"""Optional fixed-runtime GPU screening with original Faiss CPU IP refinement.

CPU is the default. GPU admissibility is conditional on ordinary IEEE FP32
round-to-nearest dot products with <=512 rounded operations per error path.
It is not a binary-level proof of cuBLAS or a guarantee for arbitrary runtimes.
The supplied IndexFlatIP must remain read-only for this engine's lifetime.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import os
import ctypes
from math import isqrt
import threading

import faiss
import numpy as np

_DIM = 256
_K = 1000
_CAP = 65536
_LOCK = threading.RLock()
_U = Fraction(1, 2**24)
_GAMMA = 512 * _U / (1 - 512 * _U)
_UNDERFLOW = 512 * Fraction(1, 2**126) / (1 - 512 * _U)


@dataclass(frozen=True)
class SearchDiagnostics:
    path: str
    fallback_reason: str | None = None
    candidate_count: int | None = None
    delta: float | None = None


@dataclass(frozen=True)
class NormCertificate:
    numerator: int
    denominator: int
    rows: int
    source_sha256: str

    @property
    def upper(self):
        return Fraction(self.numerator, self.denominator)


def _norm_upper(observed_sum):
    """Upward dyadic sqrt, including a strict FP64 positive-reduction bound."""
    bound = Fraction.from_float(float(observed_sum)) / (1 - Fraction(1, 2**40))
    scaled = bound * 2**40
    integer = (scaled.numerator + scaled.denominator - 1) // scaled.denominator
    root = isqrt(integer)
    root += root * root < integer
    return Fraction(root, 2**20)


def _cutoff(tau, delta):
    exact = Fraction.from_float(float(tau)) - 2 * delta
    value = np.float32(float(exact))
    while Fraction.from_float(float(value)) > exact:
        value = np.nextafter(value, np.float32(-np.inf), dtype=np.float32)
    return value


try:
    _FENV = ctypes.CDLL('libm.so.6')
    _FENV.fegetround.restype = ctypes.c_int
except (OSError, AttributeError):
    _FENV = None


def _round_to_nearest():
    return _FENV is not None and _FENV.fegetround() == 0


@contextmanager
def _cpu_threads():
    previous = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(4)
    try:
        yield
    finally:
        faiss.omp_set_num_threads(previous)


@contextmanager
def _fp32_cuda(torch):
    precision = torch.get_float32_matmul_precision()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        with torch.no_grad(), torch.autocast('cuda', enabled=False):
            yield
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


class ExactIPFilter:
    """Faiss-style single-query search, returning (scores, original row IDs).

    Only the validated Faiss 1.7.4 AVX2 / Torch 2.9.1 CUDA 12.8 / RTX 5070 Ti
    combination, CUBLAS_WORKSPACE_CONFIG=:4096:8 and k=1000 can use GPU screening. Unsupported cases keep the
    original four-thread B1 CPU search. Three heads must call search three times;
    silently switching to a B3 dot-product kernel would change numeric semantics.
    A full-table finite/subnormal/norm certificate is computed before CUDA copy.
    No externally supplied or empirical norm certificate is accepted.
    """

    def __init__(self, index, *, enable_cuda=False):
        if (not isinstance(index, faiss.IndexFlat)
                or index.metric_type != faiss.METRIC_INNER_PRODUCT
                or index.d != _DIM):
            raise ValueError('require an original 256-dimensional IndexFlatIP')
        if not isinstance(enable_cuda, bool):
            raise ValueError('enable_cuda must be bool')
        self._index = index
        self._identity = (index.ntotal, int(index.get_xb()))
        self._vectors = faiss.rev_swig_ptr(index.get_xb(), index.ntotal * _DIM).reshape(-1, _DIM)
        self._vectors.flags.writeable = False
        self._enable_cuda = enable_cuda
        self._torch = self._gpu = self._certificate = None
        self._cuda_unavailable = None
        self._diagnostics = SearchDiagnostics('not_searched')

    @property
    def diagnostics(self):
        return self._diagnostics

    @property
    def norm_certificate(self):
        return self._certificate

    def close(self):
        """Release owned CUDA tensors; do not alter another caller's CUDA cache."""
        with _LOCK:
            self._gpu = None

    def _check_identity(self):
        if self._identity != (self._index.ntotal, int(self._index.get_xb())):
            raise ValueError('the read-only index was modified')

    def _cpu(self, query, k, reason, count=None, delta=None):
        self._diagnostics = SearchDiagnostics('cpu', reason, count, delta)
        return self._index.search(query, k)

    def _certify(self):
        maximum = 0.
        tiny = np.finfo(np.float32).tiny
        for first in range(0, len(self._vectors), 8192):
            block = self._vectors[first:first + 8192]
            absolute = np.abs(block)
            if not np.isfinite(block).all() or ((absolute > 0) & (absolute < tiny)).any():
                raise ValueError('unsupported catalog numeric domain')
            double = block.astype(np.float64)
            sums = np.einsum('ij,ij->i', double, double, optimize=False)
            maximum = max(maximum, float(sums.max(initial=0)))
        upper = _norm_upper(maximum)
        if upper > 2:
            raise ValueError('unsupported catalog norm domain')
        digest = hashlib.sha256(memoryview(self._vectors).cast('B')).hexdigest()
        return NormCertificate(upper.numerator, upper.denominator, len(self._vectors), digest)

    def _prepare_cuda(self):
        if self._gpu is not None:
            return None
        if self._cuda_unavailable:
            return self._cuda_unavailable
        try:
            import torch
            self._torch = torch
            if (faiss.__version__ != '1.7.4' or 'AVX2' not in faiss.get_compile_options()
                    or torch.__version__ != '2.9.1+cu128' or torch.version.cuda != '12.8'
                    or not torch.cuda.is_available()):
                self._cuda_unavailable = 'unsupported_runtime'
                return self._cuda_unavailable
            if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
                self._cuda_unavailable = 'unsupported_cublas_config'
                return self._cuda_unavailable
            if ('5070 Ti' not in torch.cuda.get_device_name(0)
                    or torch.cuda.get_device_capability(0) != (12, 0)):
                self._cuda_unavailable = 'unsupported_device'
                return self._cuda_unavailable
            if self._vectors.nbytes + 16 * len(self._vectors) >= 3 * 1024**3:
                return 'catalog_exceeds_cuda_budget'
            if torch.cuda.mem_get_info(0)[0] < 3 * 1024**3:
                return 'insufficient_cuda_memory'
            certificate = self._certify()
            with _fp32_cuda(torch):
                self._gpu = torch.from_numpy(self._vectors.copy()).to('cuda:0')
            if hashlib.sha256(memoryview(self._vectors).cast('B')).hexdigest() != certificate.source_sha256:
                self._gpu = None
                raise ValueError('catalog changed during CUDA preparation')
            self._certificate = certificate
            return None
        except (ImportError, RuntimeError, ValueError) as error:
            self._gpu = None
            self._cuda_unavailable = 'cuda_preparation_' + type(error).__name__
            return self._cuda_unavailable

    def _subset(self, query, rows):
        rows = np.ascontiguousarray(rows, np.int64).reshape(1, -1)
        values = np.empty(rows.shape, np.float32)
        self._index.compute_distance_subset(1, faiss.swig_ptr(query), rows.shape[1],
                                           faiss.swig_ptr(values), faiss.swig_ptr(rows))
        return values[0]

    @staticmethod
    def _heap(values, rows):
        values = np.ascontiguousarray(values, np.float32).reshape(1, -1)
        rows = np.ascontiguousarray(rows, np.int64).reshape(1, -1)
        scores = np.empty((1, _K), np.float32)
        ids = np.empty((1, _K), np.int64)
        heap = faiss.float_minheap_array_t()
        heap.nh, heap.k = 1, _K
        heap.val, heap.ids = faiss.swig_ptr(scores), faiss.swig_ptr(ids)
        heap.heapify()
        heap.addn_with_ids(rows.shape[1], faiss.swig_ptr(values), faiss.swig_ptr(rows))
        heap.reorder()
        return scores, ids

    def _refine(self, query, gpu_scores, query_upper):
        """Shared numeric core; receives all GPU scores, never reference CPU top-K."""
        if (gpu_scores.dtype != np.float32 or gpu_scores.shape != (len(self._vectors),)
                or not np.isfinite(gpu_scores).all()):
            return self._cpu(query, _K, 'unsupported_gpu_scores')
        delta = 2 * (_GAMMA * self._certificate.upper * query_upper + _UNDERFLOW)
        tau = np.partition(gpu_scores, len(gpu_scores) - _K)[-_K]
        rows = np.flatnonzero(gpu_scores >= _cutoff(tau, delta)).astype(np.int64)
        if len(rows) > _CAP:
            return self._cpu(query, _K, 'candidate_cap', len(rows), float(delta))
        values = self._subset(query, rows)
        difference = float(np.abs(gpu_scores[rows].astype(np.float64) - values.astype(np.float64)).max())
        if not np.isfinite(values).all() or Fraction.from_float(difference) > delta:
            return self._cpu(query, _K, 'arithmetic_bound_violation', len(rows), float(delta))
        boundary = np.partition(values, len(values) - _K)[-_K]
        if int((values == boundary).sum()) > _K - int((values > boundary).sum()):
            return self._cpu(query, _K, 'cpu_boundary_tie', len(rows), float(delta))
        result = self._heap(values, rows)
        self._diagnostics = SearchDiagnostics('cuda_filter_cpu_refine', None, len(rows), float(delta))
        return result

    def search(self, query, k=1000):
        if (not isinstance(query, np.ndarray) or query.dtype != np.float32
                or query.shape not in [(_DIM,), (1, _DIM)] or not np.isfinite(query).all()):
            raise ValueError('query must be finite FP32 [256] or [1,256]; batch search is unsupported')
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or not 1 <= k <= self._index.ntotal:
            raise ValueError('k must be an integer within the original catalog size')
        query = np.ascontiguousarray(query.reshape(1, _DIM))
        with _LOCK, _cpu_threads():
            self._check_identity()
            if not self._enable_cuda:
                return self._cpu(query, k, 'cuda_disabled')
            if k != _K:
                return self._cpu(query, k, 'unsupported_k')
            absolute = np.abs(query)
            if not absolute.any():
                return self._cpu(query, k, 'zero_query')
            if ((absolute > 0) & (absolute < np.finfo(np.float32).tiny)).any():
                return self._cpu(query, k, 'query_subnormal')
            double = query[0].astype(np.float64)
            upper = _norm_upper(np.einsum('i,i->', double, double, optimize=False))
            if upper > 2:
                return self._cpu(query, k, 'query_norm_domain')
            if (faiss.cvar.distance_compute_blas_threshold != 20
                    or faiss.cvar.distance_compute_min_k_reservoir != 100):
                return self._cpu(query, k, 'unsupported_faiss_dispatch')
            if not _round_to_nearest():
                return self._cpu(query, k, 'unsupported_rounding_mode')
            reason = self._prepare_cuda()
            if reason:
                return self._cpu(query, k, reason)
            try:
                with _fp32_cuda(self._torch):
                    scores = self._torch.mv(self._gpu, self._torch.from_numpy(query[0]).to('cuda:0'))
                    gpu_scores = scores.cpu().numpy()
                return self._refine(query, gpu_scores, upper)
            except RuntimeError:
                return self._cpu(query, k, 'cuda_search_RuntimeError')
