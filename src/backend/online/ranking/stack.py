"""Serve the same versioned feature and scoring contract used by temporal ranking."""
from __future__ import annotations

import json
import math
from collections import OrderedDict
from threading import RLock
from functools import wraps, lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from backend.online.item_features import STATIC_ITEM_FEATURES, get_item_feature_store
from backend.online.shared_query_encoder import get_shared_query_encoder, encode_current_query
from training.dien_ranker import DIENRanker
from training.rank_stack import MappedVectors, RequestContext, SEMANTIC_FEATURES, group_minmax

ROOT = Path(__file__).resolve().parents[4]


def validate_rank_prior(prior):
    if prior is None:
        return
    try:
        valid = (isinstance(prior, dict) and set(prior) == {'alpha', 'k'} and
                 not isinstance(prior['alpha'], bool) and math.isfinite(float(prior['alpha'])) and
                 float(prior['alpha']) >= 0 and prior['k'] == 60)
    except (ValueError, TypeError):
        valid = False
    if not valid:
        raise ValueError('Unsupported frozen recall-rank prior')


def synchronized(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return call


_VECTOR_LOCK = RLock()


@lru_cache(maxsize=8)
def _mapped_vectors_cached(folder, name, file_identity):
    vectors = MappedVectors(Path(folder), name)
    vectors.ids.flags.writeable = False
    vectors.positions.flags.writeable = False
    return vectors


def shared_mapped_vectors(folder, name):
    folder = Path(folder).resolve()
    files = [folder / f"{name}_map.json", folder / f"{name}.bin"]
    metadata = [path.stat() for path in files]
    identity = tuple((info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_ino, info.st_dev) for info in metadata)
    with _VECTOR_LOCK:
        return _mapped_vectors_cached(str(folder), name, identity)


class RankStackPredictor:
    """One frozen tree/neural bundle; never infer missing model inputs as zero."""

    def __init__(self, scene: str, run_dir: Path, *, root: Path = ROOT, device=None, query_encoder_dir=None, query_encoder_revision=None):
        self._lock = RLock()
        self.query_encoder_dir = query_encoder_dir
        self.query_encoder_revision = query_encoder_revision
        self.scene, self.root = scene, Path(root)
        self.run_dir = Path(run_dir)
        self.manifest = json.loads((self.run_dir / 'manifest.json').read_text())
        if self.manifest['scene'] != scene:
            raise ValueError('Ranking bundle scene mismatch')
        self.group_key = self.manifest['key']
        self.feature_cols = list(self.manifest['features'])
        allowed = {*STATIC_ITEM_FEATURES, *SEMANTIC_FEATURES}
        if scene == 'search':
            allowed.add('query_item_cosine')
        if set(self.feature_cols) - allowed:
            raise ValueError('Unsupported or unsafe ranking bundle features')
        self.num_feats = list(self.manifest['dien']['numeric_features'])
        if self.num_feats != [*self.feature_cols, 'coarse_score']:
            raise ValueError('Fine rank features do not match the teacher contract')
        architecture = self.manifest['dien']['architecture']
        if architecture.get('inference_precision') != 'fp32':
            raise ValueError('Serving rank bundles require explicit FP32 inference')
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        contract = self.manifest['dien'].get('score_contract', 'request_minmax_v1')
        if contract not in {'request_minmax_v1', 'probability_v2'}:
            raise ValueError('Unsupported ranking score contract')
        self.coarse = lgb.Booster(model_file=str(self.run_dir / 'gbdt.txt'))
        if contract == 'probability_v2' and self.coarse.params.get('objective') != 'binary':
            raise ValueError('Probability fusion requires a binary-classifier teacher')
        if self.coarse.feature_name() != self.feature_cols:
            raise ValueError('Tree feature order differs from manifest')
        self.lgb_model, self.xgb_model = self.coarse, None
        self.model = DIENRanker(scene, self.num_feats, **architecture).to(self.device)
        self.model.load_state_dict(torch.load(self.run_dir / 'dien.pt', map_location=self.device, weights_only=True))
        self.model.eval()
        self.weight = float(self.manifest['dien']['chosen']['weight'])
        self.score_contract = self.manifest['dien'].get('score_contract', 'request_minmax_v1')
        self.rank_prior = self.manifest['dien'].get('rank_prior')
        validate_rank_prior(self.rank_prior)
        if not 0 <= self.weight <= 1:
            raise ValueError('Invalid frozen blend weight')
        self.text = shared_mapped_vectors(self.root / 'embeddings/note_text_emb', 'note_text')
        self.image = shared_mapped_vectors(self.root / 'embeddings/note_img_emb', 'note_img')
        self.static = get_item_feature_store(self.root)
        self._query_cache = OrderedDict()
        self._context_cache = OrderedDict()

    def _query_vector(self, query):
        text = str(query or '').strip()
        if not text:
            return np.zeros(768, dtype=np.float32)
        if self.query_encoder_dir is not None:
            return encode_current_query(text, self.query_encoder_dir, self.query_encoder_revision)
        if text in self._query_cache:
            self._query_cache.move_to_end(text)
            return self._query_cache[text]
        tokenizer, model = (get_shared_query_encoder(self.query_encoder_dir, self.query_encoder_revision)
                            if self.query_encoder_dir is not None else get_shared_query_encoder())
        if tokenizer is None or model is None:
            raise RuntimeError('Current-query encoder is unavailable')
        with torch.no_grad():
            inputs = tokenizer([text], padding=True, truncation=True, max_length=512, return_tensors='pt')
            vector = model(**inputs).last_hidden_state[:, 0].float()
            vector = vector / vector.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        result = vector.cpu().numpy()[0]
        self._query_cache[text] = result
        if len(self._query_cache) > 1024:
            self._query_cache.popitem(last=False)
        return result

    def _context(self, frame):
        required = {self.group_key, 'user_idx', 'query', 'recent_clicked_note_idxs'}
        if not required.issubset(frame):
            raise ValueError(f'Missing live ranking context: {sorted(required - set(frame))}')
        requests = []
        fingerprints = []
        for gid, group in frame.groupby(self.group_key, sort=False):
            first = group.iloc[0]
            query = str(first.get('raw_query', first['query']) or '').strip()
            history = tuple(int(x) for x in first.recent_clicked_note_idxs if int(x) >= 0)
            if group.user_idx.nunique() != 1 or group['query'].nunique(dropna=False) != 1:
                raise ValueError('Request contains conflicting user/query contexts')
            if any(tuple(int(x) for x in value if int(x) >= 0) != history for value in group.recent_clicked_note_idxs):
                raise ValueError('Request contains conflicting histories')
            fingerprints.append((int(gid), int(first.user_idx), query, history))
            requests.append({self.group_key: int(gid), 'recent_clicked_note_idxs': list(history), 'query': query})
        cache_key = tuple(fingerprints)
        if cache_key in self._context_cache:
            self._context_cache.move_to_end(cache_key)
            return self._context_cache[cache_key]
        request_frame = pd.DataFrame(requests)
        vectors = np.stack([self._query_vector(x['query']) for x in requests]) if self.scene == 'search' else None
        context = RequestContext(request_frame, self.group_key, self.text, self.image, vectors)
        self._context_cache[cache_key] = context
        if len(self._context_cache) > 64:
            self._context_cache.popitem(last=False)
        return context

    def _frame(self, candidates, history_note_ids=None):
        frame = candidates.copy().reset_index(drop=True)
        if history_note_ids is not None:
            if frame[self.group_key].nunique() != 1:
                raise ValueError('A history override must identify exactly one request')
            frame['recent_clicked_note_idxs'] = [list(history_note_ids) for _ in range(len(frame))]
        context = self._context(frame)
        frame = context.feature_frame(frame, self.static)
        if not np.isfinite(frame[self.feature_cols].to_numpy(dtype=np.float32)).all():
            raise ValueError('Nonfinite required ranking features')
        return frame, context

    @synchronized
    def predict_with_components(self, candidates):
        if candidates.empty:
            return {key: np.zeros(0, dtype=np.float32) for key in ('lgb_score', 'xgb_score', 'gbdt_score')}
        frame, _ = self._frame(candidates)
        score = self.coarse.predict(frame[self.feature_cols]).astype(np.float32)
        if not np.isfinite(score).all():
            raise ValueError('Nonfinite coarse prediction')
        return {'lgb_score': score, 'xgb_score': np.zeros_like(score), 'gbdt_score': score}

    @synchronized
    def predict(self, candidates, history_note_ids=None):
        if candidates.empty:
            return np.zeros(0, dtype=np.float32)
        frame, context = self._frame(candidates, history_note_ids)
        frame['coarse_score'] = self.coarse.predict(frame[self.feature_cols]).astype(np.float32)
        if not self.weight:
            from training.scoring import blend_rank_scores
            score = blend_rank_scores(frame.coarse_score.to_numpy(), np.zeros(len(frame), dtype=np.float32),
                frame[self.group_key].to_numpy(), 0., getattr(self, "score_contract", "request_minmax_v1"))
            return self._apply_rank_prior(frame, score)
        chunks = []
        with torch.no_grad():
            for start in range(0, len(frame), 1024):
                batch = frame.iloc[start:start + 1024]
                raw, _ = self.model(context.batch(batch, self.num_feats, self.device))
                chunks.append(raw.detach().float().cpu().numpy().reshape(-1))
        raw = np.concatenate(chunks)
        from training.scoring import blend_rank_scores
        score = blend_rank_scores(frame.coarse_score.to_numpy(), raw, frame[self.group_key].to_numpy(),
            self.weight, self.score_contract)
        if not np.isfinite(score).all():
            raise ValueError('Nonfinite fine prediction')
        return self._apply_rank_prior(frame, score)

    def _apply_rank_prior(self, frame, score):
        prior = getattr(self, 'rank_prior', None)
        if prior is None:
            return np.asarray(score, dtype=np.float32)
        if 'rank' not in frame or (pd.to_numeric(frame['rank'], errors='coerce') <= 0).any():
            raise ValueError('Recall-rank prior requires original positive ranks')
        from training.recall_prior import blend_recall_prior
        result = np.empty(len(frame), dtype=np.float64)
        for indices in frame.groupby(self.group_key, sort=False).indices.values():
            result[indices] = blend_recall_prior(frame['rank'].to_numpy()[indices], np.asarray(score)[indices],
                prior['alpha'], prior['k'])
        return result
