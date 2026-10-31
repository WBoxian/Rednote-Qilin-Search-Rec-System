"""Leakage-aware feature construction and forward-only stacking primitives."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

from backend.online.item_features import STATIC_ITEM_FEATURES
from training.history import recent_history_ids, history_ids_oldest_first

SEMANTIC_FEATURES = ('history_mean_cosine', 'history_recent_cosine')


def forward_blocks(frame, group_key, n_blocks=4):
    """Warm up on block zero, predict later blocks using strictly earlier requests."""
    if n_blocks < 2:
        raise ValueError('At least two forward blocks required')
    timestamps = pd.to_numeric(frame.event_timestamp, errors='raise')
    if not np.isfinite(timestamps).all() or frame[group_key].isna().any():
        raise ValueError('Finite timestamps and complete request IDs required')
    intervals = frame.assign(_time=timestamps).groupby(group_key)._time.agg(['min', 'max'])
    maxima = np.sort(intervals['max'].to_numpy())
    for block in range(1, n_blocks):
        boundary = float(np.nextafter(maxima[max(0, len(maxima)*block//n_blocks-1)], np.inf))
        upper = float(np.nextafter(maxima[max(0, len(maxima)*(block+1)//n_blocks-1)], np.inf)) if block+1<n_blocks else np.inf
        train_ids = intervals.index[intervals['max'] < boundary]
        # Requests spanning either boundary are excluded, not used to move the cutoff backwards.
        prediction_ids = intervals.index[(intervals['min'] >= boundary) & (intervals['max'] < upper)]
        fit = np.flatnonzero(frame[group_key].isin(train_ids))
        pred = np.flatnonzero(frame[group_key].isin(prediction_ids))
        if len(fit) and len(pred):
            assert frame.iloc[fit].event_timestamp.max() < frame.iloc[pred].event_timestamp.min()
            yield fit, pred


def group_minmax(scores, groups):
    frame = pd.DataFrame({'score':np.asarray(scores, dtype=float), 'group':np.asarray(groups)})
    low = frame.groupby('group').score.transform('min').to_numpy()
    high = frame.groupby('group').score.transform('max').to_numpy()
    return ((frame.score.to_numpy()-low)/np.maximum(high-low, 1e-8)).astype(np.float32)


class MappedVectors:
    """Read immutable embedding snapshots without request-ID zero fallbacks."""
    def __init__(self, folder: Path, name: str, dimension=768):
        self.files = [folder/f'{name}_map.json', folder/f'{name}.bin']
        mapping = json.loads(self.files[0].read_text())
        ids = np.fromiter((int(k) for k in mapping), dtype=np.int64)
        positions = np.fromiter(mapping.values(), dtype=np.int64)
        order = np.argsort(ids)
        self.ids, self.positions = ids[order], positions[order]
        self.dimension = dimension
        expected = len(ids)*dimension*np.dtype(np.float16).itemsize
        if self.files[1].stat().st_size != expected:
            raise ValueError(f'Embedding shape mismatch: {name}')
        self.values = np.memmap(self.files[1], dtype=np.float16, mode='r', shape=(len(ids),dimension))

    def get(self, ids, required=False):
        ids = np.asarray(ids, dtype=np.int64)
        pos = np.searchsorted(self.ids, ids)
        found = pos < len(self.ids)
        found[found] &= self.ids[pos[found]] == ids[found]
        if required and not found.all():
            raise ValueError(f'{int((~found).sum())} missing required embedding IDs')
        result = np.zeros((len(ids),self.dimension), dtype=np.float32)
        result[found] = self.values[self.positions[pos[found]]]
        return result


def normalize(vectors):
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-8)


class RequestContext:
    def __init__(self, requests, key, text, image, query_vectors=None, seq_len=20):
        if requests[key].duplicated().any():
            raise ValueError('Request context IDs must be unique')
        self.key, self.text, self.image, self.seq_len = key,text,image,seq_len
        self.requests = requests.set_index(key)
        self.positions = {int(gid):i for i,gid in enumerate(self.requests.index)}
        self.histories = []
        for value in self.requests.recent_clicked_note_idxs:
            history = recent_history_ids(value, seq_len)
            self.histories.append(history)
        self.query_vectors = query_vectors
        self.mean = np.zeros((len(requests),768), dtype=np.float32)
        self.recent = np.zeros_like(self.mean)
        for i,ids in enumerate(self.histories):
            if ids:
                values = text.get(ids, required=True)
                self.mean[i] = normalize(values.mean(axis=0, keepdims=True))[0]
                self.recent[i] = normalize(values[:1])[0]
        if query_vectors is not None and query_vectors.shape != (len(requests),768):
            raise ValueError('Query vectors must align one-to-one with raw requests')

    def feature_frame(self, frame, static_store):
        out = static_store.attach(frame)
        if not out.item_features_found.all():
            raise ValueError('Ranking item missing from provided catalog')
        pos = np.array([self.positions[int(v)] for v in out[self.key]], dtype=np.int64)
        for name in SEMANTIC_FEATURES:
            out[name] = np.zeros(len(out), dtype=np.float32)
        if self.query_vectors is not None:
            out['query_item_cosine'] = np.zeros(len(out), dtype=np.float32)
        for start in range(0,len(out),8192):
            stop = min(len(out),start+8192)
            note = normalize(self.text.get(out.note_idx.iloc[start:stop],required=True))
            p = pos[start:stop]
            out.loc[out.index[start:stop],'history_mean_cosine'] = np.einsum('ij,ij->i',note,self.mean[p])
            out.loc[out.index[start:stop],'history_recent_cosine'] = np.einsum('ij,ij->i',note,self.recent[p])
            if self.query_vectors is not None:
                out.loc[out.index[start:stop],'query_item_cosine'] = np.einsum('ij,ij->i',note,normalize(self.query_vectors[p]))
        return out

    def batch(self, frame, numeric, device):
        import torch
        features = {name:torch.as_tensor(frame[name].to_numpy(np.float32)[:,None],device=device) for name in numeric}
        features['note_text_emb'] = torch.as_tensor(self.text.get(frame.note_idx,required=True),device=device)
        features['note_img_emb'] = torch.as_tensor(self.image.get(frame.note_idx),device=device)
        ids = [self.positions[int(v)] for v in frame[self.key]]
        # The raw snapshot lists most-recent first; DIEN consumes oldest first.
        sequences = np.zeros((len(frame),self.seq_len,768),dtype=np.float32)
        histories = [history_ids_oldest_first(self.histories[pos], self.seq_len) for pos in ids]
        lengths = np.array([len(history) for history in histories], dtype=np.int64)
        if lengths.sum():
            flat = np.concatenate([np.asarray(history, dtype=np.int64) for history in histories])
            rows = np.repeat(np.arange(len(frame)), lengths)
            columns = np.concatenate([np.arange(length) for length in lengths])
            sequences[rows, columns] = self.text.get(flat, required=True)
        features['seq_embs'] = torch.as_tensor(sequences,device=device)
        if self.query_vectors is not None:
            features['query_emb'] = torch.as_tensor(self.query_vectors[ids],device=device)
        return features