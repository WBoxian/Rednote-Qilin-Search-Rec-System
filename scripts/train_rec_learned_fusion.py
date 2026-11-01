"""Train the frozen 23-feature Recommendation fusion from SHA-bound prepared fit data.

This TRAIN entrypoint neither prepares candidates nor evaluates quality. Historical
candidate PREPARE still depends on the archived script bound by the manifest.
No validation/test inputs, resampling, hyperparameter flags, or activation are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from recall.learned_context_fusion import FEATURE_NAMES

PARAMETERS = dict(objective='lambdarank', num_leaves=31, n_estimators=200,
    learning_rate=.05, random_state=42, n_jobs=3, lambdarank_truncation_level=100,
    label_gain=[0, 1], verbosity=-1, deterministic=True, force_col_wise=True)
ASSET_ROLES = frozenset({'training_npz', 'group_manifest', 'historical_plan',
    'past_split', 'training_summary', 'prefix_requests', 'suffix_requests', 'preparation_script'})
LABEL_SEMANTICS = 'known_click_vs_unjudged_implicit_retrieval'
GROUP_SELECTION = 'union_has_known_positive'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object: ' + str(path))
    return value


def _no_feedback(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {'official_test_read', 'validation_labels_gradient', 'test_read'} and item is not False:
                raise ValueError('Training manifest must exclude validation/test feedback')
            _no_feedback(item)
    elif isinstance(value, list):
        for item in value:
            _no_feedback(item)


def _validate_manifest(manifest):
    if (manifest.get('version') != 1 or type(manifest.get('version')) is not int
            or manifest.get('kind') != 'rec_learned_fusion_train_v1'
            or manifest.get('training_scope') != 'original_fit_prefix_suffix_only'
            or manifest.get('feature_names') != list(FEATURE_NAMES)
            or manifest.get('label_semantics') != LABEL_SEMANTICS
            or manifest.get('positive_group_selection') != GROUP_SELECTION
            or manifest.get('resample_prepared_rows') is not False
            or json.dumps(manifest.get('parameters'), sort_keys=True) != json.dumps(PARAMETERS, sort_keys=True)
            or manifest.get('preparation') != {'status': 'historical_prepared_candidates',
                'entrypoint': 'preparation_script', 'clean_checkout_end_to_end_prepare': False}):
        raise ValueError('Unsupported frozen fusion training/schema/label/preparation contract')
    if set(manifest.get('assets', {})) != ASSET_ROLES:
        raise ValueError('Training manifest requires exactly the declared fit-only asset roles')
    _no_feedback(manifest)


def _assets(manifest, manifest_path):
    bound = {}
    for role, entry in manifest['assets'].items():
        if not isinstance(entry, dict) or set(entry) != {'path', 'sha256'}:
            raise ValueError('Each training asset requires a path and SHA256')
        text, expected = entry['path'], entry['sha256']
        if (not isinstance(text, str) or not text or not isinstance(expected, str)
                or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected)):
            raise ValueError('Invalid training asset path/SHA256')
        path = Path(text)
        path = (path if path.is_absolute() else manifest_path.parent / path).resolve()
        if not path.is_file() or sha256(path) != expected:
            raise ValueError('Training asset SHA256 mismatch: ' + role)
        bound[role] = {'path': str(path), 'sha256': expected}
    return bound


def _integer_columns(frame, columns):
    for key in columns:
        if key not in frame or frame[key].dtype.kind not in 'iu' or (frame[key] < 0).any():
            raise ValueError('Expected nonnegative integer manifest column: ' + key)


def _training_data(assets):
    path = lambda role: Path(assets[role]['path'])
    historical, split, summary = (_json(path(role)) for role in ('historical_plan', 'past_split', 'training_summary'))
    if (historical.get('features') != list(FEATURE_NAMES) or json.dumps(historical.get('model'), sort_keys=True) != json.dumps(PARAMETERS, sort_keys=True)
            or historical.get('label_semantics') != 'known train clicks versus unjudged implicit retrieval contrast items; zeros do not mean observed CTR negatives'
            or historical.get('code_sha256') != assets['preparation_script']['sha256']
            or 'zero-positive groups excluded gradient only' not in historical.get('training_sampling', '')):
        raise ValueError('Historical plan does not attest the actual schema/labels/positive group preparation')
    # The archived plan describes its old evaluation too. Never resolve/read its validation input paths.
    prefix = pd.read_parquet(path('prefix_requests'), columns=['gid', 'user_idx', 'lo', 'hi'])
    suffix = pd.read_parquet(path('suffix_requests'), columns=['gid', 'user_idx', 'lo', 'hi'])
    boundary = historical.get('P_T_boundary')
    if isinstance(boundary, bool) or not isinstance(boundary, (int, float)) or not np.isfinite(boundary):
        raise ValueError('Historical plan requires a finite whole-request P/T boundary')
    for frame in (prefix, suffix):
        _integer_columns(frame, ['gid', 'user_idx'])
        if (not len(frame) or frame.gid.duplicated().any()
                or not np.isfinite(frame[['lo', 'hi']].to_numpy()).all() or (frame.lo > frame.hi).any()):
            raise ValueError('Invalid complete-request prefix/suffix manifest')
    if (set(prefix.gid) & set(suffix.gid) or prefix.hi.max() >= boundary or suffix.lo.min() < boundary
            or split.get('prefix_requests') != len(prefix) or split.get('suffix_requests') != len(suffix)
            or split.get('prefix_max_hi') != float(prefix.hi.max())
            or split.get('suffix_min_lo') != float(suffix.lo.min())
            or split.get('suffix_max_hi') != float(suffix.hi.max())):
        raise ValueError('Prefix/suffix request split differs from the declared past-only contract')
    groups = pd.read_parquet(path('group_manifest'))
    required = ['gid', 'user_idx', 'union_candidates', 'union_positive_pairs', 'sample_candidates']
    if list(groups.columns) != required:
        raise ValueError('Prepared group manifest schema/order mismatch')
    _integer_columns(groups, required)
    if not len(groups) or groups.gid.duplicated().any() or (groups.gid.diff().dropna() <= 0).any():
        raise ValueError('Prepared groups must be unique in their original ascending request order')
    matched = groups.merge(suffix[['gid', 'user_idx']], on='gid', how='left', validate='one_to_one', suffixes=('', '_suffix'))
    if not np.array_equal(matched.user_idx, matched.user_idx_suffix):
        raise ValueError('Prepared training group is not the declared suffix request/user')
    if ((groups.sample_candidates <= 0).any() or (groups.union_positive_pairs <= 0).any()
            or (groups.union_positive_pairs > groups.sample_candidates).any()
            or (groups.sample_candidates > groups.union_candidates).any() or (groups.union_candidates > 3000).any()):
        raise ValueError('Prepared groups violate positive-only selection or candidate budgets')
    with np.load(path('training_npz'), allow_pickle=False) as archive:
        if set(archive.files) != {'X', 'y', 'group'}:
            raise ValueError('Training NPZ requires exactly X/y/group')
        X, y, group = (archive[name] for name in ('X', 'y', 'group'))
    if (X.dtype != np.float32 or X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES)
            or not len(X) or not np.isfinite(X).all() or y.dtype != np.int32 or y.shape != (len(X),)
            or not np.isin(y, [0, 1]).all() or group.dtype != np.int32 or group.shape != (len(groups),)
            or (group <= 0).any() or int(group.astype(np.int64).sum()) != len(X)
            or not np.array_equal(group, groups.sample_candidates.to_numpy())):
        raise ValueError('Prepared NPZ shape/dtype/finiteness/binary label/group contract mismatch')
    offsets = np.r_[0, np.cumsum(group, dtype=np.int64)]
    positives = np.add.reduceat(y.astype(np.int64), offsets[:-1])
    if not np.array_equal(positives, groups.union_positive_pairs.to_numpy()):
        raise ValueError('Group positives differ from the declared unresampled prepared groups')
    if (type(split.get('suffix_positive_pairs')) is not int or split['suffix_positive_pairs']<=0
            or type(summary.get('total_positive_pairs')) is not int or summary['total_positive_pairs']<int(y.sum())
            or summary.get('requests') != len(suffix)
            or summary.get('zero_positive_union_requests') != len(suffix) - len(groups)
            or summary.get('sampled_candidates') != len(X)
            or summary.get('union_positive_pairs') != int(y.sum())
            or summary.get('total_positive_pairs') != split.get('suffix_positive_pairs')
            or type(summary.get('union_candidates')) is not int
            or summary['union_candidates'] < int(groups.union_candidates.sum())):
        raise ValueError('Prepared positive-group selection/counts differ from the past summary')
    return X, y, group, {'prefix_requests': len(prefix), 'suffix_requests': len(suffix),
        'selected_positive_groups': len(groups), 'excluded_zero_positive_groups': len(suffix) - len(groups),
        'rows': len(X), 'positive_rows': int(y.sum()), 'unjudged_contrast_rows': int((y == 0).sum()),
        'P_T_boundary': boundary}


def _array_fingerprint(X, y, group):
    digest = hashlib.sha256()
    for name, value in [('X', X), ('y', y), ('group', group)]:
        digest.update(json.dumps([name, value.dtype.str, list(value.shape)]).encode())
        digest.update(value.tobytes(order='C'))
    return digest.hexdigest()


def train_rec_learned_fusion(training_manifest, destination):
    """Fit exactly the declared prepared rows once; no validation or selection."""
    manifest_path, destination = Path(training_manifest).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    manifest_hash = sha256(manifest_path)
    manifest = _json(manifest_path)
    _validate_manifest(manifest)
    assets = _assets(manifest, manifest_path)
    X, y, group, stats = _training_data(assets)
    fingerprint = _array_fingerprint(X, y, group)
    started = time.perf_counter()
    model = lgb.LGBMRanker(**PARAMETERS)
    model.fit(X, y, group=group, feature_name=list(FEATURE_NAMES))
    elapsed = time.perf_counter() - started
    if model.booster_.feature_name() != list(FEATURE_NAMES) or model.booster_.num_trees() <= 0:
        raise ValueError('Trained model does not satisfy the actual production feature schema')
    if sha256(manifest_path) != manifest_hash or any(sha256(entry['path']) != entry['sha256'] for entry in assets.values()):
        raise ValueError('Bound training input changed during fit')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + destination.name + '-', dir=destination.parent) as tmp:
        stage = Path(tmp) / 'release'
        stage.mkdir()
        model.booster_.save_model(str(stage / 'model.txt'))
        lineage = dict(version=1, status='trained_not_evaluated_or_deployed',
            training_manifest={'path': str(manifest_path), 'sha256': manifest_hash}, sources=assets,
            feature_names=list(FEATURE_NAMES), parameters=PARAMETERS, data=stats,
            label_semantics=LABEL_SEMANTICS, positive_group_selection=GROUP_SELECTION,
            prepared_array_fingerprint=fingerprint, original_row_order_preserved=True,
            resampled_rows=False, actual_trees=model.booster_.num_trees(),
            actual_boosting_iterations=model.booster_.current_iteration(), fit_seconds=elapsed,
            model_sha256=sha256(stage / 'model.txt'), producer_script_sha256=sha256(Path(__file__)),
            versions={'lightgbm': lgb.__version__, 'numpy': np.__version__},
            preparation=manifest['preparation'], official_test_read=False,
            validation_feedback_read=False, quality_evaluation_performed=False, activated=False)
        (stage / 'training_lineage.json').write_text(json.dumps(lineage, indent=2, allow_nan=False) + '\n')
        if destination.exists():
            raise FileExistsError(destination)
        stage.rename(destination)
    return destination / 'training_lineage.json'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    path = train_rec_learned_fusion(args.training_manifest, args.output_dir)
    print(json.dumps({'training_lineage': str(path), 'activated': False,
                      'evaluated': False, 'preparation_included': False}))


if __name__ == '__main__':
    main()
