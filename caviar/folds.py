"""Assign whole training sequences to stratified folds and describe the mask.

Reads only target and mask columns. Run from the repository root:

    python -m caviar.folds

Safe to stop and restart: finished sequences are kept in
caviar_cache/folds_scan.pkl and are not read again.

Writes:
    reports/caviar_seq_stats.csv   one row per training sequence
    reports/caviar_folds.csv       fold for each sequence in each round (-1 = unused)
    reports/caviar_folds.json      balance checks and validation mask summary
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import pickle
import threading

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / 'datasets' / 'train.parquet'
VALID = ROOT / 'datasets' / 'valid.parquet'
REPORTS = ROOT / 'reports'
CACHE = ROOT / 'caviar_cache'
THREADS = 4
SAVE_EVERY = 200
_local = threading.local()
WARMUP = 99
CLIP = 2.0
SEED = 20261105
N_BINS = 5
MIN_STRATUM = 5
ROUNDS = {'r1': (1500, 3), 'r2': (5000, 5)}
EXTRA_SEQUENCES = 3000


def _open(path: Path) -> pq.ParquetFile:
    """Keep one open file per thread, since opening re-reads the large file footer."""
    files = _local.__dict__.setdefault('files', {})
    if path not in files:
        files[path] = pq.ParquetFile(path)
    return files[path]


def _train_stats(group: int) -> dict:
    table = _open(TRAIN).read_row_group(
        group, columns=['seq_ix', 't0', 't1'], use_threads=False)
    raw = np.column_stack([table['t0'].to_numpy(), table['t1'].to_numpy()])[WARMUP:]
    y = np.clip(raw, -CLIP, CLIP)
    weight = np.abs(y)
    return {
        'row_group': group,
        'seq_ix': int(table['seq_ix'][0].as_py()),
        'mean_t0': y[:, 0].mean(), 'mean_t1': y[:, 1].mean(),
        'abs_t0': weight[:, 0].mean(), 'abs_t1': weight[:, 1].mean(),
        'wy_t0': (weight[:, 0] * y[:, 0]).sum(), 'wy_t1': (weight[:, 1] * y[:, 1]).sum(),
        'w_t0': weight[:, 0].sum(), 'w_t1': weight[:, 1].sum(),
        'wyy_t0': (weight[:, 0] * y[:, 0] ** 2).sum(), 'wyy_t1': (weight[:, 1] * y[:, 1] ** 2).sum(),
        'change_t0': (raw[1:, 0] != raw[:-1, 0]).mean(),
        'change_t1': (raw[1:, 1] != raw[:-1, 1]).mean(),
        'unique_t0': np.unique(raw[:, 0]).size,
    }


def _mask_stats(group: int) -> dict:
    table = _open(VALID).read_row_group(
        group, columns=['is_scored', 't0', 't1'], use_threads=False)
    mask = table['is_scored'].to_numpy().astype(bool)
    y = np.clip(np.column_stack([table['t0'].to_numpy(), table['t1'].to_numpy()]), -CLIP, CLIP)
    other = ~mask
    other[:WARMUP] = False
    edges = np.diff(mask.astype(np.int8))
    any_scored = bool(mask.any())
    return {
        'scored': int(mask.sum()),
        'runs': int((edges == 1).sum() + mask[0]),
        'first': int(np.argmax(mask)) if any_scored else -1,
        'abs_scored': float(np.abs(y[mask]).mean()) if any_scored else np.nan,
        'abs_other': float(np.abs(y[other]).mean()),
    }


def between_sequence_share(stats: pd.DataFrame, target: str) -> float:
    """Share of the metric's weighted target variance that lies between sequences."""
    w, wy, wyy = stats[f'w_{target}'], stats[f'wy_{target}'], stats[f'wyy_{target}']
    mean = wy.sum() / w.sum()
    total = wyy.sum() / w.sum() - mean ** 2
    between = (w * (wy / w - mean) ** 2).sum() / w.sum()
    return float(between / total)


def make_strata(stats: pd.DataFrame) -> pd.Series:
    level = pd.qcut(stats['mean_t0'], N_BINS, labels=False, duplicates='drop')
    size = pd.qcut((stats['abs_t0'] + stats['abs_t1']) / 2, N_BINS, labels=False, duplicates='drop')
    strata = level.astype(str) + '_' + size.astype(str)
    counts = strata.map(strata.value_counts())
    # Merge sparse joint bins into their level bin, then into one shared bin.
    strata = strata.where(counts >= MIN_STRATUM, 'level' + level.astype(str))
    counts = strata.map(strata.value_counts())
    return strata.where(counts >= MIN_STRATUM, 'rare')


def assign_round(strata: pd.Series, n_sequences: int, n_folds: int, seed: int) -> np.ndarray:
    fold = np.full(len(strata), -1, dtype=int)
    picker = StratifiedShuffleSplit(n_splits=1, train_size=n_sequences, random_state=seed)
    chosen, _ = next(picker.split(np.zeros(len(strata)), strata))
    splitter = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    for index, (_, held) in enumerate(splitter.split(chosen, strata.iloc[chosen])):
        fold[chosen[held]] = index
    return fold


def assign_extra(strata: pd.Series, fold_r2: pd.Series, n_sequences: int, seed: int) -> np.ndarray:
    """Pick training-only sequences outside Round 2, so its held-out folds stay clean."""
    fold = np.full(len(strata), -1, dtype=int)
    unused = np.where(fold_r2.to_numpy() < 0)[0]
    picker = StratifiedShuffleSplit(n_splits=1, train_size=n_sequences, random_state=seed)
    chosen, _ = next(picker.split(np.zeros(len(unused)), strata.iloc[unused]))
    fold[unused[chosen]] = 0
    return fold


def _save_progress(progress: dict) -> None:
    path = CACHE / 'folds_scan.pkl'
    temporary = path.with_suffix('.tmp')
    temporary.write_bytes(pickle.dumps(progress))
    os.replace(temporary, path)


def _scan(name: str, source: Path, read, progress: dict) -> pd.DataFrame:
    """Read every sequence not yet in `progress`, saving as it goes."""
    done = progress[name]
    total = pq.ParquetFile(source).num_row_groups
    todo = [group for group in range(total) if group not in done]
    with ThreadPoolExecutor(THREADS) as pool:
        for start in range(0, len(todo), SAVE_EVERY):
            chunk = todo[start:start + SAVE_EVERY]
            done.update(zip(chunk, pool.map(read, chunk)))
            _save_progress(progress)
            print(f'{name}: {len(done)}/{total}', flush=True)
    return pd.DataFrame([done[group] for group in range(total)])


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    path = CACHE / 'folds_scan.pkl'
    # Safe to unpickle: this script writes the file locally, in a git-ignored folder.
    progress = pickle.loads(path.read_bytes()) if path.exists() else {'train': {}, 'valid': {}}
    stats = _scan('train', TRAIN, _train_stats, progress)
    mask = _scan('valid', VALID, _mask_stats, progress)
    stats.to_csv(REPORTS / 'caviar_seq_stats.csv', index=False)

    strata = make_strata(stats)
    folds = stats[['row_group', 'seq_ix']].assign(stratum=strata)
    balance = {}
    for name, (count, n_folds) in ROUNDS.items():
        folds[f'fold_{name}'] = assign_round(strata, count, n_folds, SEED)
        used = stats.assign(fold=folds[f'fold_{name}']).query('fold >= 0')
        assert used['seq_ix'].is_unique and len(used) == count
        balance[name] = {
            'sequences': count,
            'fold_sizes': used['fold'].value_counts().sort_index().tolist(),
            'fold_means': used.groupby('fold')[['mean_t0', 'mean_t1', 'abs_t0', 'abs_t1']]
                              .mean().round(4).to_dict('list'),
        }
    folds['fold_extra'] = assign_extra(strata, folds['fold_r2'], EXTRA_SEQUENCES, SEED)
    assert not ((folds['fold_extra'] >= 0) & (folds['fold_r2'] >= 0)).any()
    # Every other sequence, for a final model trained on all of the training data.
    folds['fold_rest'] = ((folds['fold_r2'] < 0) & (folds['fold_extra'] < 0)).map({True: 0, False: -1})
    folds.to_csv(REPORTS / 'caviar_folds.csv', index=False)

    summary = {
        'sequences': len(stats),
        'strata': int(strata.nunique()),
        'smallest_stratum': int(strata.value_counts().min()),
        'largest_stratum': int(strata.value_counts().max()),
        'between_sequence_share': {t: between_sequence_share(stats, t) for t in ('t0', 't1')},
        'target_change_rate': {t: float(stats[f'change_{t}'].mean()) for t in ('t0', 't1')},
        'median_unique_t0_per_sequence': float(stats['unique_t0'].median()),
        'whole_train_means': stats[['mean_t0', 'mean_t1', 'abs_t0', 'abs_t1']].mean().round(4).to_dict(),
        'rounds': balance,
        'valid_mask': {
            'sequences': len(mask),
            'scored_rows': int(mask['scored'].sum()),
            'scored_fraction': float(mask['scored'].sum() / (len(mask) * (20_000 - WARMUP))),
            'sequences_without_scored_rows': int((mask['scored'] == 0).sum()),
            'scored_per_sequence_p05_p50_p95': mask['scored'].quantile([.05, .5, .95]).tolist(),
            'runs_per_sequence_p05_p50_p95': mask['runs'].quantile([.05, .5, .95]).tolist(),
            'mean_abs_target_scored': float(mask['abs_scored'].mean()),
            'mean_abs_target_other': float(mask['abs_other'].mean()),
        },
    }
    (REPORTS / 'caviar_folds.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
