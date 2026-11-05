"""Model 31: moving averages at several speeds, then LightGBM.

    python -m caviar.gbm --round r1 --fold 0

Each target gets its own booster. Rows are sampled from the training folds, since
neighbouring rows are nearly identical. Writes the same JSON as train.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np

from caviar import data as cache
from utils import GlobalAccumulator, SEQUENCE_LENGTH, WARMUP

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'caviar_runs'
RATES = (1 / 8, 1 / 64, 1 / 512)
STRIDE = 25
MAX_TRAIN_SEQUENCES = 400
NEED = np.arange(SEQUENCE_LENGTH) >= WARMUP
PARAMS = dict(objective='regression', learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=0,
              num_threads=8)
ROUNDS, PATIENCE = 500, 25


def features(x: np.ndarray) -> np.ndarray:
    """The current row, plus its gap from a moving average at each speed."""
    x = np.asarray(x, dtype=np.float32)
    out = np.empty((x.shape[0], x.shape[1] * (1 + len(RATES))), dtype=np.float32)
    out[:, :x.shape[1]] = x
    for r, rate in enumerate(RATES, 1):
        average = x[0].copy()
        for t in range(x.shape[0]):
            average += rate * (x[t] - average)
            out[t, r * x.shape[1]:(r + 1) * x.shape[1]] = x[t] - average
    return out


def _rows(x, y, indices):
    part_x, part_y = [], []
    for seq in indices:
        part_x.append(features(x[seq])[WARMUP::STRIDE])
        part_y.append(np.clip(y[seq, WARMUP::STRIDE], -2, 2))
    return np.concatenate(part_x), np.concatenate(part_y)


def _score(boosters, x, y, indices, masks=None):
    full, masked = GlobalAccumulator(), GlobalAccumulator()
    for seq in indices:
        f = features(x[seq])
        pred = np.column_stack([b.predict(f) for b in boosters]).astype(np.float32)
        target = np.asarray(y[seq])
        full.add(target, pred, NEED)
        if masks is not None:
            masked.add(target, pred, masks[seq] & NEED)
    return full.result(), masked.result() if masks is not None else None


def run(round_name, fold, seed=0, force=False, tag=None):
    out_dir = RUNS / (tag or round_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f'ema_gbm_s{seed}_f{fold}.json'
    if out_json.exists() and not force:
        return
    data, valid = cache.load(round_name), cache.load('valid')
    x, y, folds = data['x'], data['y'], data['fold']
    rng = np.random.default_rng(seed + 1000 * fold)
    held = np.where(folds == fold)[0]
    rest = rng.permutation(np.where(folds != fold)[0])
    n_dev = max(8, int(len(rest) * 0.05))
    dev, train = np.sort(rest[:n_dev]), np.sort(rest[n_dev:n_dev + MAX_TRAIN_SEQUENCES])

    begin = time.perf_counter()
    train_x, train_y = _rows(x, y, train)
    dev_x, dev_y = _rows(x, y, dev)
    boosters = []
    for k in range(2):
        train_set = lgb.Dataset(train_x, train_y[:, k], weight=np.abs(train_y[:, k]))
        dev_set = lgb.Dataset(dev_x, dev_y[:, k], weight=np.abs(dev_y[:, k]), reference=train_set)
        boosters.append(lgb.train({**PARAMS, 'seed': seed + fold}, train_set, ROUNDS, [dev_set],
                                  callbacks=[lgb.early_stopping(PATIENCE, verbose=False)]))
    train_seconds = time.perf_counter() - begin

    begin = time.perf_counter()
    cv, _ = _score(boosters, x, y, held)
    v_all, v_mask = _score(boosters, valid['x'], valid['y'], np.arange(len(valid['x'])), valid['mask'])
    result = {
        'id': 31, 'key': 'ema_gbm', 'name': 'Moving averages then LightGBM', 'source': 'suggested',
        'round': round_name, 'fold': fold, 'seed': seed, 'config': {**PARAMS, 'rates': RATES, 'stride': STRIDE},
        'params': sum(b.num_trees() for b in boosters),
        'cv_wp': cv['weighted_pearson'], 'cv_t0': cv['t0'], 'cv_t1': cv['t1'],
        'valid_masked_wp': v_mask['weighted_pearson'], 'valid_masked_t0': v_mask['t0'],
        'valid_masked_t1': v_mask['t1'], 'valid_masked_rows': v_mask['selected_rows'],
        'valid_all_wp': v_all['weighted_pearson'], 'valid_all_t0': v_all['t0'], 'valid_all_t1': v_all['t1'],
        'eval_seconds': time.perf_counter() - begin, 'peak_gpu_mb': None,
        'best_step': boosters[0].best_iteration, 'stopped_step': boosters[1].best_iteration,
        'dev_wp': None, 'history': [], 'train_seconds': train_seconds, 'train_rows_per_second': None,
    }
    for k, booster in enumerate(boosters):
        booster.save_model(str(out_json.with_name(f'{out_json.stem}_t{k}.txt')))
    out_json.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f"[ema_gbm f{fold}] done cv {result['cv_wp']:.4f} valid_masked {result['valid_masked_wp']:.4f}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', default='r1', choices=['r1', 'r2'])
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    run(args.round, args.fold, args.seed, args.force, args.tag)
