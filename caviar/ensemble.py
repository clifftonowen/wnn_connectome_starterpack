"""Measure what averaging trained models gains on validation. No training.

    python -m caviar.ensemble

Saves each model's validation predictions under caviar_runs/ensemble_preds/, about
300 MB each, and skips any already saved, so it is safe to stop and restart. Each
single model must reproduce the score stored by its run, or the script stops.
Writes reports/caviar_ensemble.csv. `gain` is the average's score minus the mean
score of its members. Delete the predictions folder afterwards to get the disk back.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from caviar import data as cache
from caviar.models import build
from caviar.train import EVAL_BATCH, GPU_FRACTION, RUNS, predict_rows
from utils import GlobalAccumulator, SEQUENCE_LENGTH, WARMUP

ROOT = Path(__file__).resolve().parents[1]
OUT = RUNS / 'ensemble_preds'
REPORT = ROOT / 'reports' / 'caviar_ensemble.csv'
TOLERANCE = 1e-5


def member(key: str, seed: int, fold: int, tag: str = 'r2') -> str:
    return f'{tag}/{key}_s{seed}_f{fold}'


GROUP = [member('group_gru', seed, fold) for seed in range(3) for fold in range(5)]
LONG = [member('group_gru', 0, fold, 'r2_long_extra') for fold in range(3)]
ENSEMBLES = {
    'pair, same data, different seed': [[member('group_gru', 0, f), member('group_gru', 1, f)] for f in range(5)],
    'pair, same seed, different folds': [[member('group_gru', 0, f), member('group_gru', 0, (f + 1) % 5)] for f in range(5)],
    'pair, group_gru + instr_gru': [[member('group_gru', 0, f), member('instr_gru', 0, f)] for f in range(5)],
    'pair, group_gru + gru2': [[member('group_gru', 0, f), member('gru2', 0, f)] for f in range(5)],
    'pair, 12,000 steps + extra sequences': [[LONG[0], LONG[1]], [LONG[0], LONG[2]], [LONG[1], LONG[2]]],
    '3, same data, different seeds': [[member('group_gru', s, f) for s in range(3)] for f in range(5)],
    '3, 12,000 steps + extra sequences': [LONG],
    '5, group_gru, one seed': [[member('group_gru', s, f) for f in range(5)] for s in range(3)],
    '15, group_gru': [GROUP],
    '15, instr_gru': [[member('instr_gru', s, f) for s in range(3) for f in range(5)]],
    '30, group_gru + instr_gru': [GROUP + [member('instr_gru', s, f) for s in range(3) for f in range(5)]],
}


def path_for(name: str) -> Path:
    return OUT / (name.replace('/', '__') + '.npy')


def save_predictions(name: str, x, device) -> None:
    path = path_for(name)
    if path.exists():
        return
    saved = torch.load(RUNS / f'{name}.pt', map_location='cpu', weights_only=True)
    model = build(saved['key'])
    model.load_state_dict(saved['state_dict'])
    model.to(device).eval()
    temporary = path.with_suffix('.tmp')
    out = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32, shape=(len(x), SEQUENCE_LENGTH, 2))
    for start in range(0, len(x), EVAL_BATCH):
        rows = np.arange(start, min(start + EVAL_BATCH, len(x)))
        out[rows] = predict_rows(model, x[rows], device)
    out.flush()
    del out
    temporary.replace(path)
    print('predicted', name, flush=True)


def score(names: list[str], y, masks) -> dict:
    total = np.zeros((len(y), SEQUENCE_LENGTH, 2), dtype=np.float32)
    for name in names:
        total += np.load(path_for(name), mmap_mode='r')
    total /= len(names)
    full, masked = GlobalAccumulator(), GlobalAccumulator()
    need = np.arange(SEQUENCE_LENGTH) >= WARMUP
    for seq in range(len(y)):
        targets = np.asarray(y[seq])
        full.add(targets, total[seq], need)
        masked.add(targets, total[seq], masks[seq] & need)
    return {'masked': masked.result()['weighted_pearson'], 'all_rows': full.result()['weighted_pearson']}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(GPU_FRACTION)
    valid = cache.load('valid')
    x, y, masks = valid['x'], valid['y'], valid['mask']
    needed = sorted({name for groups in ENSEMBLES.values() for group in groups for name in group})
    for name in needed:
        save_predictions(name, x, device)

    singles = {}
    for name in needed:
        singles[name] = score([name], y, masks)
        stored = json.loads((RUNS / f'{name}.json').read_text(encoding='utf-8'))['valid_masked_wp']
        gap = abs(singles[name]['masked'] - stored)
        if gap > TOLERANCE:
            raise SystemExit(f'{name} scored {singles[name]["masked"]}, but its run stored {stored}')
    print(f'all {len(needed)} single models match their stored scores', flush=True)

    rows = []
    for label, groups in ENSEMBLES.items():
        for group in groups:
            result = score(group, y, masks)
            single_masked = np.mean([singles[name]['masked'] for name in group])
            single_all = np.mean([singles[name]['all_rows'] for name in group])
            rows.append({'ensemble': label, 'members': ' '.join(group), 'size': len(group),
                         'masked': result['masked'], 'members_masked': single_masked,
                         'gain_masked': result['masked'] - single_masked,
                         'all_rows': result['all_rows'], 'members_all_rows': single_all,
                         'gain_all_rows': result['all_rows'] - single_all})
            print(label, {k: round(v, 5) for k, v in rows[-1].items() if isinstance(v, float)}, flush=True)
            pd.DataFrame(rows).to_csv(REPORT, index=False)
    table = pd.DataFrame(rows).groupby('ensemble', sort=False)[
        ['masked', 'gain_masked', 'all_rows', 'gain_all_rows']].mean().round(5)
    table['min_gain_masked'] = pd.DataFrame(rows).groupby('ensemble', sort=False)['gain_masked'].min().round(5)
    print(table.to_string())


if __name__ == '__main__':
    main()
