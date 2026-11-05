"""Stop each step on purpose, restart it, and compare with an uninterrupted run.

    python -m caviar.check_resume <empty folder>

Uses a small made-up dataset in that folder and runs on CPU, so training results
can be compared exactly. It never reads the real data.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import threading

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from caviar import data, folds, train
from utils import FEATURE_COLUMNS, N_FEATURES, SEQUENCE_LENGTH, WARMUP

TRAIN_SEQS, VALID_SEQS = 30, 4


class Stop(Exception):
    pass


def stop_after(function, calls: int):
    """Wrap a function so that it raises after a number of calls, and count calls."""
    lock = threading.Lock()

    def wrapped(*args, **kwargs):
        with lock:
            wrapped.calls += 1
            if wrapped.calls > calls:
                raise Stop
        return function(*args, **kwargs)

    wrapped.calls = 0
    return wrapped


def write_parquet(path: Path, sequences: int, scored: bool, seed: int) -> None:
    rng = np.random.default_rng(seed)
    writer = None
    for seq in range(sequences):
        x = rng.standard_normal((SEQUENCE_LENGTH, N_FEATURES), dtype=np.float32)
        t0 = np.convolve(x[:, 0], np.ones(50) / 7, mode='full')[:SEQUENCE_LENGTH] + 0.3 * rng.standard_normal()
        t1 = -0.7 * t0 + 0.2 * x[:, 1]
        columns = {'seq_ix': np.full(SEQUENCE_LENGTH, seq + 1000, np.int32),
                   'step_in_seq': np.arange(SEQUENCE_LENGTH, dtype=np.int32),
                   'need_prediction': np.arange(SEQUENCE_LENGTH) >= WARMUP}
        if scored:
            columns['is_scored'] = (np.arange(SEQUENCE_LENGTH) >= WARMUP) & (rng.random(SEQUENCE_LENGTH) < 0.13)
        columns.update({name: x[:, j] for j, name in enumerate(FEATURE_COLUMNS)})
        columns.update(t0=t0.astype(np.float32), t1=t1.astype(np.float32))
        table = pa.table(columns)
        if writer is None:
            writer = pq.ParquetWriter(path, table.schema, compression='snappy')
        writer.write_table(table)
    writer.close()


def check_folds(root: Path) -> None:
    folds.TRAIN, folds.VALID = root / 'datasets/train.parquet', root / 'datasets/valid.parquet'
    folds.REPORTS, folds.CACHE = root / 'reports', root / 'caviar_cache'
    folds.ROUNDS = {'r1': (24, 3), 'r2': (25, 5)}
    folds.THREADS, folds.SAVE_EVERY = 2, 10
    original = folds._train_stats

    folds._train_stats = stop_after(original, 15)
    try:
        folds.main()
        raise AssertionError('the stop did not happen')
    except Stop:
        pass
    saved = len(pd.read_pickle(folds.CACHE / 'folds_scan.pkl')['train'])
    assert 10 <= saved < TRAIN_SEQS, saved

    folds._train_stats = stop_after(original, 10 ** 9)
    folds.main()
    assert folds._train_stats.calls == TRAIN_SEQS - saved, folds._train_stats.calls
    resumed = pd.read_csv(folds.REPORTS / 'caviar_folds.csv')

    (folds.CACHE / 'folds_scan.pkl').unlink()
    folds._train_stats = original
    folds.main()
    pd.testing.assert_frame_equal(resumed, pd.read_csv(folds.REPORTS / 'caviar_folds.csv'))
    print(f'folds: stopped with {saved}/{TRAIN_SEQS} saved, read only the rest, same result')


def check_data(root: Path) -> None:
    data.ROOT, data.CACHE = root, root / 'caviar_cache'
    data.FOLDS = root / 'reports' / 'caviar_folds.csv'
    data.SAVE_EVERY = 5
    original = data._read_sequence

    for split in ('r1', 'valid'):
        data._read_sequence = stop_after(original, 3 if split == 'valid' else 12)
        try:
            data.build(split)
            raise AssertionError('the stop did not happen')
        except Stop:
            pass
        try:
            data.load(split)
            raise AssertionError('an unfinished cache was accepted')
        except RuntimeError:
            pass
        data._read_sequence = stop_after(original, 10 ** 9)
        data.build(split)
        resumed_calls = data._read_sequence.calls
        loaded = data.load(split)
        resumed = {name: np.array(values) for name, values in loaded.items()}
        del loaded

        data._read_sequence = original
        data._progress_path(split).unlink()
        data.build(split)
        fresh = data.load(split)
        for name in resumed:
            assert np.array_equal(resumed[name], np.asarray(fresh[name])), (split, name)
        del fresh
        print(f'data {split}: restart read {resumed_calls} sequences, arrays identical')


def check_train(root: Path) -> None:
    train.RUNS = root / 'caviar_runs'
    torch.set_num_threads(2)
    config = {**train.DEFAULTS, 'batch': 4, 'window': 200, 'max_steps': 20, 'eval_every': 5,
              'warmup_steps': 2, 'patience': 100, 'dev_rows': 500}
    quiet = lambda message: None
    lines = []

    whole = train.run('gru2', 'r1', 0, 0, config, tag='whole', log=quiet)

    original = train.Sampler

    class StoppingSampler(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.calls = 0

        def __call__(self):
            self.calls += 1
            if self.calls > 12:
                raise Stop
            return super().__call__()

    train.Sampler = StoppingSampler
    try:
        train.run('gru2', 'r1', 0, 0, config, tag='stopped', log=quiet)
        raise AssertionError('the stop did not happen')
    except Stop:
        pass
    train.Sampler = original
    checkpoint = train.RUNS / 'stopped' / 'gru2_s0_f0.ckpt.pkl'
    assert checkpoint.exists()

    resumed = train.run('gru2', 'r1', 0, 0, config, tag='stopped', log=lines.append)
    assert 'resumed from step 10' in lines, lines
    assert not checkpoint.exists(), 'checkpoint should be deleted after finishing'
    for name in ('cv_wp', 'valid_masked_wp', 'valid_all_wp', 'best_step', 'stopped_step'):
        assert whole[name] == resumed[name], (name, whole[name], resumed[name])
    assert [h['dev_wp'] for h in whole['history']] == [h['dev_wp'] for h in resumed['history']]
    a = torch.load(train.RUNS / 'whole' / 'gru2_s0_f0.pt', weights_only=False)['state_dict']
    b = torch.load(train.RUNS / 'stopped' / 'gru2_s0_f0.pt', weights_only=False)['state_dict']
    assert all(torch.equal(a[k], b[k]) for k in a), 'weights differ'
    print(f"train: stopped at step 12, resumed from 10, identical weights and scores "
          f"(cv {whole['cv_wp']:.4f})")

    lines.clear()
    train.run('gru2', 'r1', 0, 0, config, tag='stopped', log=lines.append)
    assert 'already finished' in lines[0], lines
    print('train: a finished run is skipped')


def main() -> None:
    root = Path(sys.argv[1])
    (root / 'datasets').mkdir(parents=True)
    write_parquet(root / 'datasets/train.parquet', TRAIN_SEQS, False, 1)
    write_parquet(root / 'datasets/valid.parquet', VALID_SEQS, True, 2)
    check_folds(root)
    check_data(root)
    check_train(root)
    print('all resume checks passed')


if __name__ == '__main__':
    main()
