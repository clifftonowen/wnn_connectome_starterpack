"""Convert chosen sequences to uncompressed on-disk arrays for fast random access.

    python -m caviar.data --split r1      # training sequences of a round
    python -m caviar.data --split valid   # the full validation file
    python -m caviar.data --split extra   # extra training sequences outside Round 2
    python -m caviar.data --split rest    # all training sequences not cached above

Each cached sequence uses about 9 MB. Several training runs share one cache, and
the operating system keeps the parts in use in memory.

Safe to stop and restart: caviar_cache/<split>_progress.pkl records which
sequences are written, and only those are skipped. load() refuses an unfinished
cache.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import os
from pathlib import Path
import pickle
import threading

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from utils import FEATURE_COLUMNS, N_FEATURES, SEQUENCE_LENGTH, TARGET_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'caviar_cache'
FOLDS = ROOT / 'reports' / 'caviar_folds.csv'
THREADS = 4
SAVE_EVERY = 100
WORKING_SET_LIMIT_GB = 4


def _paths(split: str) -> dict[str, Path]:
    return {name: CACHE / f'{split}_{name}.npy' for name in ('x', 'y', 'mask', 'fold')}


# Progress files are safe to unpickle: this script writes them locally, in a
# git-ignored folder. Do not point CACHE at files from anywhere else.
def _progress_path(split: str) -> Path:
    return CACHE / f'{split}_progress.pkl'


def _save_progress(split: str, progress: dict) -> None:
    path = _progress_path(split)
    temporary = path.with_suffix('.tmp')
    temporary.write_bytes(pickle.dumps(progress))
    os.replace(temporary, path)


def _read_sequence(file: pq.ParquetFile, group: int):
    table = file.read_row_group(group, use_threads=False)
    x = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
    y = np.column_stack([table[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32)
    mask = table['is_scored'].to_numpy().astype(bool) if 'is_scored' in table.column_names else None
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f'nonfinite values in row group {group}')
    return x, y, mask


def build(split: str) -> None:
    limit_memory()
    CACHE.mkdir(exist_ok=True)
    paths = _paths(split)
    if split == 'valid':
        source = ROOT / 'datasets' / 'valid.parquet'
        groups = np.arange(pq.ParquetFile(source).num_row_groups)
        folds = None
    else:
        source = ROOT / 'datasets' / 'train.parquet'
        table = pd.read_csv(FOLDS)
        table = table[table[f'fold_{split}'] >= 0]
        groups = table['row_group'].to_numpy()
        folds = table[f'fold_{split}'].to_numpy()

    count = len(groups)
    group_list = groups.tolist()
    progress = None
    if _progress_path(split).exists():
        progress = pickle.loads(_progress_path(split).read_bytes())
        if progress['groups'] != group_list:
            progress = None  # The folds changed, so the partial cache no longer applies.
    if progress is not None and progress['complete']:
        print(f'{split}: already complete')
        return

    has_mask = split == 'valid'
    if progress is None:
        progress = {'groups': group_list, 'done': set(), 'complete': False}
        mode = 'w+'
    else:
        mode = 'r+'
    x = np.lib.format.open_memmap(paths['x'], mode, np.float32, (count, SEQUENCE_LENGTH, N_FEATURES))
    y = np.lib.format.open_memmap(paths['y'], mode, np.float32, (count, SEQUENCE_LENGTH, 2))
    mask = np.lib.format.open_memmap(paths['mask'], mode, bool, (count, SEQUENCE_LENGTH)) if has_mask else None
    local = threading.local()

    def write_one(index: int) -> None:
        if not hasattr(local, 'file'):
            local.file = pq.ParquetFile(source)
        seq_x, seq_y, seq_mask = _read_sequence(local.file, group_list[index])
        x[index] = seq_x
        y[index] = seq_y
        if has_mask:
            mask[index] = seq_mask

    todo = [i for i in range(count) if i not in progress['done']]
    with ThreadPoolExecutor(THREADS) as pool:
        for start in range(0, len(todo), SAVE_EVERY):
            chunk = todo[start:start + SAVE_EVERY]
            list(pool.map(write_one, chunk))
            # Data reaches the disk before it is recorded as done.
            x.flush()
            y.flush()
            if has_mask:
                mask.flush()
            progress['done'].update(chunk)
            _save_progress(split, progress)
            print(f'{split}: {len(progress["done"])}/{count}', flush=True)
    if folds is not None:
        np.save(paths['fold'], folds)
    progress['complete'] = True
    _save_progress(split, progress)


def limit_memory(gigabytes: float = WORKING_SET_LIMIT_GB) -> None:
    """Stop Windows keeping every cache page a run touches inside that one run.

    Without this, a few runs fill the RAM and pages get re-read from disk. Above the
    limit, pages move to the system's shared cache, where other runs can still use
    them without a disk read, and the system can reclaim them when it needs to.
    """
    if os.name != 'nt':
        return
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.SetProcessWorkingSetSizeEx.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_ulong]
    hard_maximum = 0x4
    if not kernel.SetProcessWorkingSetSizeEx(kernel.GetCurrentProcess(), 2 ** 27,
                                             int(gigabytes * 2 ** 30), hard_maximum):
        raise ctypes.WinError(ctypes.get_last_error())


def load(split: str) -> dict[str, np.ndarray]:
    limit_memory()
    path = _progress_path(split)
    if not path.exists() or not pickle.loads(path.read_bytes())['complete']:
        raise RuntimeError(f'the {split} cache is unfinished; run: python -m caviar.data --split {split}')
    paths = _paths(split)
    out = {'x': np.load(paths['x'], mmap_mode='r'), 'y': np.load(paths['y'], mmap_mode='r')}
    if split == 'valid':
        out['mask'] = np.load(paths['mask'])
    else:
        out['fold'] = np.load(paths['fold'])
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', required=True, choices=['r1', 'r2', 'valid', 'extra', 'rest'])
    build(parser.parse_args().split)
