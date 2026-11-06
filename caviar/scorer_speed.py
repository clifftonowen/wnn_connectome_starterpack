"""Time models through the real scorer on part of validation, a minute or so each.

    python -m caviar.scorer_speed --floors --keys instr_gru gru1_h96
    python -m caviar.scorer_speed --no-merge --keys instr_gru

--floors also times two lower bounds: the scorer with a constant prediction, and
the scorer with a tiny ONNX model. Time per row does not depend on the weights
much, so untrained models are fine for screening. Run on a quiet machine. Results
are added to reports/caviar_scorer_speed.csv.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
import onnx
import pandas as pd
import pyarrow.parquet as pq
import torch

from caviar.models import N_FEATURES, build
from caviar.speed import export
from caviar.verify_top import ROOT, VALID, StreamingModel
from utils import ScorerStepByStep

SUBSET = ROOT / 'caviar_cache' / 'valid_subset.parquet'
SEQUENCES = 30
OUT = ROOT / 'reports' / 'caviar_scorer_speed.csv'


class Constant:
    def predict(self, point):
        return np.zeros(2, dtype=np.float32) if point.need_prediction else None


class Tiny(torch.nn.Module):
    """About the smallest model with state: one matrix, and state passed through."""

    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(N_FEATURES, 2)

    def forward(self, x, state=None):
        state = torch.zeros(1, 1, 8) if state is None else state
        return self.head(x), state + 0


def make_subset() -> None:
    if SUBSET.exists():
        return
    source = pq.ParquetFile(VALID)
    temporary = SUBSET.with_suffix('.tmp')
    with pq.ParquetWriter(temporary, source.schema_arrow) as writer:
        for group in range(SEQUENCES):
            writer.write_table(source.read_row_group(group))
    os.replace(temporary, SUBSET)


def time_pass(model) -> tuple[float, int]:
    begin = time.perf_counter()
    rows = ScorerStepByStep(SUBSET).score(model)['rows_seen']
    return (time.perf_counter() - begin) * 1e6 / rows, rows


def time_model(key: str, model: torch.nn.Module, merge: bool, folder: str) -> dict:
    path = str(Path(folder) / f'{key}.onnx')
    params = sum(p.numel() for p in model.parameters())
    sess, names, zero = export(model, path, merge)
    us, rows = time_pass(StreamingModel(sess, names, zero))
    return {'key': key, 'merged': merge, 'us_per_row': round(us, 1), 'rows': rows, 'params': params,
            'onnx_nodes': len(onnx.load(path).graph.node)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--keys', nargs='*', default=[])
    parser.add_argument('--floors', action='store_true')
    parser.add_argument('--no-merge', action='store_true')
    args = parser.parse_args()
    make_subset()
    rows = []
    with tempfile.TemporaryDirectory() as folder:
        if args.floors:
            us, count = time_pass(Constant())
            rows.append({'key': 'floor_scorer', 'merged': False, 'us_per_row': round(us, 1), 'rows': count,
                         'params': 0, 'onnx_nodes': 0})
            print(rows[-1], flush=True)
            rows.append(time_model('floor_onnx', Tiny(), False, folder))
            print(rows[-1], flush=True)
        for key in args.keys:
            torch.manual_seed(0)
            rows.append(time_model(key, build(key), not args.no_merge, folder))
            print(rows[-1], flush=True)
    frame = pd.DataFrame(rows)
    if OUT.exists():
        old = pd.read_csv(OUT)
        keep = ~old.set_index(['key', 'merged']).index.isin(frame.set_index(['key', 'merged']).index)
        frame = pd.concat([old[keep], frame])
    frame.to_csv(OUT, index=False)


if __name__ == '__main__':
    main()
