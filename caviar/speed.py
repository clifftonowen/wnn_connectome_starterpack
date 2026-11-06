"""Export each model's single-row step to ONNX and time it on one CPU thread.

    python -m caviar.speed [--keys gru2 mgu] [--rows 20000]

Uses the same ONNX Runtime settings as baseline/solution.py and calls it once per
row from Python, which is how a submission would run. Results go to
reports/caviar_speed.csv. Weights are untrained: speed does not depend on them.
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
import onnxruntime as ort
import pandas as pd
import torch

from caviar.models import N_FEATURES, SPECS, TUNING, build, flatten_state, merge_encoders, unflatten_state

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports' / 'caviar_speed.csv'
LIMIT_US = 3600e6 / 39_400_000  # 60 minutes over the 39.4M test rows
RISKY_US = 60.0


class StepModel(torch.nn.Module):
    def __init__(self, model, template):
        super().__init__()
        self.model = model
        self.template = template

    def forward(self, x, *flat):
        pred, state = self.model(x, unflatten_state(self.template, flat))
        return (pred, *flatten_state(state))


def session(path: str) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry('session.intra_op.allow_spinning', '0')
    options.add_session_config_entry('session.inter_op.allow_spinning', '0')
    return ort.InferenceSession(path, sess_options=options, providers=['CPUExecutionProvider'])


def export(model, path: str, merge: bool = True):
    """Export the row step and confirm ONNX matches torch. Returns the zero state."""
    model = (merge_encoders(model) if merge else model).eval()
    x = torch.randn(1, 1, N_FEATURES)
    with torch.no_grad():
        _, template = model(x)
        flat = [torch.zeros_like(t) for t in flatten_state(template)]
        names = [f'state_{i}' for i in range(len(flat))]
        step = StepModel(model, template)
        torch.onnx.export(step, (x, *flat), path, input_names=['x', *names],
                          output_names=['pred', *[f'next_{i}' for i in range(len(flat))]],
                          opset_version=17, dynamo=False)
        sess = session(path)
        state = [t.numpy() for t in flat]
        torch_state = unflatten_state(template, flat)
        for _ in range(8):
            row = torch.randn(1, 1, N_FEATURES)
            pred, torch_state = model(row, torch_state)
            out = sess.run(None, {'x': row.numpy(), **dict(zip(names, state))})
            state = out[1:]
            gap = np.abs(out[0] - pred.numpy()).max()
            if gap > 1e-3:
                raise ValueError(f'ONNX differs from torch by {gap}')
    return sess, names, [t.numpy() for t in flat]


def time_steps(sess, names, zero, rows: int) -> float:
    features = np.random.default_rng(0).standard_normal((rows, N_FEATURES), dtype=np.float32)
    state = [z.copy() for z in zero]
    begin = time.perf_counter()
    for t in range(rows):
        out = sess.run(None, {'x': features[t].reshape(1, 1, N_FEATURES), **dict(zip(names, state))})
        state = out[1:]
        _ = np.asarray(out[0][0, 0], dtype=np.float32)
    return (time.perf_counter() - begin) * 1e6 / rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--keys', nargs='*', default=[s.key for s in SPECS])
    parser.add_argument('--rows', type=int, default=20_000)
    args = parser.parse_args()
    rows = []
    with tempfile.TemporaryDirectory() as folder:
        for spec in [*SPECS, *TUNING]:
            if spec.key not in args.keys:
                continue
            path = str(Path(folder) / f'{spec.key}.onnx')
            torch.manual_seed(0)
            try:
                sess, names, zero = export(build(spec.key), path)
                us = time_steps(sess, names, zero, args.rows)
                size = Path(path).stat().st_size / 2 ** 20
                row = {'id': spec.id, 'key': spec.key, 'us_per_row': round(us, 1), 'onnx_mb': round(size, 2),
                       'status': 'over' if us > LIMIT_US else 'risky' if us > RISKY_US else 'ok', 'error': ''}
                del sess
            except Exception as error:
                row = {'id': spec.id, 'key': spec.key, 'us_per_row': None, 'onnx_mb': None,
                       'status': 'export_failed', 'error': str(error)[:200]}
            rows.append(row)
            print(row, flush=True)
    frame = pd.DataFrame(rows)
    if OUT.exists() and set(args.keys) != {s.key for s in SPECS}:
        old = pd.read_csv(OUT)
        frame = pd.concat([old[~old['key'].isin(frame['key'])], frame]).sort_values('id')
    frame.to_csv(OUT, index=False)
    print(f'limit {LIMIT_US:.1f} us per row, risky above {RISKY_US:.0f}')


if __name__ == '__main__':
    main()
