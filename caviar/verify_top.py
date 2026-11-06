"""Final check on a trained model: export to ONNX and run the real scorer twice.

    python -m caviar.verify_top caviar_runs/r2/group_gru_s0_f0.pt

Each pass takes about 50 minutes on one thread, so run it on a quiet machine. It
passes when both runs give identical predictions, the score matches the one from
training, and the row time is under the limit. Writes
reports/caviar_verify_<name>.json and caviar_runs/verify/<name>.onnx.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
import torch

from caviar.models import N_FEATURES, build
from caviar.speed import LIMIT_US, export
from utils import ScorerStepByStep

ROOT = Path(__file__).resolve().parents[1]
VALID = ROOT / 'datasets' / 'valid.parquet'
OUT_DIR = ROOT / 'caviar_runs' / 'verify'
SCORE_TOLERANCE = 1e-4


class StreamingModel:
    """Same contract as baseline/solution.py: state updates on every row, resets per sequence."""

    def __init__(self, sess, names, zero):
        self.sess, self.names, self.zero = sess, names, zero
        self.state = [z.copy() for z in zero]
        self.current = None
        self.hash = hashlib.blake2b(digest_size=16)

    def predict(self, point):
        if point.seq_ix != self.current:
            self.current = point.seq_ix
            self.state = [z.copy() for z in self.zero]
        row = np.asarray(point.state, dtype=np.float32).reshape(1, 1, N_FEATURES)
        out = self.sess.run(None, {'x': row, **dict(zip(self.names, self.state))})
        self.state = out[1:]
        if not point.need_prediction:
            return None
        prediction = np.asarray(out[0][0, 0], dtype=np.float32)
        self.hash.update(prediction.tobytes())
        return prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--passes', type=int, default=2)
    args = parser.parse_args()

    saved = torch.load(args.run, map_location='cpu', weights_only=True)
    expected = json.loads(args.run.with_suffix('.json').read_text(encoding='utf-8'))
    model = build(saved['key'])
    model.load_state_dict(saved['state_dict'])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name = args.run.stem if args.run.parent.name == 'r2' else f'{args.run.parent.name}_{args.run.stem}'
    onnx_path = OUT_DIR / f'{name}.onnx'
    sess, names, zero = export(model, str(onnx_path))

    passes = []
    for index in range(args.passes):
        stream = StreamingModel(sess, names, zero)
        begin = time.perf_counter()
        score = ScorerStepByStep(VALID).score(stream)
        seconds = time.perf_counter() - begin
        passes.append({'score': score['weighted_pearson'], 't0': score['t0'], 't1': score['t1'],
                       'seconds': seconds, 'us_per_row': seconds * 1e6 / score['rows_seen'],
                       'hash': stream.hash.hexdigest()})
        print(f'pass {index}: {passes[-1]}', flush=True)

    gap = abs(passes[0]['score'] - expected['valid_masked_wp'])
    result = {
        'run': str(args.run), 'onnx_mb': onnx_path.stat().st_size / 2 ** 20, 'passes': passes,
        'score_from_training': expected['valid_masked_wp'], 'score_gap': gap,
        'limit_us_per_row': LIMIT_US,
        'same_predictions': len({p['hash'] for p in passes}) == 1,
        'score_matches': gap < SCORE_TOLERANCE,
        'under_limit': all(p['us_per_row'] < LIMIT_US for p in passes),
    }
    out = ROOT / 'reports' / f'caviar_verify_{name}.json'
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    if not (result['same_predictions'] and result['score_matches'] and result['under_limit']):
        raise SystemExit('verification failed')


if __name__ == '__main__':
    main()
