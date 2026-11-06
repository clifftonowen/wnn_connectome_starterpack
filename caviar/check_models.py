"""Test every model: no future leakage, and row-by-row output matches batch output.

    python -m caviar.check_models [--device cuda]
"""
from __future__ import annotations

import argparse
import copy
import traceback

import torch

from caviar.models import N_FEATURES, SPECS, TUNING, build, merge_encoders

ROWS = 300
CUT = 200
TOLERANCE = 2e-3
# S4D uses an FFT for many rows, whose rounding spreads across rows. In double
# precision this leak is about 1e-15, so it is rounding and not future data.
LEAK_LIMIT = {'s4d': 1e-5}


def check(key: str, device: str) -> dict:
    torch.manual_seed(0)
    model = build(key).to(device).eval()
    x = torch.randn(2, ROWS, N_FEATURES, device=device)
    with torch.no_grad():
        batch, _ = model(x)
        changed = x.clone()
        changed[:, CUT:] += 5.0
        other, _ = model(changed)
        leak = (batch[:, :CUT] - other[:, :CUT]).abs().max().item()
        reacts = (batch[:, CUT:] - other[:, CUT:]).abs().max().item()
        state = None
        rows = []
        for t in range(ROWS):
            y, state = model(x[:, t:t + 1], state)
            rows.append(y)
        stream_gap = (torch.cat(rows, 1) - batch).abs().max().item()
        finite = bool(torch.isfinite(batch).all())
        merge_gap = (merge_encoders(copy.deepcopy(model))(x)[0] - batch).abs().max().item()
    return {
        'params': sum(p.numel() for p in model.parameters()),
        'leak': leak,
        'reacts': reacts,
        'stream_gap': stream_gap,
        'merge_gap': merge_gap,
        'ok': (finite and leak <= LEAK_LIMIT.get(key, 0.0) and reacts > 0 and stream_gap < TOLERANCE
               and merge_gap < TOLERANCE),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--keys', nargs='*')
    args = parser.parse_args()
    failures = 0
    print(f"{'#':>2} {'key':<12} {'params':>9} {'leak':>9} {'reacts':>9} {'stream gap':>11} {'merge gap':>9}")
    for spec in [*SPECS, *TUNING]:
        if args.keys and spec.key not in args.keys:
            continue
        try:
            r = check(spec.key, args.device)
        except Exception:
            failures += 1
            print(f'{spec.id:>2} {spec.key:<12} ERROR')
            traceback.print_exc(limit=3)
            continue
        failures += not r['ok']
        print(f"{spec.id:>2} {spec.key:<12} {r['params']:>9,} {r['leak']:>9.2e} {r['reacts']:>9.2e} "
              f"{r['stream_gap']:>11.2e} {r['merge_gap']:>9.2e} {'ok' if r['ok'] else 'FAIL'}")
    print('failures:', failures)
    return failures


if __name__ == '__main__':
    raise SystemExit(main())
