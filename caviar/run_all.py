"""Run many models side by side, one GPU slot per job. Safe to stop and restart.

    python -m caviar.run_all --round r1 --parallel 3
    python -m caviar.run_all --round r2 --parallel 3 --keys gru2 mgu --seeds 0
    python -m caviar.run_all --round r2 --keys gru1_h128 --folds 0 1 2
    python -m caviar.run_all --round r2 --keys gru1_h128 --folds 0 1 2 --tag r2_long --max-steps 12000
    python -m caviar.run_all --collect

A tag keeps runs with other settings in their own folder. It must start with r, so
that --collect finds it. Finished runs are skipped. Your 25 models run first, then the references, then my
suggestions, so a time cut removes suggestions first. Each job logs to
caviar_runs/<round>/logs/. Results are gathered into reports/caviar_results.csv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd

from caviar.models import REFERENCES, SPECS, TUNING

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'caviar_runs'
OUT = ROOT / 'reports' / 'caviar_results.csv'
FOLD_COUNT = {'r1': 3, 'r2': 5}
GBM = 'ema_gbm'


def job_order(round_name, keys, seeds, folds=None):
    order = [s.key for s in SPECS if s.source == 'user']
    order += [s.key for s in REFERENCES]
    order += [s.key for s in SPECS if s.source == 'suggested'] + [GBM]
    if keys:
        order = [k for k in [*order, *[s.key for s in TUNING]] if k in keys]
        unknown = set(keys) - set(order)
        if unknown:
            raise ValueError(f'unknown keys: {sorted(unknown)}')
    folds = range(FOLD_COUNT[round_name]) if folds is None else folds
    return [(k, f, seed) for k in order for seed in seeds for f in folds]


def command(key, round_name, fold, seed, extra, tag=None):
    module = 'caviar.gbm' if key == GBM else 'caviar.train'
    cmd = [sys.executable, '-m', module, '--round', round_name, '--fold', str(fold), '--seed', str(seed)]
    if key != GBM:
        cmd += ['--key', key, *extra]
        if tag:
            cmd += ['--tag', tag]
    return cmd


def collect() -> pd.DataFrame:
    rows = []
    for path in RUNS.glob('r*/*.json'):
        record = json.loads(path.read_text(encoding='utf-8'))
        record.pop('history', None)
        record.pop('config', None)
        record['tag'] = path.parent.name
        rows.append(record)
    frame = pd.DataFrame(rows)
    if len(frame):
        frame = frame.sort_values(['round', 'id', 'seed', 'fold'])
        frame.to_csv(OUT, index=False)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', default='r1', choices=['r1', 'r2'])
    parser.add_argument('--parallel', type=int, default=3)
    parser.add_argument('--keys', nargs='*')
    parser.add_argument('--seeds', nargs='*', type=int, default=[0])
    parser.add_argument('--folds', nargs='*', type=int)
    parser.add_argument('--tag', help='results folder for runs with other settings; must start with r')
    parser.add_argument('--collect', action='store_true')
    args, extra = parser.parse_known_args()
    if args.collect:
        print(len(collect()), 'runs collected')
        return

    if args.tag and not args.tag.startswith('r'):
        parser.error('--tag must start with r')
    folder = RUNS / (args.tag or args.round)
    log_dir = folder / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    queue = [job for job in job_order(args.round, args.keys, args.seeds, args.folds)
             if not (folder / f'{job[0]}_s{job[2]}_f{job[1]}.json').exists()]
    print(f'{len(queue)} jobs to run', flush=True)
    running, failed = [], []
    begin = time.time()
    while queue or running:
        while queue and len(running) < args.parallel:
            key, fold, seed = queue.pop(0)
            log = open(log_dir / f'{key}_s{seed}_f{fold}.log', 'a', encoding='utf-8')
            process = subprocess.Popen(command(key, args.round, fold, seed, extra, args.tag),
                                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            running.append((process, log, (key, fold, seed)))
            print(f'{time.time() - begin:8.0f}s start {key} fold {fold} seed {seed}', flush=True)
        for item in list(running):
            process, log, job = item
            if process.poll() is not None:
                log.close()
                running.remove(item)
                if process.returncode:
                    failed.append(job)
                print(f'{time.time() - begin:8.0f}s end {job} exit {process.returncode}, '
                      f'{len(queue)} queued', flush=True)
                collect()
        time.sleep(2)
    collect()
    print('failed:', failed, flush=True)


if __name__ == '__main__':
    main()
