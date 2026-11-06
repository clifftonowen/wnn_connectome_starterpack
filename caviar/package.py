"""Build a submission ZIP from a trained run, and check the ZIP through the real scorer.

    python -m caviar.package build caviar_runs/r2_long_extra/group_gru_s0_f0.pt --name best
    python -m caviar.package build run_a.pt run_b.pt --name pair
    python -m caviar.package check caviar_runs/submission/best.zip --data subset
    python -m caviar.package check caviar_runs/submission/best.zip --data full --passes 2

The check unzips into an empty folder and loads solution.py from there, so it tests
exactly what gets uploaded. A full pass takes about 50 minutes on one thread, so run
it on a quiet machine. It passes when repeat passes give identical predictions, the
score matches training and the row time is under the limit. Writes
reports/caviar_submission_<name>.json.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
import zipfile

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'caviar_runs' / 'submission'
DATA = {'subset': ROOT / 'caviar_cache' / 'valid_subset.parquet', 'full': ROOT / 'datasets' / 'valid.parquet'}
MAX_MB = 20
LIMIT_US = 3600e6 / 39_400_000  # 60 minutes over the 39.4M test rows
SCORE_TOLERANCE = 1e-4


class Hashing:
    """Pass predictions through unchanged, recording a fingerprint of them."""

    def __init__(self, model):
        self.model = model
        self.hash = hashlib.blake2b(digest_size=16)

    def predict(self, point):
        value = self.model.predict(point)
        if value is not None:
            self.hash.update(value.tobytes())
        return value


def load_models(runs: list[Path]):
    import torch

    from caviar.models import build

    models = []
    for run in runs:
        saved = torch.load(run, map_location='cpu', weights_only=True)
        model = build(saved['key'])
        model.load_state_dict(saved['state_dict'])
        models.append(model.eval())
    return models


def build_zip(runs: list[Path], name: str) -> Path:
    from caviar.models import Averaged
    from caviar.speed import export

    results = [json.loads(run.with_suffix('.json').read_text(encoding='utf-8')) for run in runs]
    keys = sorted({result['key'] for result in results})
    # An average has no score from training, so the check then compares nothing.
    expected = results[0]['valid_masked_wp'] if len(runs) == 1 else None
    models = load_models(runs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f'{name}.zip'
    with tempfile.TemporaryDirectory() as folder:
        notes = f"# {' + '.join(keys)}\n\nAverage of {len(runs)} stateful model(s), ONNX, one row per call, on one CPU thread.\n"
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
            onnx_path = Path(folder) / 'model.onnx'
            export(models[0] if len(models) == 1 else Averaged(models), str(onnx_path))
            archive.write(Path(__file__).with_name('solution_template.py'), 'solution.py')
            archive.write(onnx_path, 'model.onnx')
            archive.writestr('MODEL.md', notes)
    size = path.stat().st_size / 2 ** 20
    if size > MAX_MB:
        raise SystemExit(f'{path.name} is {size:.1f} MB, over the {MAX_MB} MB limit')
    path.with_suffix('.json').write_text(json.dumps({
        'runs': [str(run) for run in runs], 'keys': keys, 'score_from_training': expected, 'zip_mb': size,
    }, indent=2), encoding='utf-8')
    print(f'{path}: {size:.2f} MB')
    return path


def load_solution(folder: str):
    spec = importlib.util.spec_from_file_location('solution', Path(folder) / 'solution.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PredictionModel


def check_zip(path: Path, data: str, passes: int, label: str = '', data_path: Path | None = None) -> None:
    from utils import ScorerStepByStep

    expected = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))['score_from_training']
    results = []
    with tempfile.TemporaryDirectory() as folder:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            archive.extractall(folder)
        if 'solution.py' not in names:
            raise SystemExit('solution.py is not at the ZIP root')
        model_class = load_solution(folder)
        for index in range(passes):
            model = Hashing(model_class())
            begin = time.perf_counter()
            score = ScorerStepByStep(data_path or DATA[data]).score(model)
            seconds = time.perf_counter() - begin
            results.append({'score': score['weighted_pearson'], 't0': score['t0'], 't1': score['t1'],
                            'seconds': seconds, 'us_per_row': seconds * 1e6 / score['rows_seen'],
                            'hash': model.hash.hexdigest()})
            print(f'pass {index}: {results[-1]}', flush=True)
    report = {
        'zip': path.name, 'data': data, 'files': names, 'passes': results, 'limit_us_per_row': LIMIT_US,
        'repeat_tested': passes > 1,
        'same_predictions': len({p['hash'] for p in results}) == 1,
        'under_limit': all(p['us_per_row'] < LIMIT_US for p in results),
    }
    if data == 'full' and expected is not None:
        report['score_from_training'] = expected
        report['score_gap'] = abs(results[0]['score'] - expected)
        report['score_matches'] = report['score_gap'] < SCORE_TOLERANCE
    suffix = f'_{data}{label}'
    out = ROOT / 'reports' / f'caviar_submission_{path.stem}{suffix}.json'
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    if not (report['same_predictions'] and report['under_limit'] and report.get('score_matches', True)):
        raise SystemExit('check failed')


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    make = commands.add_parser('build')
    make.add_argument('runs', type=Path, nargs='+')
    make.add_argument('--name', required=True)
    check = commands.add_parser('check')
    check.add_argument('zip', type=Path)
    check.add_argument('--data', choices=sorted(DATA), default='subset')
    check.add_argument('--passes', type=int, default=1)
    check.add_argument('--label', default='', help='added to the report name, such as _linux')
    check.add_argument('--data-path', type=Path, help='use this file instead of the default for --data')
    args = parser.parse_args()
    if args.command == 'build':
        build_zip(args.runs, args.name)
    else:
        check_zip(args.zip, args.data, args.passes, args.label, args.data_path)


if __name__ == '__main__':
    main()
