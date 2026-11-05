"""Check the evaluation code against the supplied baseline, and check a trained run.

    python -m caviar.score_baseline
    python -m caviar.score_baseline --run caviar_runs/pilot/gru2_s0_f0.pt

The baseline ONNX file only accepts one row at a time, so scoring it takes about
50 minutes. Here its weights are copied into a torch GRU, shown to match ONNX, and
scored with the same evaluate() that scores every model. It must give 0.617052.
Writes reports/caviar_baseline.json.

With --run, also checks that a trained model gives the same output row by row as
in batches, and that changing later rows leaves earlier predictions unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import torch

from caviar import data as cache
from caviar.models import build, detach_state
from caviar.train import evaluate
from utils import N_FEATURES

ROOT = Path(__file__).resolve().parents[1]
ONNX_PATH = ROOT / 'baseline' / 'baseline.onnx'
OUT = ROOT / 'reports' / 'caviar_baseline.json'
HIDDEN = 128
EXPECTED_MASKED = 0.617052
MATCH_ROWS = 2000


class BaselineCopy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gru_0 = torch.nn.GRU(N_FEATURES, HIDDEN, batch_first=True)
        self.gru_1 = torch.nn.GRU(HIDDEN, HIDDEN, batch_first=True)
        self.head = torch.nn.Linear(HIDDEN, 2)

    def forward(self, x, state=None):
        first, second = state if state is not None else (None, None)
        h, first = self.gru_0(x, first)
        h, second = self.gru_1(h, second)
        return self.head(h), (first, second)


def _torch_gate_order(values: np.ndarray) -> np.ndarray:
    """ONNX stores GRU gates as update, reset, hidden. Torch stores reset, update, hidden."""
    update, reset, hidden = np.split(values, 3, axis=0)
    return np.concatenate([reset, update, hidden], axis=0)


def load_baseline() -> BaselineCopy:
    graph = onnx.load(ONNX_PATH).graph
    weights = {t.name: numpy_helper.to_array(t).copy() for t in graph.initializer}
    grus = [node for node in graph.node if node.op_type == 'GRU']
    head = next(node for node in graph.node if node.op_type == 'MatMul')
    model = BaselineCopy()
    with torch.no_grad():
        for gru, node in zip((model.gru_0, model.gru_1), grus):
            input_weight = weights[node.input[1]][0]
            hidden_weight = weights[node.input[2]][0]
            bias = weights[node.input[3]][0]
            gru.weight_ih_l0.copy_(torch.from_numpy(_torch_gate_order(input_weight)))
            gru.weight_hh_l0.copy_(torch.from_numpy(_torch_gate_order(hidden_weight)))
            gru.bias_ih_l0.copy_(torch.from_numpy(_torch_gate_order(bias[:3 * HIDDEN])))
            gru.bias_hh_l0.copy_(torch.from_numpy(_torch_gate_order(bias[3 * HIDDEN:])))
        model.head.weight.copy_(torch.from_numpy(weights[head.input[1]].T))
        model.head.bias.copy_(torch.from_numpy(weights['reg_head.bias']))
    return model.eval()


def check_matches_onnx(model: BaselineCopy, features: np.ndarray) -> float:
    session = ort.InferenceSession(str(ONNX_PATH), providers=['CPUExecutionProvider'])
    hidden_0 = np.zeros((1, 1, HIDDEN), np.float32)
    hidden_1 = np.zeros((1, 1, HIDDEN), np.float32)
    rows = []
    for row in features:
        out, hidden_0, hidden_1 = session.run(None, {
            'features': row.reshape(1, 1, N_FEATURES), 'hidden_0': hidden_0, 'hidden_1': hidden_1})
        rows.append(out[0, 0])
    with torch.no_grad():
        copied = model(torch.from_numpy(features[None]))[0][0].numpy()
    return float(np.abs(np.array(rows) - copied).max())


def score_baseline() -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    valid = cache.load('valid')
    model = load_baseline()
    difference = check_matches_onnx(model, np.array(valid['x'][0, :MATCH_ROWS]))
    print(f'largest difference from ONNX over {MATCH_ROWS} rows: {difference:.2e}', flush=True)
    if difference > 1e-3:
        raise RuntimeError('the torch copy does not match the ONNX baseline')

    model.to(device)
    scores = evaluate(model, valid['x'], valid['y'], np.arange(len(valid['x'])), device, valid['mask'])
    train = cache.load('r1')
    folds = {}
    for fold in sorted(set(train['fold'].tolist())):
        held = np.where(train['fold'] == fold)[0]
        folds[str(fold)] = evaluate(model, train['x'], train['y'], held, device)['all']
    result = {'onnx_difference': difference, 'valid_masked': scores['masked'],
              'valid_all': scores['all'], 'r1_folds': folds}
    OUT.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    gap = abs(scores['masked']['wp'] - EXPECTED_MASKED)
    print(f'masked {scores["masked"]["wp"]:.6f}, expected {EXPECTED_MASKED}, gap {gap:.6f}')
    if gap > 1e-4:
        raise RuntimeError('evaluate() does not reproduce the published baseline score')


def check_run(path: Path) -> None:
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = build(saved['key'])
    model.load_state_dict(saved['state_dict'])
    model.eval()
    valid = cache.load('valid')
    x = torch.from_numpy(np.array(valid['x'][0, :1500]))[None]
    with torch.no_grad():
        batch = model(x)[0]
        state, rows = None, []
        for t in range(x.shape[1]):
            out, state = model(x[:, t:t + 1], state)
            state = detach_state(state)
            rows.append(out)
        step_gap = (torch.cat(rows, 1) - batch).abs().max().item()
        changed = x.clone()
        changed[:, 1000:] = torch.randn_like(changed[:, 1000:])
        other = model(changed)[0]
        leak = (other[:, :1000] - batch[:, :1000]).abs().max().item()
        moved = (other[:, 1000:] - batch[:, 1000:]).abs().max().item()
    print(f'{path.name}: row by row vs batch gap {step_gap:.2e}, '
          f'earlier rows moved {leak:.2e} when later rows changed, later rows moved {moved:.2e}')
    if step_gap > 1e-3 or leak != 0 or moved == 0:
        raise RuntimeError('trained model failed the step or leakage check')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    if args.run:
        check_run(args.run)
    else:
        score_baseline()
