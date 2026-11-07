"""Train one model on one fold and score it. Finished runs are skipped.

    python -m caviar.train --key gru2 --round r1 --fold 0

Fold -1 holds out nothing, for a final model. It has no fold score, so judge it on
validation.

Writes caviar_runs/<round>/<key>_s<seed>_f<fold>.json and .pt. Valid.parquet is
only ever scored, never trained on, and held-out fold rows are not used for early
stopping: a small dev set split from the training folds is used instead.

Safe to stop and restart: a stopped run continues from <name>.ckpt.pkl, saved at
each dev-set check, which is deleted once the run finishes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from caviar import data as cache
from caviar.models import BY_KEY, build, detach_state
from utils import GlobalAccumulator, SEQUENCE_LENGTH, WARMUP, weighted_pearson

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'caviar_runs'
CLIP = 2.0
DEFAULTS = dict(batch=64, window=1024, max_steps=3000, lr=1e-3, weight_decay=1e-5,
                warmup_steps=100, eval_every=250, patience=5, grad_clip=1.0,
                dev_fraction=0.05, dev_rows=5000, ema=0.0, dropout=0.0, noise=0.0)
EVAL_BATCH = 64
EVAL_CHUNK = 2000
CPU_THREADS = 2
GPU_FRACTION = 0.45


def scaling(x: np.ndarray, indices: np.ndarray, rng: np.random.Generator):
    picked = rng.choice(indices, size=min(64, len(indices)), replace=False)
    sample = np.concatenate([np.asarray(x[i, ::20]) for i in np.sort(picked)])
    return torch.from_numpy(sample.mean(0)), torch.from_numpy(sample.std(0))


def weighted_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """MSE weighted by |clipped target|, over rows after warm-up handling."""
    y = y.clamp(-CLIP, CLIP)
    w = y.abs()
    return (w * (pred - y) ** 2).sum() / w.sum().clamp_min(1.0)


class Joined:
    """Cached arrays read as one, for the sampler's (sequence, rows) lookups."""

    def __init__(self, *arrays):
        self.arrays = arrays
        self.starts = np.cumsum([0, *[len(a) for a in arrays[:-1]]])

    def __getitem__(self, key):
        sequence, rows = key
        part = int(np.searchsorted(self.starts, sequence, side='right')) - 1
        return self.arrays[part][sequence - self.starts[part], rows]


class Sampler:
    def __init__(self, x, y, indices, batch, window, rng, device):
        self.x, self.y, self.indices = x, y, indices
        self.batch, self.window, self.rng, self.device = batch, window, rng, device

    def __call__(self):
        seqs = self.rng.choice(self.indices, self.batch)
        starts = self.rng.integers(0, SEQUENCE_LENGTH - self.window + 1, self.batch)
        xb = np.stack([self.x[s, b:b + self.window] for s, b in zip(seqs, starts)])
        yb = np.stack([self.y[s, b:b + self.window] for s, b in zip(seqs, starts)])
        return (torch.from_numpy(xb).to(self.device, non_blocking=True),
                torch.from_numpy(yb).to(self.device, non_blocking=True),
                torch.from_numpy(starts).to(self.device))


@torch.no_grad()
def predict_rows(model, xb: np.ndarray, device) -> np.ndarray:
    """Predict whole sequences in pieces, carrying state, as a stream would."""
    state = None
    out = []
    for start in range(0, xb.shape[1], EVAL_CHUNK):
        part = torch.from_numpy(np.array(xb[:, start:start + EVAL_CHUNK])).to(device)
        pred, state = model(part, state)
        state = detach_state(state)
        out.append(pred.cpu().numpy())
    return np.concatenate(out, 1)


def evaluate(model, x, y, indices, device, masks=None) -> dict:
    """Global WP over all prediction rows, and over the scoring mask when given."""
    model.eval()
    full, masked = GlobalAccumulator(), GlobalAccumulator()
    need = np.arange(SEQUENCE_LENGTH) >= WARMUP
    for start in range(0, len(indices), EVAL_BATCH):
        batch = np.sort(indices[start:start + EVAL_BATCH])
        preds = predict_rows(model, x[batch], device)
        targets = np.asarray(y[batch])
        for j, seq in enumerate(batch):
            full.add(targets[j], preds[j], need)
            if masks is not None:
                masked.add(targets[j], preds[j], masks[seq] & need)
    result = {'all': _short(full.result())}
    if masks is not None:
        result['masked'] = _short(masked.result())
    return result


def _short(report: dict) -> dict:
    return {'wp': report['weighted_pearson'], 't0': report['t0'], 't1': report['t1'],
            'rows': report['selected_rows']}


@torch.no_grad()
def dev_score(model, x, y, indices, rows, device) -> float:
    model.eval()
    preds, targets = [], []
    for start in range(0, len(indices), EVAL_BATCH):
        batch = np.sort(indices[start:start + EVAL_BATCH])
        preds.append(predict_rows(model, x[batch, :rows], device)[:, WARMUP:])
        targets.append(np.asarray(y[batch, WARMUP:rows]))
    p = np.concatenate(preds).reshape(-1, 2)
    t = np.concatenate(targets).reshape(-1, 2)
    return float(np.mean([weighted_pearson(t[:, k], p[:, k]) for k in range(2)]))


def save_checkpoint(path: Path, content: dict) -> None:
    temporary = path.with_suffix('.tmp')
    torch.save(content, temporary)
    os.replace(temporary, path)


def fit_gradient(model, x, y, train_idx, dev_idx, cfg, rng, device, log, checkpoint: Path,
                 pool=None) -> dict:
    model.train()
    model.dropout = cfg['dropout']
    decay = [p for p in model.parameters() if p.ndim > 1]
    other = [p for p in model.parameters() if p.ndim <= 1]
    optimizer = torch.optim.AdamW([{'params': decay, 'weight_decay': cfg['weight_decay']},
                                   {'params': other, 'weight_decay': 0.0}], lr=cfg['lr'])

    def schedule(step):
        if step < cfg['warmup_steps']:
            return (step + 1) / cfg['warmup_steps']
        progress = (step - cfg['warmup_steps']) / max(1, cfg['max_steps'] - cfg['warmup_steps'])
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    lr_schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    pool_x, pool_y = pool or (x, y)
    sample = Sampler(pool_x, pool_y, train_idx, cfg['batch'], cfg['window'], rng, device)
    rows = torch.arange(cfg['window'], device=device)
    best, best_step, bad, history, step = -math.inf, 0, 0, [], 0
    best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    trained_rows, train_seconds, running = 0, 0.0, []
    # A slowly updated copy of the weights. When on, it is what gets scored and kept.
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()} if cfg['ema'] else None
    if checkpoint.exists():
        # Our own file, so the full pickle load needed for random states is fine.
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved['config'] != cfg:
            raise ValueError(f'{checkpoint.name} used different settings; rerun with --force to discard it')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        lr_schedule.load_state_dict(saved['schedule'])
        best, best_step, bad = saved['best'], saved['best_step'], saved['bad']
        best_weights, history, step = saved['best_weights'], saved['history'], saved['step']
        trained_rows, train_seconds = saved['trained_rows'], saved['train_seconds']
        ema = saved['ema']
        rng.bit_generator.state = saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng'].cpu())
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
        log(f'resumed from step {step}')
    begin = time.perf_counter()
    first_step = step + 1 if bad < cfg['patience'] else cfg['max_steps'] + 1
    for step_number in range(first_step, cfg['max_steps'] + 1):
        step = step_number
        xb, yb, starts = sample()
        if cfg['noise']:
            xb = xb + cfg['noise'] * model.std * torch.randn_like(xb)
        pred, _ = model(xb)
        keep = (starts.unsqueeze(1) + rows) >= WARMUP
        loss = weighted_loss(pred[keep], yb[keep])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
        optimizer.step()
        lr_schedule.step()
        if ema is not None:
            # Early on the copy follows the weights closely, so it forgets the random start.
            keep = min(cfg['ema'], (1 + step) / (10 + step))
            for name, value in model.state_dict().items():
                ema[name].lerp_(value.detach(), 1 - keep)
        running.append(loss.item())
        trained_rows += xb.shape[0] * xb.shape[1]
        if step % cfg['eval_every'] == 0 or step == cfg['max_steps']:
            train_seconds += time.perf_counter() - begin
            live = None
            if ema is not None:
                live = {k: v.detach().clone() for k, v in model.state_dict().items()}
                model.load_state_dict(ema)
            score = dev_score(model, x, y, dev_idx, cfg['dev_rows'], device)
            history.append({'step': step, 'loss': float(np.mean(running)), 'dev_wp': score})
            log(f"step {step} loss {history[-1]['loss']:.4f} dev_wp {score:.4f} "
                f'{trained_rows / train_seconds:,.0f} rows/s')
            running = []
            if math.isfinite(score) and score > best:
                best, best_step, bad = score, step, 0
                best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            if live is not None:
                model.load_state_dict(live)
            save_checkpoint(checkpoint, {
                'config': cfg, 'step': step, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'schedule': lr_schedule.state_dict(),
                'best': best, 'best_step': best_step, 'bad': bad, 'best_weights': best_weights,
                'history': history, 'trained_rows': trained_rows, 'train_seconds': train_seconds, 'ema': ema,
                'numpy_rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            })
            if bad >= cfg['patience']:
                break
            model.train()
            begin = time.perf_counter()
    model.load_state_dict(best_weights)
    return {'best_step': best_step, 'stopped_step': step, 'dev_wp': best, 'history': history,
            'train_seconds': train_seconds, 'train_rows_per_second': trained_rows / max(train_seconds, 1e-9)}


@torch.no_grad()
def fit_ridge(model, x, y, train_idx, rng, device, log) -> dict:
    """Closed-form weighted ridge on the model's features, separately per target."""
    model.eval()
    picked = rng.choice(train_idx, size=min(256, len(train_idx)), replace=False)
    begin = time.perf_counter()
    size = model.head.in_features + 1
    gram = torch.zeros(2, size, size, dtype=torch.float64, device=device)
    cross = torch.zeros(2, size, dtype=torch.float64, device=device)
    rows = 0
    for start in range(0, len(picked), EVAL_BATCH):
        batch = np.sort(picked[start:start + EVAL_BATCH])
        xb = torch.from_numpy(np.array(x[batch, :5000])).to(device)
        yb = torch.from_numpy(np.array(y[batch, :5000])).to(device).clamp(-CLIP, CLIP)
        h, _ = model.features(xb)
        keep = torch.arange(h.shape[1], device=device)[WARMUP::5]
        h = h[:, keep].reshape(-1, h.shape[-1]).double()
        h = torch.cat([h, torch.ones(len(h), 1, device=device, dtype=torch.float64)], 1)
        t = yb[:, keep].reshape(-1, 2).double()
        for k in range(2):
            w = t[:, k].abs().unsqueeze(1)
            gram[k] += (h * w).T @ h
            cross[k] += (h * w).T @ t[:, k]
        rows += len(h)
    weights = []
    for k in range(2):
        ridge = 1e-3 * gram[k].diagonal().mean() * torch.eye(size, device=device, dtype=torch.float64)
        ridge[-1, -1] = 0
        weights.append(torch.linalg.solve(gram[k] + ridge, cross[k]))
    weights = torch.stack(weights).float()
    model.head.weight.copy_(weights[:, :-1])
    model.head.bias.copy_(weights[:, -1])
    log(f'ridge fit on {rows:,} rows')
    return {'best_step': 0, 'stopped_step': 0, 'dev_wp': None, 'history': [],
            'train_seconds': time.perf_counter() - begin, 'train_rows_per_second': None}


@torch.no_grad()
def fit_constant(model, y, train_idx, device) -> dict:
    means = np.clip(np.asarray(y[np.sort(train_idx[:64]), WARMUP:]), -CLIP, CLIP).mean((0, 1))
    model.head.weight.zero_()
    model.head.bias.copy_(torch.from_numpy(means).to(device))
    return {'best_step': 0, 'stopped_step': 0, 'dev_wp': None, 'history': [],
            'train_seconds': 0.0, 'train_rows_per_second': None}


def valid_halves(count: int) -> tuple[np.ndarray, np.ndarray]:
    """Fixed halves of validation: A is the first half in file order (sorted seq_ix), B the rest.

    seq_ix looks like a random id, so this is an unbiased split, and it never changes.
    """
    return np.arange(count // 2), np.arange(count // 2, count)


def run(key, round_name, fold, seed, overrides, tag=None, force=False, log=print,
        gpu_fraction=GPU_FRACTION, extra_split=None, valid_train=None) -> dict:
    out_dir = RUNS / (tag or round_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f'{key}_s{seed}_f{fold}.json'
    checkpoint = out_dir / f'{key}_s{seed}_f{fold}.ckpt.pkl'
    if out_json.exists() and not force:
        log(f'{out_json.name} already finished')
        return json.loads(out_json.read_text(encoding='utf-8'))
    if force:
        checkpoint.unlink(missing_ok=True)

    cfg = {**DEFAULTS, **overrides}
    if extra_split:
        cfg['extra_split'] = extra_split
    if valid_train:
        cfg['valid_train'] = valid_train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed + 1000 * max(fold, 0))
    rng = np.random.default_rng(seed + 1000 * max(fold, 0))
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(CPU_THREADS)
    if torch.cuda.is_available():
        # Past the GPU's memory, Windows swaps it and every run slows about tenfold.
        # With this cap a run that needs too much stops with a clear error instead.
        torch.cuda.set_per_process_memory_fraction(gpu_fraction)

    train_data, valid_data = cache.load(round_name), cache.load('valid')
    x, y, folds = train_data['x'], train_data['y'], train_data['fold']
    held = np.where(folds == fold)[0]
    rest = rng.permutation(np.where(folds != fold)[0])
    n_dev = max(8, int(len(rest) * cfg['dev_fraction']))
    dev_idx, train_idx = rest[:n_dev], rest[n_dev:]
    assert not set(held) & set(train_idx) and not set(held) & set(dev_idx)

    model = build(key).to(device)
    model.set_scaling(*[t.to(device) for t in scaling(x, train_idx, rng)])
    spec = BY_KEY[key]
    pool = None
    # Training only: dev, held-out rows and input scaling stay on the round's own sequences.
    # Whole arrays go into the pool, read lazily, so nothing large is copied into memory.
    extras = [cache.load(split) for split in extra_split or []]
    added = sum(len(e['x']) for e in extras)
    scored = np.arange(len(valid_data['x']))
    if valid_train:
        trained, scored = valid_halves(len(valid_data['x']))
        extras.append(valid_data)
    if extras:
        pool = Joined(x, *[e['x'] for e in extras]), Joined(y, *[e['y'] for e in extras])
        train_idx = np.concatenate([train_idx, len(x) + np.arange(added)])
        if valid_train:
            train_idx = np.concatenate([train_idx, len(x) + added + trained])
            log(f'adding validation half {valid_train}, {len(trained)} sequences; scoring the other half')
        log(f'training on {len(train_idx)} sequences, {len(train_idx) - len(rest) + n_dev} of them extra')
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    if spec.fit == 'gradient':
        fit = fit_gradient(model, x, y, train_idx, dev_idx, cfg, rng, device, log, checkpoint, pool)
    elif spec.fit == 'ridge':
        fit = fit_ridge(model, x, y, train_idx, rng, device, log)
    else:
        fit = fit_constant(model, y, train_idx, device)

    begin = time.perf_counter()
    cv = evaluate(model, x, y, held, device)['all'] if len(held) else dict.fromkeys(['wp', 't0', 't1'])
    valid = evaluate(model, valid_data['x'], valid_data['y'], scored, device, valid_data['mask'])
    eval_seconds = time.perf_counter() - begin

    if valid_train:
        # Only half B was scored. Keep it out of the full-validation columns so tables can't mix them.
        scores = {'validB_masked_wp': valid['masked']['wp'], 'validB_all_wp': valid['all']['wp'],
                  'validB_masked_rows': valid['masked']['rows'],
                  'valid_masked_wp': None, 'valid_all_wp': None}
    else:
        scores = {'valid_masked_wp': valid['masked']['wp'], 'valid_masked_t0': valid['masked']['t0'],
                  'valid_masked_t1': valid['masked']['t1'], 'valid_masked_rows': valid['masked']['rows'],
                  'valid_all_wp': valid['all']['wp'], 'valid_all_t0': valid['all']['t0'],
                  'valid_all_t1': valid['all']['t1']}
    result = {
        'id': spec.id, 'key': key, 'name': spec.name, 'source': spec.source, 'round': round_name,
        'fold': fold, 'seed': seed, 'config': cfg,
        'params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'cv_wp': cv['wp'], 'cv_t0': cv['t0'], 'cv_t1': cv['t1'],
        **scores, 'eval_seconds': eval_seconds,
        'peak_gpu_mb': torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else None,
        **fit,
    }
    torch.save({'key': key, 'state_dict': {k: v.cpu() for k, v in model.state_dict().items()}},
               out_json.with_suffix('.pt'))
    temporary = out_json.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2), encoding='utf-8')
    temporary.replace(out_json)
    checkpoint.unlink(missing_ok=True)
    log(f"done cv {cv['wp']} valid_masked {valid['masked']['wp']:.4f} valid_all {valid['all']['wp']:.4f}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--key', required=True, choices=sorted(BY_KEY))
    parser.add_argument('--round', default='r1', choices=['r1', 'r2'])
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tag', default=None, help='results folder name, defaults to the round')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--gpu-fraction', type=float, default=GPU_FRACTION,
                        help='raise it to run one memory-heavy model alone')
    parser.add_argument('--extra-split', nargs='+', help='extra cached sequences to train on, such as extra rest')
    parser.add_argument('--valid-train', choices=['A'],
                        help='also train on validation half A, and score validation on half B only')
    for name, value in DEFAULTS.items():
        parser.add_argument('--' + name.replace('_', '-'), type=type(value), default=value)
    args = parser.parse_args()
    overrides = {name: getattr(args, name) for name in DEFAULTS}
    run(args.key, args.round, args.fold, args.seed, overrides, args.tag, args.force,
        log=lambda message: print(f'[{args.key} f{args.fold}] {message}', flush=True),
        gpu_fraction=args.gpu_fraction, extra_split=args.extra_split, valid_train=args.valid_train)


if __name__ == '__main__':
    main()
