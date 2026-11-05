"""All caviar models behind one interface.

Every layer has forward(x, state=None) -> (output, new_state), where x is
(batch, rows, channels). Passing one row at a time with the returned state must
give the same output as passing many rows at once, and no output may depend on
later rows. check_models.py tests both for every model.

State is a nested list of tensors. None means "start of sequence".
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F

N_FEATURES = 112
ATTENTION_CHUNK = 256


def _states(state, count):
    return [None] * count if state is None else state


def linear_recurrence(u, coef, start, varying):
    """Solve y[t] = coef[t] * y[t - 1] + u[t] for all rows at once, with y[-1] = start.

    A Python loop over rows is very slow on a GPU. Here each pass adds in the part
    of the sum that lies `shift` rows back, and shift doubles every pass, so 1,024
    rows take 10 passes. Set varying when coef changes by row. Otherwise coef has no
    row dimension, and broadcasts from the right.
    """
    first = coef[:, 0] if varying else coef
    y = torch.cat([(u[:, 0] + first * start).unsqueeze(1), u[:, 1:]], 1)
    shift = 1
    while shift < u.shape[1]:
        if varying:
            later = coef[:, shift:]
            y = torch.cat([y[:, :shift], y[:, shift:] + later * y[:, :-shift]], 1)
            coef = torch.cat([coef[:, :shift], later * coef[:, :-shift]], 1)
        else:
            y = torch.cat([y[:, :shift], y[:, shift:] + coef * y[:, :-shift]], 1)
            coef = coef * coef
        shift *= 2
    return y


class Chain(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, state=None):
        state = _states(state, len(self.layers))
        new = []
        for layer, layer_state in zip(self.layers, state):
            x, s = layer(x, layer_state)
            new.append(s)
        return x, new


class Plain(nn.Module):
    """A stateless module used as a layer."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, state=None):
        return self.module(x), None


def linear(c_in, c_out):
    return Plain(nn.Linear(c_in, c_out))


class Feedforward(nn.Module):
    def __init__(self, dim, factor=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim * factor), nn.GELU(), nn.Linear(dim * factor, dim))

    def forward(self, x):
        return self.net(x)


class Residual(nn.Module):
    """Normalize, apply a layer, add the input back."""

    def __init__(self, dim, layer):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layer = layer

    def forward(self, x, state=None):
        y, s = self.layer(self.norm(x), state)
        return x + y, s


def res_ff(dim):
    return Residual(dim, Plain(Feedforward(dim)))


# ---------------------------------------------------------------- recurrent


class TorchRNN(nn.Module):
    """nn.GRU, nn.LSTM or nn.RNN with states stored in a list."""

    def __init__(self, kind, c_in, hidden, layers=1, proj=0):
        super().__init__()
        cls = {'gru': nn.GRU, 'lstm': nn.LSTM, 'rnn': nn.RNN}[kind]
        extra = {'proj_size': proj} if proj else {}
        self.rnn = cls(c_in, hidden, layers, batch_first=True, **extra)
        self.is_lstm = kind == 'lstm'

    def forward(self, x, state=None):
        if self.is_lstm:
            y, s = self.rnn(x, None if state is None else tuple(state))
            return y, list(s)
        y, s = self.rnn(x, None if state is None else state[0])
        return y, [s]


class Cell(nn.Module):
    """Base for recurrences that step through rows one at a time."""

    def zero(self, x):
        raise NotImplementedError

    def prepare(self):
        return None

    def step(self, x, s, shared):
        raise NotImplementedError

    def forward(self, x, state=None):
        s = self.zero(x) if state is None else state
        shared = self.prepare()
        out = []
        for t in range(x.shape[1]):
            y, s = self.step(x[:, t], s, shared)
            out.append(y)
        return torch.stack(out, 1), s


class MGU(Cell):
    """Minimal gated unit: a GRU with one gate."""

    def __init__(self, c_in, hidden):
        super().__init__()
        self.hidden = hidden
        self.gate = nn.Linear(c_in + hidden, hidden)
        self.update = nn.Linear(c_in + hidden, hidden)

    def zero(self, x):
        return [x.new_zeros(x.shape[0], self.hidden)]

    def step(self, x, s, shared):
        h = s[0]
        f = torch.sigmoid(self.gate(torch.cat([x, h], -1)))
        n = torch.tanh(self.update(torch.cat([x, f * h], -1)))
        h = (1 - f) * h + f * n
        return h, [h]


class SRU(nn.Module):
    """Simple recurrent unit (Lei et al. 2017). The gates read only the input."""

    def __init__(self, hidden):
        super().__init__()
        self.hidden = hidden
        self.proj = nn.Linear(hidden, hidden)
        self.f = nn.Linear(hidden, hidden)
        self.r = nn.Linear(hidden, hidden)

    def forward(self, x, state=None):
        start = x.new_zeros(x.shape[0], self.hidden) if state is None else state[0]
        f = torch.sigmoid(self.f(x))
        c = linear_recurrence((1 - f) * self.proj(x), f, start, varying=True)
        r = torch.sigmoid(self.r(x))
        return r * c + (1 - r) * x, [c[:, -1]]


class QRNN(nn.Module):
    """Quasi-recurrent layer: causal convolution, then a gated running state."""

    def __init__(self, c_in, hidden, kernel=2):
        super().__init__()
        self.hidden = hidden
        self.conv = CausalConv(c_in, 3 * hidden, kernel)

    def forward(self, x, state=None):
        conv_state, c = (None, x.new_zeros(x.shape[0], self.hidden)) if state is None else state
        y, conv_state = self.conv(x, conv_state)
        z, f, o = y.chunk(3, -1)
        z, f, o = torch.tanh(z), torch.sigmoid(f), torch.sigmoid(o)
        running = linear_recurrence((1 - f) * z, f, c, varying=True)
        return o * running, [conv_state, running[:, -1]]


class TwoSpeedGRU(nn.Module):
    """Fast GRU on every row. Slow GRU reads the fast output every `every` rows."""

    def __init__(self, c_in, fast, slow, every=16):
        super().__init__()
        self.fast = TorchRNN('gru', c_in, fast)
        self.slow = TorchRNN('gru', fast, slow)
        self.every = every
        self.slow_size = slow

    def forward(self, x, state=None):
        batch = x.shape[0]
        if state is None:
            fast_state = None
            slow_state = [x.new_zeros(1, batch, self.slow_size)]
            count = x.new_zeros(1)
        else:
            fast_state, slow_state, count = state
        y, fast_state = self.fast(x, fast_state)
        hidden = slow_state[0]
        if x.shape[1] > 1:
            return self._many_rows(y, fast_state, hidden, count)
        out = []
        for t in range(x.shape[1]):
            count = count + 1
            _, s = self.slow(y[:, t:t + 1], [hidden])
            hidden = torch.where(torch.fmod(count, self.every) == 0, s[0], hidden)
            out.append(hidden[0])
        return torch.cat([y, torch.stack(out, 1)], -1), [fast_state, [hidden], count]

    def _many_rows(self, y, fast_state, hidden, count):
        """Same result as the row loop above: run the slow GRU only on the rows it reads."""
        total = y.shape[1]
        seen = int(count.item())
        rows = [t for t in range(total) if (seen + t + 1) % self.every == 0]
        history = hidden[0].unsqueeze(1)
        if rows:
            slow_out, s = self.slow(y[:, rows], [hidden])
            history, hidden = torch.cat([history, slow_out], 1), s[0]
        which = torch.zeros(total, dtype=torch.long, device=y.device)
        which[rows] = 1
        return (torch.cat([y, history[:, which.cumsum(0)]], -1),
                [fast_state, [hidden], count + total])


# ------------------------------------------------------------ convolutional


class CausalConv(nn.Module):
    """Convolution over the current and earlier rows, keeping past rows as state."""

    def __init__(self, c_in, c_out, kernel, dilation=1, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel, dilation=dilation, groups=groups)
        self.pad = (kernel - 1) * dilation
        self.c_in = c_in

    def forward(self, x, state=None):
        if state is None:
            state = x.new_zeros(x.shape[0], self.pad, self.c_in)
        z = torch.cat([state, x], 1)
        return self.conv(z.transpose(1, 2)).transpose(1, 2), z[:, -self.pad:]


class ConvBlock(nn.Module):
    def __init__(self, channels, kernel, dilation=1):
        super().__init__()
        self.conv = CausalConv(channels, channels, kernel, dilation)

    def forward(self, x, state=None):
        y, s = self.conv(x, state)
        return F.gelu(y), s


class DepthwiseBlock(nn.Module):
    def __init__(self, channels, kernel, dilation=1):
        super().__init__()
        self.conv = CausalConv(channels, channels, kernel, dilation, groups=channels)
        self.mix = nn.Linear(channels, channels)

    def forward(self, x, state=None):
        y, s = self.conv(x, state)
        return self.mix(F.gelu(y)), s


class WaveNet(nn.Module):
    def __init__(self, channels, layers):
        super().__init__()
        self.convs = nn.ModuleList(CausalConv(channels, 2 * channels, 2, 2 ** i) for i in range(layers))
        self.res = nn.ModuleList(nn.Linear(channels, channels) for _ in range(layers))
        self.skip = nn.ModuleList(nn.Linear(channels, channels) for _ in range(layers))

    def forward(self, x, state=None):
        state = _states(state, len(self.convs))
        total = 0
        new = []
        for conv, res, skip, s in zip(self.convs, self.res, self.skip, state):
            y, s = conv(x, s)
            a, b = y.chunk(2, -1)
            gated = torch.tanh(a) * torch.sigmoid(b)
            x = x + res(gated)
            total = total + skip(gated)
            new.append(s)
        return F.gelu(total), new


class MultiKernel(nn.Module):
    """Parallel causal convolutions of different widths, joined together."""

    def __init__(self, c_in, branch, kernels, out):
        super().__init__()
        self.convs = nn.ModuleList(CausalConv(c_in, branch, k) for k in kernels)
        self.mix = nn.Linear(branch * len(kernels), out)

    def forward(self, x, state=None):
        state = _states(state, len(self.convs))
        parts, new = [], []
        for conv, s in zip(self.convs, state):
            y, s = conv(x, s)
            parts.append(y)
            new.append(s)
        return self.mix(F.gelu(torch.cat(parts, -1))), new


class Inception(nn.Module):
    """InceptionTime block made causal: narrowing layer, three widths, max pool."""

    def __init__(self, channels, branch=32, kernels=(9, 19, 39), pool=3):
        super().__init__()
        self.narrow = nn.Linear(channels, branch)
        self.convs = nn.ModuleList(CausalConv(branch, branch, k) for k in kernels)
        self.pool = pool
        self.pool_proj = nn.Linear(channels, branch)
        self.mix = nn.Linear(branch * (len(kernels) + 1), channels)

    def forward(self, x, state=None):
        conv_states = _states(None if state is None else state[0], len(self.convs))
        history = x.new_zeros(x.shape[0], self.pool - 1, x.shape[2]) if state is None else state[1]
        narrow = self.narrow(x)
        parts, new = [], []
        for conv, s in zip(self.convs, conv_states):
            y, s = conv(narrow, s)
            parts.append(y)
            new.append(s)
        z = torch.cat([history, x], 1)
        pooled = F.max_pool1d(z.transpose(1, 2), self.pool, 1).transpose(1, 2)
        parts.append(self.pool_proj(pooled))
        return self.mix(F.gelu(torch.cat(parts, -1))), [new, z[:, -(self.pool - 1):]]


# ---------------------------------------------------------------- attention


class WindowAttention(nn.Module):
    """Multi-head attention over the last `window` rows, caching keys and values."""

    def __init__(self, dim, heads, window):
        super().__init__()
        self.heads, self.head_dim, self.window = heads, dim // heads, window
        self.query = nn.Linear(dim, dim)
        self.key_value = nn.Linear(dim, 2 * dim)
        self.out = nn.Linear(dim, dim)
        self.position = nn.Parameter(torch.zeros(heads, window))
        self.dim = dim

    def _attend(self, q, windows):
        # q: (B, T, dim). windows: (B, T, W, 2 * dim).
        b, t = q.shape[:2]
        k, v = windows.chunk(2, -1)
        q = q.reshape(b, t, 1, self.heads, self.head_dim)
        k = k.reshape(b, t, self.window, self.heads, self.head_dim)
        v = v.reshape(b, t, self.window, self.heads, self.head_dim)
        logits = (q * k).sum(-1) / math.sqrt(self.head_dim) + self.position.T
        weights = logits.softmax(2).unsqueeze(-1)
        return (weights * v).sum(2).reshape(b, t, self.dim)

    def forward(self, x, state=None):
        if state is None:
            state = x.new_zeros(x.shape[0], self.window - 1, 2 * self.dim)
        out = []
        for start in range(0, x.shape[1], ATTENTION_CHUNK):
            part = x[:, start:start + ATTENTION_CHUNK]
            history = torch.cat([state, self.key_value(part)], 1)
            if part.shape[1] == 1:
                windows = history.unsqueeze(1)
            else:
                windows = history.unfold(1, self.window, 1).permute(0, 1, 3, 2)
            out.append(self._attend(self.query(part), windows))
            state = history[:, -(self.window - 1):]
        return self.out(torch.cat(out, 1)), state


class LinearAttention(nn.Module):
    """Causal linear attention (Katharopoulos et al.) as running sums, no window."""

    def __init__(self, dim, heads):
        super().__init__()
        self.heads, self.head_dim, self.dim = heads, dim // heads, dim
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, state=None):
        b = x.shape[0]
        if state is None:
            s_kv = x.new_zeros(b, self.heads, self.head_dim, self.head_dim)
            s_k = x.new_zeros(b, self.heads, self.head_dim)
        else:
            s_kv, s_k = state
        out = []
        for start in range(0, x.shape[1], ATTENTION_CHUNK):
            part = x[:, start:start + ATTENTION_CHUNK]
            t = part.shape[1]
            q, k, v = self.qkv(part).reshape(b, t, 3, self.heads, self.head_dim).unbind(2)
            q, k = F.elu(q) + 1, F.elu(k) + 1
            kv = (s_kv.unsqueeze(1) + torch.cumsum(k.unsqueeze(-2) * v.unsqueeze(-1), 1))
            kk = s_k.unsqueeze(1) + torch.cumsum(k, 1)
            top = (kv * q.unsqueeze(-1)).sum(-2)
            bottom = (kk * q).sum(-1, keepdim=True)
            out.append((top / bottom).reshape(b, t, self.dim))
            s_kv, s_k = kv[:, -1], kk[:, -1]
        return self.out(torch.cat(out, 1)), [s_kv, s_k]


class GRUWithAttention(nn.Module):
    def __init__(self, c_in, hidden, heads, window):
        super().__init__()
        self.gru = TorchRNN('gru', c_in, hidden)
        self.attention = WindowAttention(hidden, heads, window)

    def forward(self, x, state=None):
        gru_state, attention_state = _states(state, 2)
        h, gru_state = self.gru(x, gru_state)
        a, attention_state = self.attention(h, attention_state)
        return torch.cat([h, a], -1), [gru_state, attention_state]


# --------------------------------------------------------------- state-space


class LRU(Cell):
    """Linear Recurrent Unit (Orvieto et al.), complex numbers kept as real pairs."""

    def __init__(self, dim, r_min=0.9, r_max=0.9995):
        super().__init__()
        self.dim = dim
        u = torch.rand(dim)
        self.nu = nn.Parameter(torch.log(-0.5 * torch.log(u * (r_max ** 2 - r_min ** 2) + r_min ** 2)))
        self.theta = nn.Parameter(torch.rand(dim) * math.pi / 5)
        self.b = nn.Linear(dim, 2 * dim, bias=False)
        self.c = nn.Linear(2 * dim, dim, bias=False)
        self.d = nn.Linear(dim, dim, bias=False)

    def zero(self, x):
        return [x.new_zeros(x.shape[0], self.dim), x.new_zeros(x.shape[0], self.dim)]

    def prepare(self):
        size = torch.exp(-torch.exp(self.nu))
        return size * torch.cos(self.theta), size * torch.sin(self.theta), torch.sqrt(1 - size ** 2)

    def step(self, x, s, shared):
        lam_re, lam_im, gamma = shared
        re, im = s
        in_re, in_im = (gamma.repeat(2) * self.b(x)).chunk(2, -1)
        re, im = lam_re * re - lam_im * im + in_re, lam_re * im + lam_im * re + in_im
        return self.c(torch.cat([re, im], -1)) + self.d(x), [re, im]

    def forward(self, x, state=None):
        if x.shape[1] == 1:  # Row by row stays real-valued, so that it exports to ONNX.
            return super().forward(x, state)
        re, im = self.zero(x) if state is None else state
        lam_re, lam_im, gamma = self.prepare()
        in_re, in_im = (gamma.repeat(2) * self.b(x)).chunk(2, -1)
        z = linear_recurrence(torch.complex(in_re, in_im), torch.complex(lam_re, lam_im),
                              torch.complex(re, im), varying=False)
        return self.c(torch.cat([z.real, z.imag], -1)) + self.d(x), [z[:, -1].real, z[:, -1].imag]


class S4D(Cell):
    """Diagonal state-space layer: several decaying oscillators per channel."""

    def __init__(self, dim, states=16):
        super().__init__()
        self.dim, self.states = dim, states
        # Time constants spread from about 1 to about 2,000 rows.
        self.log_tau = nn.Parameter(torch.empty(dim, states).uniform_(0, math.log(2000)))
        self.theta = nn.Parameter(torch.rand(dim, states) * math.pi / 5)
        self.c_re = nn.Parameter(torch.randn(dim, states) / math.sqrt(states))
        self.c_im = nn.Parameter(torch.randn(dim, states) / math.sqrt(states))
        self.glu = nn.Linear(dim, 2 * dim)

    def zero(self, x):
        shape = (x.shape[0], self.dim, self.states)
        return [x.new_zeros(shape), x.new_zeros(shape)]

    def prepare(self):
        size = torch.exp(-torch.exp(-self.log_tau))
        return size * torch.cos(self.theta), size * torch.sin(self.theta), torch.sqrt(1 - size ** 2)

    def step(self, x, s, shared):
        lam_re, lam_im, gamma = shared
        re, im = s
        drive = gamma * x.unsqueeze(-1)
        re, im = lam_re * re - lam_im * im + drive, lam_re * im + lam_im * re
        y = (self.c_re * re - self.c_im * im).sum(-1)
        return F.glu(self.glu(y), -1), [re, im]

    def forward(self, x, state=None):
        """Same result as stepping through rows, using an FFT for the long filter."""
        if x.shape[1] == 1:
            return super().forward(x, state)
        re, im = self.zero(x) if state is None else state
        total = x.shape[1]
        rows = torch.arange(total + 1, dtype=x.dtype, device=x.device)
        decay = torch.exp(-rows * torch.exp(-self.log_tau).unsqueeze(-1))
        angle = rows * self.theta.unsqueeze(-1)
        power_re, power_im = decay * torch.cos(angle), decay * torch.sin(angle)
        gamma = self.prepare()[2].unsqueeze(-1)

        # What each input row adds to the output, k rows later, for each channel.
        kernel = (gamma * (self.c_re.unsqueeze(-1) * power_re[..., :total]
                           - self.c_im.unsqueeze(-1) * power_im[..., :total])).sum(1)
        drive = x.transpose(1, 2)
        y = torch.fft.irfft(torch.fft.rfft(drive, 2 * total) * torch.fft.rfft(kernel, 2 * total),
                            2 * total)[..., :total]
        # What the state from before this chunk adds, as it decays.
        carried_re = self.c_re * re - self.c_im * im
        carried_im = self.c_re * im + self.c_im * re
        y = y + (torch.einsum('bds,dsk->bdk', carried_re, power_re[..., 1:])
                 - torch.einsum('bds,dsk->bdk', carried_im, power_im[..., 1:]))
        # The state after the last row, for the next chunk.
        back_re, back_im = power_re[..., :total].flip(-1), power_im[..., :total].flip(-1)
        new_re = (torch.einsum('bdk,dsk->bds', drive, gamma * back_re)
                  + power_re[..., total] * re - power_im[..., total] * im)
        new_im = (torch.einsum('bdk,dsk->bds', drive, gamma * back_im)
                  + power_re[..., total] * im + power_im[..., total] * re)
        return F.glu(self.glu(y.transpose(1, 2)), -1), [new_re, new_im]


class RWKVMix(nn.Module):
    """RWKV-style time mixing: per-channel decayed average with a learned gate."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.rkv = nn.Linear(dim, 3 * dim)
        self.decay = nn.Parameter(torch.empty(dim).uniform_(-2, 6))
        self.out = nn.Linear(dim, dim)

    def forward(self, x, state=None):
        zeros = x.new_zeros(x.shape[0], self.dim)
        a, b = (zeros, zeros) if state is None else state
        keep = torch.exp(-torch.exp(-self.decay))
        r, k, v = self.rkv(x).chunk(3, -1)
        k = F.softplus(k)
        a = linear_recurrence(k * v, keep, a, varying=False)
        b = linear_recurrence(k, keep, b, varying=False)
        return self.out(torch.sigmoid(r) * a / (b + 1e-6)), [a[:, -1], b[:, -1]]


class RetNetMix(nn.Module):
    """Recurrent retention (Sun et al.): a matrix per head, decayed at its own rate."""

    def __init__(self, dim, heads=4):
        super().__init__()
        self.dim, self.heads, self.head_dim = dim, heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.register_buffer('gamma', torch.tensor([0.9, 0.99, 0.995, 0.999][:heads]).reshape(-1, 1, 1))

    def forward(self, x, state=None):
        batch, rows = x.shape[:2]
        if state is None:
            start = x.new_zeros(batch, self.heads, self.head_dim, self.head_dim)
        else:
            start = state[0]
        q, k, v = self.qkv(x).reshape(batch, rows, 3, self.heads, self.head_dim).unbind(2)
        k = k / math.sqrt(self.head_dim)
        added = (1 - self.gamma) * (k.unsqueeze(-1) * v.unsqueeze(-2))
        retained = linear_recurrence(added, self.gamma, start, varying=False)
        y = (retained @ q.unsqueeze(-1)).squeeze(-1)
        return self.out(y.reshape(batch, rows, self.dim)), [retained[:, -1]]


class MambaSSM(nn.Module):
    """Input-gated diagonal state space, the core recurrence of Mamba."""

    def __init__(self, dim, states=8):
        super().__init__()
        self.dim, self.states = dim, states
        self.log_a = nn.Parameter(torch.empty(dim, states).uniform_(-math.log(2000), 0))
        self.bc = nn.Linear(dim, 2 * states)
        self.delta = nn.Linear(dim, dim)
        self.d = nn.Parameter(torch.ones(dim))

    def forward(self, x, state=None):
        h = x.new_zeros(x.shape[0], self.dim, self.states) if state is None else state[0]
        a = -torch.exp(self.log_a)
        keep = torch.exp(a * F.softplus(self.delta(x)).unsqueeze(-1))
        b, c = self.bc(x).chunk(2, -1)
        added = (keep - 1) / a * b.unsqueeze(2) * x.unsqueeze(-1)
        # Doubling would store too many large tensors here, so this one small loop stays.
        rows = []
        for keep_t, added_t in zip(keep.unbind(1), added.unbind(1)):
            h = keep_t * h + added_t
            rows.append(h)
        y = (c.unsqueeze(2) * torch.stack(rows, 1)).sum(-1)
        return y + self.d * x, [h]


class MambaBlock(nn.Module):
    def __init__(self, dim, expand=2, kernel=4, states=8):
        super().__init__()
        inner = dim * expand
        self.up = nn.Linear(dim, 2 * inner)
        self.conv = CausalConv(inner, inner, kernel, groups=inner)
        self.ssm = MambaSSM(inner, states)
        self.down = nn.Linear(inner, dim)

    def forward(self, x, state=None):
        conv_state, ssm_state = _states(state, 2)
        u, gate = self.up(x).chunk(2, -1)
        u, conv_state = self.conv(u, conv_state)
        y, ssm_state = self.ssm(F.silu(u), ssm_state)
        return self.down(y * F.silu(gate)), [conv_state, ssm_state]


# ------------------------------------------------------------ fixed reservoir


class Reservoir(Cell):
    """Echo state network: a fixed random RNN. Only the output layer is fitted."""

    def __init__(self, c_in, size=512, radius=0.95, leak=0.3, density=0.05, seed=0):
        super().__init__()
        rng = torch.Generator().manual_seed(seed)
        w = torch.randn(size, size, generator=rng)
        w = w * (torch.rand(size, size, generator=rng) < density)
        w = w * radius / torch.linalg.eigvals(w).abs().max()
        self.register_buffer('w', w)
        self.register_buffer('w_in', 0.2 * torch.randn(c_in, size, generator=rng))
        self.size, self.leak = size, leak

    def zero(self, x):
        return [x.new_zeros(x.shape[0], self.size)]

    def step(self, x, s, shared):
        h = (1 - self.leak) * s[0] + self.leak * torch.tanh(s[0] @ self.w + x @ self.w_in)
        return torch.cat([x, h], -1), [h]


# ---------------------------------------------------------- feature grouping


class GroupEncoder(nn.Module):
    """Encode column groups separately, then join them."""

    def __init__(self, groups, width):
        super().__init__()
        self.groups = groups
        self.encoders = nn.ModuleList(
            nn.Sequential(nn.Linear(len(g), width), nn.GELU(), nn.Linear(width, width)) for g in groups)

    def forward(self, x, state=None):
        return torch.cat([e(x[..., g]) for e, g in zip(self.encoders, self.groups)], -1), None


def _span(start, stop):
    return list(range(start, stop))


INSTRUMENT_GROUPS = [_span(0, 52), _span(52, 104), _span(104, 112)]
FEATURE_GROUPS = [_span(0, 44), _span(44, 52), _span(52, 96), _span(96, 104), _span(104, 112)]


# ------------------------------------------------------------- whole models


class SeqModel(nn.Module):
    """Standardize inputs, run the body, predict the two targets."""

    def __init__(self, body, out):
        super().__init__()
        self.register_buffer('mean', torch.zeros(N_FEATURES))
        self.register_buffer('std', torch.ones(N_FEATURES))
        self.body = body
        self.head = nn.Linear(out, 2)

    def set_scaling(self, mean, std):
        self.mean.copy_(mean)
        self.std.copy_(std.clamp_min(1e-6))

    def features(self, x, state=None):
        return self.body(torch.clamp((x - self.mean) / self.std, -10, 10), state)

    def forward(self, x, state=None):
        h, state = self.features(x, state)
        return self.head(h), state


def tcn(channels, kernel, layers, block=ConvBlock):
    return [Residual(channels, block(channels, kernel, 2 ** i)) for i in range(layers)]


def cnn(channels, kernel, layers):
    return [Residual(channels, ConvBlock(channels, kernel)) for _ in range(layers)]


def transformer(window, dim=64, heads=4, layers=2):
    blocks = [linear(N_FEATURES, dim)]
    for _ in range(layers):
        blocks += [Residual(dim, WindowAttention(dim, heads, window)), res_ff(dim)]
    return Chain(*blocks), dim


def mixer(cell, dim, layers):
    """Input projection, then repeated recurrence and feedforward blocks."""
    blocks = [linear(N_FEATURES, dim)]
    for _ in range(layers):
        blocks += [Residual(dim, cell()), res_ff(dim)]
    return Chain(*blocks), dim


def _gru_after(blocks, channels, rnn='gru', hidden=96):
    return Chain(linear(N_FEATURES, channels), *blocks, TorchRNN(rnn, channels, hidden)), hidden


def _build_two_speed():
    return Chain(TwoSpeedGRU(N_FEATURES, 128, 64)), 192


@dataclass(frozen=True)
class Spec:
    id: int
    key: str
    name: str
    build: Callable
    source: str = 'user'
    fit: str = 'gradient'


SPECS = [Spec(*row) for row in [
    (1, 'gru2', 'Two-layer GRU matching the baseline',
     lambda: (TorchRNN('gru', N_FEATURES, 128, 2), 128)),
    (2, 'rnn', 'Vanilla RNN', lambda: (TorchRNN('rnn', N_FEATURES, 256, 2), 256)),
    (3, 'lstm', 'LSTM', lambda: (TorchRNN('lstm', N_FEATURES, 128, 2), 128)),
    (4, 'plstm', 'Projected LSTM', lambda: (TorchRNN('lstm', N_FEATURES, 192, 2, proj=96), 96)),
    (5, 'mgu', 'Minimal gated unit', lambda: (Chain(MGU(N_FEATURES, 128), MGU(128, 128)), 128)),
    (6, 'sru', 'Simple recurrent unit',
     lambda: (Chain(linear(N_FEATURES, 128), SRU(128), SRU(128), SRU(128)), 128)),
    (7, 'qrnn', 'Quasi-recurrent network', lambda: (Chain(QRNN(N_FEATURES, 128), QRNN(128, 128)), 128)),
    (8, 'resgru', 'Residual GRU',
     lambda: (Chain(linear(N_FEATURES, 128), *[Residual(128, TorchRNN('gru', 128, 128)) for _ in range(2)]), 128)),
    (9, 'cnn', 'Causal temporal CNN', lambda: (Chain(linear(N_FEATURES, 64), *cnn(64, 5, 4)), 64)),
    (10, 'tcn', 'Dilated TCN', lambda: (Chain(linear(N_FEATURES, 64), *tcn(64, 3, 6)), 64)),
    (11, 'wavenet', 'Gated WaveNet-style', lambda: (Chain(linear(N_FEATURES, 64), WaveNet(64, 8)), 64)),
    (12, 'dwtcn', 'Depthwise TCN',
     lambda: (Chain(linear(N_FEATURES, 96), *tcn(96, 3, 7, DepthwiseBlock)), 96)),
    (13, 'multiscale', 'Multiscale temporal CNN',
     lambda: (Chain(linear(N_FEATURES, 96),
                    *[Residual(96, MultiKernel(96, 32, (3, 9, 27), 96)) for _ in range(2)]), 96)),
    (14, 'inception', 'Causal InceptionTime',
     lambda: (Chain(linear(N_FEATURES, 128), *[Residual(128, Inception(128)) for _ in range(3)]), 128)),
    (15, 'cnn_gru', 'Temporal CNN then GRU', lambda: _gru_after(cnn(64, 5, 2), 64)),
    (16, 'cnn_lstm', 'Temporal CNN then LSTM', lambda: _gru_after(cnn(64, 5, 2), 64, 'lstm')),
    (17, 'tcn_gru', 'TCN then GRU', lambda: _gru_after(tcn(64, 3, 4), 64)),
    (18, 'gru_attn', 'GRU with causal attention',
     lambda: (Chain(GRUWithAttention(N_FEATURES, 128, 4, 32)), 256)),
    (19, 'transformer', 'Tiny causal Transformer', lambda: transformer(64)),
    (20, 'local_tf', 'Local-window Transformer', lambda: transformer(16)),
    (21, 'linear_tf', 'Linear-attention Transformer',
     lambda: mixer(lambda: LinearAttention(64, 4), 64, 2)),
    (22, 's4d', 'Diagonal state-space (S4D)', lambda: mixer(lambda: S4D(128), 128, 4)),
    (23, 'rwkv', 'RWKV-style network', lambda: mixer(lambda: RWKVMix(96), 96, 2)),
    (24, 'retnet', 'Recurrent RetNet', lambda: mixer(lambda: RetNetMix(64), 64, 2)),
    (25, 'mamba', 'Small Mamba-style', lambda: mixer(lambda: MambaBlock(64), 64, 2)),
    (26, 'instr_gru', 'Separate i0 and i1 encoders, then GRU',
     lambda: (Chain(GroupEncoder(INSTRUMENT_GROUPS, 48), TorchRNN('gru', 144, 128)), 128), 'suggested'),
    (27, 'group_gru', 'Separate price, volume, trade and a* encoders, then GRU',
     lambda: (Chain(GroupEncoder(FEATURE_GROUPS, 32), TorchRNN('gru', 160, 128)), 128), 'suggested'),
    (28, 'twospeed', 'Two-speed GRU', _build_two_speed, 'suggested'),
    (29, 'lru', 'Linear Recurrent Unit (LRU)', lambda: mixer(lambda: LRU(128), 128, 2), 'suggested'),
    (30, 'esn', 'Echo state network', lambda: (Chain(Reservoir(N_FEATURES)), N_FEATURES + 512),
     'suggested', 'ridge'),
]]
# Model 31 (moving averages then LightGBM) is not a torch model; see gbm.py.
REFERENCES = [
    Spec(-2, 'constant', 'Constant prediction', lambda: (Chain(), N_FEATURES), 'reference', 'constant'),
    Spec(-1, 'ridge', 'Ridge on the current row', lambda: (Chain(), N_FEATURES), 'reference', 'ridge'),
]
BY_KEY = {spec.key: spec for spec in [*SPECS, *REFERENCES]}


def build(key: str) -> SeqModel:
    body, out = BY_KEY[key].build()
    return SeqModel(body, out)


def flatten_state(state) -> list[torch.Tensor]:
    if state is None:
        return []
    if isinstance(state, torch.Tensor):
        return [state]
    return [tensor for part in state for tensor in flatten_state(part)]


def unflatten_state(template, tensors):
    """Rebuild a nested state from a flat list, following a template's shape."""
    iterator = iter(tensors)

    def rebuild(node):
        if node is None:
            return None
        if isinstance(node, torch.Tensor):
            return next(iterator)
        return [rebuild(part) for part in node]

    return rebuild(template)


def detach_state(state):
    if state is None:
        return None
    if isinstance(state, torch.Tensor):
        return state.detach()
    return [detach_state(part) for part in state]
