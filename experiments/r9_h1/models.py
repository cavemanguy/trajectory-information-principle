from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.r9_t1 import run_r9_t1 as base

N_VAL = base.N_VAL
SEQ_LEN = base.SEQ_LEN


def causal_mask(T: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)


def alter_code(code: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "native":
        return code
    if mode in ("zero", "off", "removed"):
        return torch.zeros_like(code)
    if mode == "shuffled":
        return torch.roll(code, shifts=7, dims=1)
    if mode in ("wrong", "wrong_context"):
        return torch.roll(code, shifts=1, dims=0)
    if mode in ("random", "matched_random"):
        g = torch.Generator(device=code.device)
        g.manual_seed(9173)
        z = torch.randn(code.shape, generator=g, device=code.device, dtype=code.dtype)
        zn = torch.linalg.vector_norm(z, dim=-1, keepdim=True).clamp_min(1e-6)
        cn = torch.linalg.vector_norm(code.detach(), dim=-1, keepdim=True)
        return z / zn * cn
    raise ValueError(f"unknown control mode: {mode}")


class CausalStack(nn.Module):
    def __init__(self, dim=48, layers=2, heads=4, ff=96):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=ff,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.stack = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        return self.norm(self.stack(x, mask=causal_mask(T, x.device)))


class BaseArm(nn.Module):
    probe_streams: Tuple[str, ...] = ()
    pair_streams: Tuple[str, str] | None = None
    controller_signal: str | None = None
    control_capable: bool = False

    def forward(self, types, payload, return_states=False, control_mode="native"):
        raise NotImplementedError


class ControlRNN(BaseArm):
    probe_streams = ("RNN",)

    def __init__(self, embed=32, hidden=64):
        super().__init__()
        self.encoder = base.TokenEncoder(embed)
        self.cell = base.GRRCell(embed, hidden)
        self.head = nn.Linear(hidden, N_VAL)
        self.hidden = hidden

    def forward(self, types, payload, return_states=False, control_mode="native"):
        x = self.encoder(types, payload)
        h = torch.zeros(types.shape[0], self.hidden, device=types.device)
        hs = []
        for t in range(types.shape[1]):
            h = self.cell(x[:, t], h)
            hs.append(h)
        hs = torch.stack(hs, 1)
        logits = self.head(hs)
        if return_states:
            return logits, {"RNN": hs}
        return logits


class ControlTransformer(BaseArm):
    probe_streams = ("T",)

    def __init__(self, dim=64):
        super().__init__()
        self.encoder = base.TokenEncoder(dim)
        self.pos = nn.Parameter(torch.randn(SEQ_LEN, dim) * 0.01)
        self.stack = CausalStack(dim=dim, layers=2, heads=4, ff=128)
        self.head = nn.Linear(dim, N_VAL)

    def forward(self, types, payload, return_states=False, control_mode="native"):
        T = types.shape[1]
        h = self.stack(self.encoder(types, payload) + self.pos[:T])
        logits = self.head(h)
        if return_states:
            return logits, {"T": h}
        return logits


class RTRModel(BaseArm):
    probe_streams = ("RNN1", "T", "RNN2")
    pair_streams = ("RNN1", "T")

    def __init__(self, dim=48):
        super().__init__()
        self.encoder = base.TokenEncoder(dim)
        self.front = base.GRRCell(dim, dim)
        self.pos = nn.Parameter(torch.randn(SEQ_LEN, dim) * 0.01)
        self.stack = CausalStack(dim=dim, layers=2, heads=4, ff=96)
        self.back = base.GRRCell(dim, dim)
        self.head = nn.Linear(dim, N_VAL)

    def forward(self, types, payload, return_states=False, control_mode="native"):
        x = self.encoder(types, payload)
        B, T, D = x.shape
        h = torch.zeros(B, D, device=x.device)
        r1 = []
        for t in range(T):
            h = self.front(x[:, t], h)
            r1.append(h)
        r1 = torch.stack(r1, 1)
        tr = self.stack(r1 + self.pos[:T])
        h = torch.zeros(B, D, device=x.device)
        r2 = []
        for t in range(T):
            h = self.back(tr[:, t], h)
            r2.append(h)
        r2 = torch.stack(r2, 1)
        logits = self.head(r2)
        if return_states:
            return logits, {"RNN1": r1, "T": tr, "RNN2": r2}
        return logits


class TRTModel(BaseArm):
    probe_streams = ("T1", "RNN", "T2")
    pair_streams = ("T1", "RNN")

    def __init__(self, dim=48):
        super().__init__()
        self.encoder = base.TokenEncoder(dim)
        self.pos1 = nn.Parameter(torch.randn(SEQ_LEN, dim) * 0.01)
        self.t1 = CausalStack(dim=dim, layers=1, heads=4, ff=96)
        self.rnn = base.GRRCell(dim, dim)
        self.pos2 = nn.Parameter(torch.randn(SEQ_LEN, dim) * 0.01)
        self.t2 = CausalStack(dim=dim, layers=1, heads=4, ff=96)
        self.head = nn.Linear(dim, N_VAL)

    def forward(self, types, payload, return_states=False, control_mode="native"):
        x = self.encoder(types, payload)
        B, T, D = x.shape
        t1 = self.t1(x + self.pos1[:T])
        h = torch.zeros(B, D, device=x.device)
        rs = []
        for t in range(T):
            h = self.rnn(t1[:, t], h)
            rs.append(h)
        rs = torch.stack(rs, 1)
        t2 = self.t2(rs + self.pos2[:T])
        logits = self.head(t2)
        if return_states:
            return logits, {"T1": t1, "RNN": rs, "T2": t2}
        return logits


class ParallelModel(BaseArm):
    probe_streams = ("RNN", "T", "FUSED")
    pair_streams = ("RNN", "T")

    def __init__(self, dim=48):
        super().__init__()
        self.encoder = base.TokenEncoder(dim)
        self.rnn = base.GRRCell(dim, dim)
        self.pos = nn.Parameter(torch.randn(SEQ_LEN, dim) * 0.01)
        self.tr = CausalStack(dim=dim, layers=2, heads=4, ff=96)
        self.fuse = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.LayerNorm(dim))
        self.head = nn.Linear(dim, N_VAL)

    def forward(self, types, payload, return_states=False, control_mode="native"):
        x = self.encoder(types, payload)
        B, T, D = x.shape
        h = torch.zeros(B, D, device=x.device)
        rs = []
        for t in range(T):
            h = self.rnn(x[:, t], h)
            rs.append(h)
        rs = torch.stack(rs, 1)
        tr = self.tr(x + self.pos[:T])
        fused = self.fuse(torch.cat([rs, tr], dim=-1))
        logits = self.head(fused)
        if return_states:
            return logits, {"RNN": rs, "T": tr, "FUSED": fused}
        return logits


class TeacherModel(BaseArm):
    probe_streams = ("BASE_RNN", "TEACHER", "RNN")
    pair_streams = ("BASE_RNN", "TEACHER")
    controller_signal = "P_CODE"
    control_capable = True

    def __init__(self, embed=32, hidden=64, tdim=48, code_dim=8):
        super().__init__()
        self.encoder = base.TokenEncoder(embed)
        self.cell = base.GRRCell(embed, hidden)
        self.base_proj = nn.Linear(hidden, tdim)
        self.x_proj = nn.Linear(embed, tdim)
        self.pos = nn.Parameter(torch.randn(SEQ_LEN, tdim) * 0.01)
        self.teacher = CausalStack(dim=tdim, layers=2, heads=4, ff=96)
        self.to_code = nn.Linear(tdim, code_dim)
        self.code_to_hidden = nn.Linear(code_dim, hidden, bias=False)
        self.out_norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, N_VAL)
        self.hidden = hidden

    def _base(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden, device=x.device)
        hs = []
        for t in range(T):
            h = self.cell(x[:, t], h)
            hs.append(h)
        return torch.stack(hs, 1)

    def forward(self, types, payload, return_states=False, control_mode="native"):
        x = self.encoder(types, payload)
        base_h = self._base(x)
        T = x.shape[1]
        teach = self.teacher(self.x_proj(x) + self.base_proj(base_h) + self.pos[:T])
        code_native = 0.35 * torch.tanh(self.to_code(teach))
        code = alter_code(code_native, control_mode)
        p = 0.10 * torch.tanh(self.code_to_hidden(code))
        B = x.shape[0]
        h = torch.zeros(B, self.hidden, device=x.device)
        hs = []
        for t in range(T):
            h = self.cell(x[:, t], h)
            h = self.out_norm(h + p[:, t])
            hs.append(h)
        hs = torch.stack(hs, 1)
        logits = self.head(hs)
        if return_states:
            return logits, {"BASE_RNN": base_h, "TEACHER": teach, "RNN": hs, "P_CODE": code_native}
        return logits


class InterrogatorModel(BaseArm):
    probe_streams = ("RNN", "TCTRL", "RESPONSE")
    pair_streams = ("RNN", "RESPONSE")
    controller_signal = "P_CODE"
    control_capable = True

    def __init__(self, embed=32, hidden=64, tdim=48, code_dim=8):
        super().__init__()
        self.encoder = base.TokenEncoder(embed)
        self.cell = base.GRRCell(embed, hidden)
        self.hidden = hidden
        self.x_proj = nn.Linear(embed, tdim)
        self.h_proj = nn.Linear(hidden, tdim)
        self.pos = nn.Parameter(torch.randn(SEQ_LEN, tdim) * 0.01)
        self.ctrl = CausalStack(dim=tdim, layers=2, heads=4, ff=96)
        self.to_code = nn.Linear(tdim, code_dim)
        self.code_to_hidden = nn.Linear(code_dim, hidden, bias=False)
        self.readout = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, N_VAL))

    def forward(self, types, payload, return_states=False, control_mode="native"):
        x = self.encoder(types, payload)
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden, device=x.device)
        hs = []
        prevs = []
        for t in range(T):
            prevs.append(h)
            h = self.cell(x[:, t], h)
            hs.append(h)
        hs = torch.stack(hs, 1)
        prevs = torch.stack(prevs, 1)
        ctrl = self.ctrl(self.x_proj(x) + self.h_proj(hs) + self.pos[:T])
        code_native = 0.35 * torch.tanh(self.to_code(ctrl))
        code = alter_code(code_native, control_mode)
        p = 0.10 * torch.tanh(self.code_to_hidden(code))
        resp = []
        for t in range(T):
            probe_h = self.cell(x[:, t], prevs[:, t] + p[:, t])
            resp.append(probe_h - hs[:, t])
        resp = torch.stack(resp, 1)
        logits = self.readout(torch.cat([hs, resp], dim=-1))
        if return_states:
            return logits, {"RNN": hs, "TCTRL": ctrl, "RESPONSE": resp, "P_CODE": code_native}
        return logits


class BiasBlock(nn.Module):
    def __init__(self, dim=48, heads=4, ff=96):
        super().__init__()
        self.heads = heads
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, ff), nn.GELU(), nn.Linear(ff, dim))

    def forward(self, x, alpha):
        B, T, _ = x.shape
        q = self.n1(x)
        i = torch.arange(T, device=x.device).view(T, 1)
        j = torch.arange(T, device=x.device).view(1, T)
        dist = (i - j).clamp_min(0).float() / max(1, T - 1)
        bias = -alpha.unsqueeze(-1) * dist.unsqueeze(0)
        future = j > i
        bias = bias.masked_fill(future.unsqueeze(0), -1e4)
        mask = bias.repeat_interleave(self.heads, dim=0)
        y, _ = self.attn(q, q, q, attn_mask=mask, need_weights=False)
        x = x + y
        x = x + self.ff(self.n2(x))
        return x


class AttentionControllerModel(BaseArm):
    probe_streams = ("RNN", "T")
    pair_streams = ("RNN", "T")
    controller_signal = "ATTN_CODE"
    control_capable = True

    def __init__(self, dim=48):
        super().__init__()
        self.encoder = base.TokenEncoder(dim)
        self.rnn = base.GRRCell(dim, dim)
        self.pos = nn.Parameter(torch.randn(SEQ_LEN, dim) * 0.01)
        self.to_alpha = nn.Linear(dim, 1)
        self.blocks = nn.ModuleList([BiasBlock(dim, 4, 96), BiasBlock(dim, 4, 96)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, N_VAL)

    def forward(self, types, payload, return_states=False, control_mode="native"):
        x = self.encoder(types, payload)
        B, T, D = x.shape
        h = torch.zeros(B, D, device=x.device)
        rs = []
        for t in range(T):
            h = self.rnn(x[:, t], h)
            rs.append(h)
        rs = torch.stack(rs, 1)
        code_native = 1.5 * torch.tanh(self.to_alpha(rs))
        code = alter_code(code_native, control_mode)
        z = x + self.pos[:T]
        for block in self.blocks:
            z = block(z, code.squeeze(-1))
        z = self.norm(z)
        logits = self.head(z)
        if return_states:
            return logits, {"RNN": rs, "T": z, "ATTN_CODE": code_native}
        return logits


class LowRankModModel(BaseArm):
    probe_streams = ("RNN", "TBASE", "T")
    pair_streams = ("RNN", "TBASE")
    controller_signal = "MOD_CODE"
    control_capable = True

    def __init__(self, dim=48, rank=8):
        super().__init__()
        self.encoder = base.TokenEncoder(dim)
        self.rnn = base.GRRCell(dim, dim)
        self.pos = nn.Parameter(torch.randn(SEQ_LEN, dim) * 0.01)
        self.tr = CausalStack(dim=dim, layers=2, heads=4, ff=96)
        self.to_code = nn.Linear(dim, rank)
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, N_VAL)

    def forward(self, types, payload, return_states=False, control_mode="native"):
        x = self.encoder(types, payload)
        B, T, D = x.shape
        h = torch.zeros(B, D, device=x.device)
        rs = []
        for t in range(T):
            h = self.rnn(x[:, t], h)
            rs.append(h)
        rs = torch.stack(rs, 1)
        tbase = self.tr(x + self.pos[:T])
        code_native = 0.5 * torch.tanh(self.to_code(rs))
        code = alter_code(code_native, control_mode)
        adapted = self.up(self.down(tbase) * code)
        z = self.norm(tbase + adapted)
        logits = self.head(z)
        if return_states:
            return logits, {"RNN": rs, "TBASE": tbase, "T": z, "MOD_CODE": code_native}
        return logits


class UpdateGateModel(BaseArm):
    probe_streams = ("RNN", "T")
    pair_streams = ("RNN", "T")
    controller_signal = "UPDATE_WEIGHT"

    def __init__(self, dim=48):
        super().__init__()
        self.encoder = base.TokenEncoder(dim)
        self.rnn = base.GRRCell(dim, dim)
        self.to_weight = nn.Linear(dim, 1)
        self.pos = nn.Parameter(torch.randn(SEQ_LEN, dim) * 0.01)
        self.tr = CausalStack(dim=dim, layers=2, heads=4, ff=96)
        self.head = nn.Linear(dim, N_VAL)

    def forward(self, types, payload, return_states=False, control_mode="native"):
        x = self.encoder(types, payload)
        B, T, D = x.shape
        h = torch.zeros(B, D, device=x.device)
        rs = []
        for t in range(T):
            h = self.rnn(x[:, t], h)
            rs.append(h)
        rs = torch.stack(rs, 1)
        w = 0.5 + torch.sigmoid(self.to_weight(rs))
        tr = self.tr(x + self.pos[:T])
        logits = self.head(tr)
        if return_states:
            return logits, {"RNN": rs, "T": tr, "UPDATE_WEIGHT": w}
        return logits

    def training_loss(self, types, payload, target):
        logits, states = self.forward(types, payload, return_states=True)
        mask = target >= 0
        safe = target.clamp_min(0)
        ce = F.cross_entropy(logits.reshape(-1, N_VAL), safe.reshape(-1), reduction="none").reshape_as(target)
        w = states["UPDATE_WEIGHT"].squeeze(-1)
        wm = w[mask]
        wm = wm / wm.detach().mean().clamp_min(1e-6)
        loss = (ce[mask] * wm).mean()
        return loss, states

    def transformer_parameters(self):
        for module in (self.tr, self.head, self.encoder):
            yield from module.parameters()


class MutualTeachingModel(BaseArm):
    probe_streams = ("RNN", "T")
    pair_streams = ("RNN", "T")

    def __init__(self, dim=48, teach_dim=16):
        super().__init__()
        self.encoder = base.TokenEncoder(dim)
        self.rnn = base.GRRCell(dim, dim)
        self.pos = nn.Parameter(torch.randn(SEQ_LEN, dim) * 0.01)
        self.tr = CausalStack(dim=dim, layers=2, heads=4, ff=96)
        self.r_head = nn.Linear(dim, N_VAL)
        self.t_head = nn.Linear(dim, N_VAL)
        self.r_proj = nn.Linear(dim, teach_dim)
        self.t_proj = nn.Linear(dim, teach_dim)

    def _states(self, types, payload):
        x = self.encoder(types, payload)
        B, T, D = x.shape
        h = torch.zeros(B, D, device=x.device)
        rs = []
        for t in range(T):
            h = self.rnn(x[:, t], h)
            rs.append(h)
        rs = torch.stack(rs, 1)
        tr = self.tr(x + self.pos[:T])
        return rs, tr

    def forward(self, types, payload, return_states=False, control_mode="native"):
        rs, tr = self._states(types, payload)
        lr = self.r_head(rs)
        lt = self.t_head(tr)
        logits = 0.5 * (lr + lt)
        if return_states:
            return logits, {"RNN": rs, "T": tr, "R_TEACH": self.r_proj(rs), "T_TEACH": self.t_proj(tr)}
        return logits

    def training_loss(self, types, payload, target):
        rs, tr = self._states(types, payload)
        lr = self.r_head(rs)
        lt = self.t_head(tr)
        mask = target >= 0
        task = 0.5 * (F.cross_entropy(lr[mask], target[mask]) + F.cross_entropy(lt[mask], target[mask]))
        valid = (types != base.PAD).unsqueeze(-1)
        zr = F.normalize(self.r_proj(rs), dim=-1)
        zt = F.normalize(self.t_proj(tr), dim=-1)
        teach = 0.5 * (
            ((zr - zt.detach()) ** 2)[valid.expand_as(zr)].mean()
            + ((zt - zr.detach()) ** 2)[valid.expand_as(zt)].mean()
        )
        return task + 0.20 * teach, {"RNN": rs, "T": tr, "R_TEACH": zr, "T_TEACH": zt}


def build_arms(seed_fn, seed: int) -> Dict[str, BaseArm]:
    ctors = [
        ("CONTROL_RNN", ControlRNN),
        ("CONTROL_T", ControlTransformer),
        ("RTR", RTRModel),
        ("TRT", TRTModel),
        ("PARALLEL", ParallelModel),
        ("TEACHER", TeacherModel),
        ("INTERROGATOR", InterrogatorModel),
        ("ATTN_CONTROLLER", AttentionControllerModel),
        ("LOWRANK_MOD", LowRankModModel),
        ("UPDATE_GATE", UpdateGateModel),
        ("MUTUAL", MutualTeachingModel),
    ]
    out = {}
    for name, ctor in ctors:
        torch.manual_seed(seed_fn(seed, f"init_{name}"))
        out[name] = ctor()
    return out


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
