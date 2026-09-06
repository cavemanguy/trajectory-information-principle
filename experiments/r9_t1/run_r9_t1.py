import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEEDS = (2111, 2131, 2153, 2179, 2203, 2221, 2243, 2267)
N_VAL = 16
N_REG = 3
SEQ_LEN = 96
BLOCK_DATA = 10
BLOCK_DISTRACT = 3
TRAIN_STEPS = 1000
BATCH = 48
EVAL_EPISODES = 1024
PROBE_EPISODES = 512
DYN_EPISODES = 512
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
RIDGE = 1e-2
RHOS = (0.05, 0.10, 0.20)
N_TRANS = 4
DYN_MAX_POINTS = 4096
LONG_GAP = 24
EPS = 1e-8

PAD, REGIME, KEY, ACTIVATE, DATA, DISTRACTOR = range(6)
N_TYPES = 6
NO_PAYLOAD = 16


def derive_seed(seed, name):
    h = hashlib.sha256(f"r9-t1|{seed}|{name}".encode()).digest()
    return int.from_bytes(h[:8], "little") % (2**31 - 1)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def make_episode(rng):
    keys = rng.integers(0, N_VAL, size=N_REG, dtype=np.int64)
    events = []
    targets = []
    active_keys = []
    new_early = []
    return_early = []
    steady = []
    long_gap = []
    active = None
    key_time = [-10_000] * N_REG

    def add(typ, payload=NO_PAYLOAD, target=-100, ak=-100,
            ne=False, re=False, st=False, lg=False):
        events.append((typ, int(payload)))
        targets.append(int(target))
        active_keys.append(int(ak))
        new_early.append(bool(ne))
        return_early.append(bool(re))
        steady.append(bool(st))
        long_gap.append(bool(lg))

    def block(kind):
        # Exactly 10 DATA and 3 DISTRACTOR events, shuffled deterministically.
        pattern = np.array([1] * BLOCK_DATA + [0] * BLOCK_DISTRACT, dtype=np.int64)
        rng.shuffle(pattern)
        seen_data = 0
        for flag in pattern:
            if flag == 0:
                add(DISTRACTOR)
                continue
            x = int(rng.integers(0, N_VAL))
            k = int(keys[active])
            y = (x + k) % N_VAL
            early = seen_data < 4
            gap = len(events) - key_time[active]
            add(
                DATA, x, y, k,
                ne=(kind == "new" and early),
                re=(kind == "return" and early),
                st=(not early),
                lg=(gap >= LONG_GAP),
            )
            seen_data += 1

    schedule = (("define", 0), ("define", 1), ("return", 0),
                ("define", 2), ("return", 1), ("return", 2))
    for kind, r in schedule:
        if kind == "define":
            add(REGIME, r)
            add(KEY, keys[r])
            active = r
            key_time[r] = len(events) - 1
            block("new")
        else:
            add(ACTIVATE, r)
            active = r
            block("return")

    if len(events) > SEQ_LEN:
        raise RuntimeError(f"episode length {len(events)} exceeds {SEQ_LEN}")
    while len(events) < SEQ_LEN:
        add(PAD)

    types = np.asarray([x[0] for x in events], dtype=np.int64)
    payload = np.asarray([x[1] for x in events], dtype=np.int64)
    return {
        "types": types,
        "payload": payload,
        "target": np.asarray(targets, dtype=np.int64),
        "active_key": np.asarray(active_keys, dtype=np.int64),
        "new_early": np.asarray(new_early, dtype=np.bool_),
        "return_early": np.asarray(return_early, dtype=np.bool_),
        "steady": np.asarray(steady, dtype=np.bool_),
        "long_gap": np.asarray(long_gap, dtype=np.bool_),
    }


def make_batch(seed, n):
    rng = np.random.default_rng(int(seed))
    eps = [make_episode(rng) for _ in range(n)]
    out = {}
    for k in eps[0]:
        out[k] = torch.from_numpy(np.stack([e[k] for e in eps], axis=0))
    return out


class TokenEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.type_emb = nn.Embedding(N_TYPES, dim)
        self.payload_emb = nn.Embedding(N_VAL + 1, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, types, payload):
        return self.norm(self.type_emb(types) + self.payload_emb(payload))


class GRRCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pre = nn.LayerNorm(hidden_dim)
        self.cand = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.post = nn.LayerNorm(hidden_dim)

    def forward(self, x, h):
        z = torch.cat([x, self.pre(h)], dim=-1)
        c = torch.tanh(self.cand(z))
        g = torch.sigmoid(self.gate(z))
        return self.post(h + g * (c - h))


class GRRModel(nn.Module):
    def __init__(self, embed=32, hidden=64):
        super().__init__()
        self.encoder = TokenEncoder(embed)
        self.cell = GRRCell(embed, hidden)
        self.head = nn.Linear(hidden, N_VAL)
        self.hidden_dim = hidden

    def forward(self, types, payload, return_states=False):
        x = self.encoder(types, payload)
        h = torch.zeros(types.shape[0], self.hidden_dim, device=types.device)
        states = []
        for t in range(types.shape[1]):
            h = self.cell(x[:, t], h)
            states.append(h)
        hs = torch.stack(states, dim=1)
        logits = self.head(hs)
        return (logits, hs) if return_states else logits

    def step_from_embedding(self, emb, h):
        return self.cell(emb, h)


class GRUModel(nn.Module):
    def __init__(self, embed=32, hidden=64):
        super().__init__()
        self.encoder = TokenEncoder(embed)
        self.rnn = nn.GRU(embed, hidden, batch_first=True)
        self.head = nn.Linear(hidden, N_VAL)

    def forward(self, types, payload, return_states=False):
        x = self.encoder(types, payload)
        hs, _ = self.rnn(x)
        logits = self.head(hs)
        return (logits, hs) if return_states else logits


class TransformerModel(nn.Module):
    def __init__(self, dim=64, layers=2, heads=4, ff=128):
        super().__init__()
        self.encoder = TokenEncoder(dim)
        self.pos = nn.Parameter(torch.randn(SEQ_LEN, dim) * 0.01)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=ff,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, N_VAL)

    def forward(self, types, payload, return_states=False):
        T = types.shape[1]
        x = self.encoder(types, payload) + self.pos[:T]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=types.device), diagonal=1)
        hs = self.norm(self.tr(x, mask=mask))
        logits = self.head(hs)
        return (logits, hs) if return_states else logits


class SandwichModel(nn.Module):
    def __init__(self, dim=48):
        super().__init__()
        self.encoder = TokenEncoder(dim)
        self.front = GRRCell(dim, dim)
        self.pos = nn.Parameter(torch.randn(SEQ_LEN, dim) * 0.01)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=4, dim_feedforward=96,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=2)
        self.mid_norm = nn.LayerNorm(dim)
        self.back = GRRCell(dim, dim)
        self.head = nn.Linear(dim, N_VAL)

    def forward(self, types, payload, return_states=False):
        x = self.encoder(types, payload)
        B, T, D = x.shape
        hf = torch.zeros(B, D, device=x.device)
        front = []
        for t in range(T):
            hf = self.front(x[:, t], hf)
            front.append(hf)
        front = torch.stack(front, 1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        mid = self.mid_norm(self.tr(front + self.pos[:T], mask=mask))
        hb = torch.zeros(B, D, device=x.device)
        back = []
        for t in range(T):
            hb = self.back(mid[:, t], hb)
            back.append(hb)
        hs = torch.stack(back, 1)
        logits = self.head(hs)
        return (logits, hs) if return_states else logits


def build_models(seed):
    models = {}
    for i, (name, ctor) in enumerate([
        ("GRR", GRRModel),
        ("GRU", GRUModel),
        ("TRANSFORMER", TransformerModel),
        ("SANDWICH", SandwichModel),
    ]):
        torch.manual_seed(derive_seed(seed, f"init_{name}"))
        models[name] = ctor()
    return models


def count_params(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def masked_loss(logits, target):
    mask = target >= 0
    return F.cross_entropy(logits[mask], target[mask])


def train_model(model, seed, name, smoke=False):
    steps = 12 if smoke else TRAIN_STEPS
    batch_n = 16 if smoke else BATCH
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    model.train()
    losses = []
    for step in range(steps):
        b = make_batch(derive_seed(seed, f"train_batch_{step}"), batch_n)
        opt.zero_grad(set_to_none=True)
        logits = model(b["types"], b["payload"])
        loss = masked_loss(logits, b["target"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            losses.append([int(step), float(loss.detach())])
    return losses


def acc_on_mask(pred, target, mask):
    m = mask & (target >= 0)
    n = int(m.sum())
    if n == 0:
        return float("nan"), 0
    return float((pred[m] == target[m]).float().mean()), n


def evaluate_model(model, seed, name, smoke=False):
    n = 64 if smoke else EVAL_EPISODES
    b = make_batch(derive_seed(seed, "test_stream"), n)
    model.eval()
    with torch.no_grad():
        logits = model(b["types"], b["payload"])
    pred = logits.argmax(-1)
    data = b["target"] >= 0
    metrics = {}
    for key, mask in [
        ("DATA_ACC", data),
        ("NEW_EARLY_ACC", b["new_early"]),
        ("RETURN_EARLY_ACC", b["return_early"]),
        ("STEADY_ACC", b["steady"]),
        ("LONG_GAP_ACC", b["long_gap"]),
    ]:
        a, count = acc_on_mask(pred, b["target"], mask)
        metrics[key] = a
        metrics[key + "_N"] = count
    metrics["CROSS_ENTROPY"] = float(masked_loss(logits, b["target"]))
    return metrics


def fit_ridge(X, y, lam=RIDGE):
    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.int64)
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True) + 1e-6
    Z = (X - mu) / sd
    Z = np.concatenate([Z, np.ones((len(Z), 1))], axis=1)
    Y = np.eye(N_VAL, dtype=np.float64)[y]
    A = Z.T @ Z + lam * np.eye(Z.shape[1])
    A[-1, -1] -= lam
    W = np.linalg.solve(A, Z.T @ Y)
    return mu, sd, W


def ridge_acc(fit, X, y):
    mu, sd, W = fit
    Z = (np.asarray(X, np.float64) - mu) / sd
    Z = np.concatenate([Z, np.ones((len(Z), 1))], axis=1)
    p = (Z @ W).argmax(1)
    return float(np.mean(p == np.asarray(y)))


def collect_probe(model, seed, split, smoke=False):
    n = 64 if smoke else PROBE_EPISODES
    b = make_batch(derive_seed(seed, f"probe_{split}"), n)
    model.eval()
    with torch.no_grad():
        _, hs = model(b["types"], b["payload"], return_states=True)
    # Probe ACTIVE KEY, a required latent variable never directly supervised.
    valid = (b["active_key"] >= 0)
    valid[:, 0] = False
    idx = valid.nonzero(as_tuple=False)
    cur = hs[idx[:, 0], idx[:, 1]].cpu().numpy()
    prev = hs[idx[:, 0], idx[:, 1] - 1].cpu().numpy()
    d = cur - prev
    y = b["active_key"][idx[:, 0], idx[:, 1]].cpu().numpy()
    return cur, d, y


def history_diagnostic(model, seed, smoke=False):
    tr_h, tr_d, tr_y = collect_probe(model, seed, "train", smoke)
    te_h, te_d, te_y = collect_probe(model, seed, "test", smoke)
    rng = np.random.default_rng(derive_seed(seed, "shuffle_history"))
    tr_perm = rng.permutation(len(tr_d))
    te_perm = rng.permutation(len(te_d))
    state_fit = fit_ridge(tr_h, tr_y)
    joint_fit = fit_ridge(np.concatenate([tr_h, tr_d], 1), tr_y)
    shuf_fit = fit_ridge(np.concatenate([tr_h, tr_d[tr_perm]], 1), tr_y)
    a_state = ridge_acc(state_fit, te_h, te_y)
    a_joint = ridge_acc(joint_fit, np.concatenate([te_h, te_d], 1), te_y)
    a_shuf = ridge_acc(shuf_fit, np.concatenate([te_h, te_d[te_perm]], 1), te_y)
    return {
        "state_acc": a_state,
        "history_acc": a_joint,
        "shuffled_acc": a_shuf,
        "h_gain": a_joint - a_state,
        "h_vs_shuffled": a_joint - a_shuf,
        "n_test": int(len(te_y)),
    }


def normalize(x):
    n = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    return x / torch.clamp(n, min=EPS), n


def orthogonal(u, seed):
    g = torch.Generator().manual_seed(int(seed))
    z = torch.randn(u.shape, generator=g, dtype=u.dtype)
    z = z - (z * u).sum(-1, keepdim=True) * u
    q, qn = normalize(z)
    bad = qn[:, 0] < 1e-7
    if bool(bad.any()):
        ub = u[bad]
        idx = torch.argmin(torch.abs(ub), dim=1)
        fb = torch.zeros_like(ub)
        fb[torch.arange(len(fb)), idx] = 1.0
        fb = fb - (fb * ub).sum(-1, keepdim=True) * ub
        fb, _ = normalize(fb)
        q[bad] = fb
    return q


def dynamics_diagnostic(model, seed, smoke=False):
    n = 64 if smoke else DYN_EPISODES
    b = make_batch(derive_seed(seed, "dyn_stream"), n)
    model.eval()
    with torch.no_grad():
        emb = model.encoder(b["types"], b["payload"])
        native_logits, hs = model(b["types"], b["payload"], return_states=True)
    pred_native = native_logits.argmax(-1)

    # Eligible t: t>=1 and t+3 exists; current and next three events are DATA.
    eligible = torch.zeros_like(b["target"], dtype=torch.bool)
    for t in range(1, SEQ_LEN - 3):
        ok = (b["types"][:, t] == DATA)
        for k in range(1, 4):
            ok &= (b["types"][:, t + k] == DATA)
        eligible[:, t] = ok
    idx = eligible.nonzero(as_tuple=False)
    if len(idx) > (512 if smoke else DYN_MAX_POINTS):
        rng = np.random.default_rng(derive_seed(seed, "dyn_subsample"))
        choose = rng.choice(len(idx), size=(512 if smoke else DYN_MAX_POINTS), replace=False)
        idx = idx[torch.from_numpy(np.sort(choose))]

    if len(idx) == 0:
        raise RuntimeError("no eligible dynamics points")

    h = hs[idx[:, 0], idx[:, 1]].detach().cpu()
    hp = hs[idx[:, 0], idx[:, 1] - 1].detach().cpu()
    d = h - hp
    u, m = normalize(d)
    valid = m[:, 0] >= 1e-7
    h = h[valid]
    u = u[valid]
    m = m[valid]
    idx = idx[valid]

    cond_rows = []
    for rho in RHOS:
        r = float(rho) * m
        arms = {
            "TP": h + r * u,
            "TM": h - r * u,
        }
        for j in range(N_TRANS):
            q = orthogonal(u, derive_seed(seed, f"dyn_q_{rho}_{j}"))
            arms[f"P{j}"] = h + r * q

        # Roll all arms under the identical future input stream for 3 steps.
        states = {k: v.clone() for k, v in arms.items()}
        for k in range(1, 4):
            e = emb[idx[:, 0], idx[:, 1] + k].detach().cpu()
            with torch.no_grad():
                for name in states:
                    states[name] = model.step_from_embedding(e, states[name])

        native_h3 = hs[idx[:, 0], idx[:, 1] + 3].detach().cpu()
        target3 = b["target"][idx[:, 0], idx[:, 1] + 3].cpu()
        native_correct = (pred_native[idx[:, 0], idx[:, 1] + 3].cpu() == target3).float()

        def arm_vals(z):
            err = torch.linalg.vector_norm(z - native_h3, dim=1) / torch.clamp(r[:, 0], min=EPS)
            logits = model.head(z)
            correct = (logits.argmax(1) == target3).float()
            # Damage is loss of correctness probability at the population level.
            damage = float(native_correct.mean() - correct.mean())
            return float(torch.median(err)), damage

        rt1, dtp = arm_vals(states["TP"])
        rt2, dtm = arm_vals(states["TM"])
        ret_tan = 0.5 * (rt1 + rt2)
        dam_tan = 0.5 * (dtp + dtm)
        trans = [arm_vals(states[f"P{j}"]) for j in range(N_TRANS)]
        ret_trans = float(np.mean([x[0] for x in trans]))
        dam_trans = float(np.mean([x[1] for x in trans]))
        cond_rows.append({
            "rho": float(rho),
            "ret_tan": ret_tan,
            "ret_trans": ret_trans,
            "d_rec": ret_tan - ret_trans,
            "damage_tan": dam_tan,
            "damage_trans": dam_trans,
            "d_func": dam_tan - dam_trans,
        })

    return {
        "n_points": int(len(idx)),
        "conditions": cond_rows,
        "ret_tan": float(np.mean([x["ret_tan"] for x in cond_rows])),
        "ret_trans": float(np.mean([x["ret_trans"] for x in cond_rows])),
        "d_rec": float(np.mean([x["d_rec"] for x in cond_rows])),
        "damage_tan": float(np.mean([x["damage_tan"] for x in cond_rows])),
        "damage_trans": float(np.mean([x["damage_trans"] for x in cond_rows])),
        "d_func": float(np.mean([x["d_func"] for x in cond_rows])),
    }


def run(seed, outdir, smoke=False):
    set_seed(seed)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    models = build_models(seed)
    architecture = {}
    for name, model in models.items():
        losses = train_model(model, seed, name, smoke)
        metrics = evaluate_model(model, seed, name, smoke)
        architecture[name] = {
            "params": count_params(model),
            "train_loss_trace": losses,
            "test": metrics,
        }

    history = history_diagnostic(models["GRR"], seed, smoke)
    dynamics = dynamics_diagnostic(models["GRR"], seed, smoke)

    finite = True
    vals = []
    for x in architecture.values():
        vals += [v for k, v in x["test"].items() if not k.endswith("_N")]
    vals += list(history.values())[:-1]
    vals += [dynamics[k] for k in ("ret_tan", "ret_trans", "d_rec", "damage_tan", "damage_trans", "d_func")]
    finite = all(math.isfinite(float(v)) for v in vals)

    summary = {
        "experiment": "R9-T1",
        "seed": int(seed),
        "expected_seed": bool(seed in SEEDS),
        "architecture": architecture,
        "history_transfer": history,
        "anisotropy_transfer": dynamics,
        "all_valid": bool(finite),
        "environment": {
            "python": __import__("sys").version,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "frozen": {
            "train_steps": 12 if smoke else TRAIN_STEPS,
            "batch": 16 if smoke else BATCH,
            "seq_len": SEQ_LEN,
            "rhos": list(RHOS),
        },
    }
    save_json(out / ("smoke_summary.json" if smoke else "seed_summary.json"), summary)
    print(
        f"seed={seed} GRR_data={architecture['GRR']['test']['DATA_ACC']:.4f} "
        f"GRR_return={architecture['GRR']['test']['RETURN_EARLY_ACC']:.4f} "
        f"H={history['h_gain']:+.4f} Drec={dynamics['d_rec']:+.4f} "
        f"Dfunc={dynamics['d_func']:+.4f} "
        f"Sret={architecture['SANDWICH']['test']['RETURN_EARLY_ACC']-architecture['TRANSFORMER']['test']['RETURN_EARLY_ACC']:+.4f}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run(args.seed, args.outdir, args.smoke)


if __name__ == "__main__":
    main()
