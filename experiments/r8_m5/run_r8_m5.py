import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

N_REL = 8
N_VAL = 16
EMB = 8
ENC_HIDDEN = 32
STEPS = 12
EPS = 1e-8
FRESH_SEEDS = (18, 33, 46, 61, 76, 91, 106, 121, 141, 156, 173, 188)
TRAIN_N = 20000
VAL_N = 2500
TEST_N = 5000
PAIR_N = 2048
BATCH = 256
LR = 1e-3
WD = 1e-4
CLIP = 1.0
CHECKPOINTS = (0, 20, 40, 60, 80, 100)
SPECS = {
    "B16": {"latent": 16, "fhidden": 32},
    "S24": {"latent": 24, "fhidden": 32},
    "S32": {"latent": 32, "fhidden": 32},
    "P16": {"latent": 16, "fhidden": 192},
}


def derive_seed(seed, name):
    h = hashlib.sha256(f"R8-M5|{seed}|{name}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def clone_state(sd):
    return {k: v.detach().cpu().clone() for k, v in sd.items()}


def sha_state_dict(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def make_memories(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, N_VAL, (n, N_REL), generator=g)


def make_perms(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.stack([torch.randperm(N_REL, generator=g) for _ in range(n)])


def init_linear_embedding(m):
    if isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def init_gru(gru):
    for name, p in gru.named_parameters():
        if "weight_ih" in name:
            nn.init.xavier_uniform_(p)
        elif "weight_hh" in name:
            for block in p.chunk(3, 0):
                nn.init.orthogonal_(block)
        elif "bias" in name:
            nn.init.zeros_(p)


class Core(nn.Module):
    def __init__(self, latent, fhidden):
        super().__init__()
        self.latent = int(latent)
        self.fhidden = int(fhidden)
        self.rel_emb = nn.Embedding(N_REL, EMB)
        self.val_emb = nn.Embedding(N_VAL, EMB)
        self.enc = nn.GRU(EMB * 2, ENC_HIDDEN, batch_first=True)
        self.to_h = nn.Linear(ENC_HIDDEN, latent)
        self.F = nn.Sequential(
            nn.Linear(latent, fhidden),
            nn.GELU(),
            nn.Linear(fhidden, latent),
            nn.Tanh(),
        )
        self.head0 = nn.ModuleList([nn.Linear(latent, N_VAL) for _ in range(N_REL)])
        self.headT = nn.ModuleList([nn.Linear(latent, N_VAL) for _ in range(N_REL)])
        self.apply(init_linear_embedding)
        init_gru(self.enc)

    def encode(self, y, perms):
        b = len(y)
        rel = torch.arange(N_REL, device=y.device).expand(b, -1).gather(1, perms)
        val = y.gather(1, perms)
        x = torch.cat([self.rel_emb(rel), self.val_emb(val)], dim=-1)
        _, s = self.enc(x)
        return torch.tanh(self.to_h(s[-1]))

    def trajectory(self, h0):
        hs = [h0]
        h = h0
        for _ in range(STEPS):
            h = self.F(h)
            hs.append(h)
        return torch.stack(hs, dim=1)

    def forward(self, y, perms):
        h0 = self.encode(y, perms)
        tr = self.trajectory(h0)
        return tr, [h(h0) for h in self.head0], [h(tr[:, -1]) for h in self.headT]


def parameter_count(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def task_loss(model, y, perms):
    ce = nn.CrossEntropyLoss()
    _, l0, lT = model(y, perms)
    return sum(ce(l0[r], y[:, r]) + ce(lT[r], y[:, r]) for r in range(N_REL)) / N_REL


def eval_model(model, y, perms):
    model.eval()
    c0 = np.zeros(N_REL, dtype=np.int64)
    cT = np.zeros(N_REL, dtype=np.int64)
    with torch.no_grad():
        for a in range(0, len(y), BATCH):
            yy = y[a:a+BATCH]
            pp = perms[a:a+BATCH]
            _, l0, lT = model(yy, pp)
            for r in range(N_REL):
                c0[r] += int((l0[r].argmax(1) == yy[:, r]).sum())
                cT[r] += int((lT[r].argmax(1) == yy[:, r]).sum())
    a0 = c0 / len(y)
    aT = cT / len(y)
    return {
        "h0_overall": float(a0.mean()),
        "h12_overall": float(aT.mean()),
        "combined": float((a0.mean() + aT.mean()) / 2.0),
        "h0_per_relation": a0.tolist(),
        "h12_per_relation": aT.tolist(),
    }


def make_pair_bank(seed, n):
    bank = []
    for r in range(N_REL):
        base = make_memories(n, derive_seed(seed, f"eval_pair_base_{r}"))
        alt = base.clone()
        g = torch.Generator().manual_seed(derive_seed(seed, f"eval_pair_offset_{r}"))
        off = torch.randint(1, N_VAL, (n,), generator=g)
        alt[:, r] = (alt[:, r] + off) % N_VAL
        perms = make_perms(n, derive_seed(seed, f"eval_pair_perm_{r}"))
        bank.append((base, alt, perms))
    return bank


def survival_summary(model, bank):
    model.eval()
    rel_terminal = []
    with torch.no_grad():
        for base, alt, perms in bank:
            vals = []
            for a in range(0, len(base), BATCH):
                p = perms[a:a+BATCH]
                h0a = model.encode(base[a:a+BATCH], p)
                h0b = model.encode(alt[a:a+BATCH], p)
                ta = model.trajectory(h0a)
                tb = model.trajectory(h0b)
                d = torch.linalg.vector_norm(ta - tb, dim=-1)
                vals.append((d[:, -1] / (d[:, 0] + EPS)).cpu())
            rel_terminal.append(float(torch.cat(vals).median()))
    arr = np.asarray(rel_terminal, dtype=np.float64)
    logs = np.log(np.clip(arr, 1e-12, None))
    return {
        "terminal_survival": arr.tolist(),
        "terminal_log_survival": logs.tolist(),
        "G": float(np.std(logs, ddof=0)),
        "C": float(np.mean(logs)),
        "winner_relation": int(np.argmax(arr)),
    }


def winner_gap(surv, perf):
    w = int(surv["winner_relation"])
    acc = np.asarray(perf["h12_per_relation"], dtype=np.float64)
    return float(acc[w] - np.delete(acc, w).mean())


def make_initial_states(seed):
    states = {}
    params = {}
    copied = {}
    set_seed(derive_seed(seed, "init_B16"))
    base = Core(**SPECS["B16"])
    base_sd = clone_state(base.state_dict())
    states["B16"] = base_sd
    params["B16"] = parameter_count(base)
    copied["B16"] = {"tensor_count": len(base_sd), "sha256": sha_state_dict(base_sd)}

    for name in ("S24", "S32", "P16"):
        set_seed(derive_seed(seed, f"init_{name}"))
        model = Core(**SPECS[name])
        sd = clone_state(model.state_dict())
        matched = 0
        for k in list(sd):
            if k in base_sd and tuple(sd[k].shape) == tuple(base_sd[k].shape):
                sd[k] = base_sd[k].clone()
                matched += 1
        model.load_state_dict(sd)
        states[name] = clone_state(model.state_dict())
        params[name] = parameter_count(model)
        copied[name] = {"tensor_count": matched, "sha256": sha_state_dict(states[name])}

    ratio = abs(params["P16"] - params["S32"]) / params["S32"]
    if ratio >= 0.05:
        raise RuntimeError(f"P16/S32 parameter mismatch too large: {ratio:.6f}")
    return states, params, copied, float(ratio)


def train_condition(seed, name, init_state, train_y, val_y, val_perm, test_y, test_perm, bank, out, smoke=False):
    spec = SPECS[name]
    model = Core(**spec)
    model.load_state_dict(clone_state(init_state))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    end_ep = 2 if smoke else 100
    checkpoints = {0, 2} if smoke else set(CHECKPOINTS)
    records = []

    if 0 in checkpoints:
        val = eval_model(model, val_y, val_perm)
        surv = survival_summary(model, bank)
        records.append({"epoch": 0, "validation": val, "survival": surv})
        torch.save({"epoch": 0, "condition": name, "state_dict": clone_state(model.state_dict())},
                   out / f"{name}_epoch_000.pt")

    for ep in range(1, end_ep + 1):
        model.train()
        train_perm = make_perms(len(train_y), derive_seed(seed, f"presentation_{ep}"))
        g = torch.Generator().manual_seed(derive_seed(seed, f"order_{ep}"))
        order = torch.randperm(len(train_y), generator=g)
        for a in range(0, len(train_y), BATCH):
            ix = order[a:a+BATCH]
            opt.zero_grad(set_to_none=True)
            loss = task_loss(model, train_y[ix], train_perm[ix])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()

        if ep in checkpoints:
            val = eval_model(model, val_y, val_perm)
            surv = survival_summary(model, bank)
            records.append({"epoch": ep, "validation": val, "survival": surv})
            torch.save({"epoch": ep, "condition": name, "state_dict": clone_state(model.state_dict())},
                       out / f"{name}_epoch_{ep:03d}.pt")
            print(f"seed={seed} cond={name} ep={ep} h0={val['h0_overall']:.4f} "
                  f"h12={val['h12_overall']:.4f} G={surv['G']:.4f}", flush=True)

    test = eval_model(model, test_y, test_perm)
    final_surv = records[-1]["survival"]
    return {
        "records": records,
        "test": test,
        "winner_gap_test": winner_gap(final_surv, test),
        "final_state_sha256": sha_state_dict(model.state_dict()),
    }


def run(seed, outdir, smoke=False):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    ntrain = 512 if smoke else TRAIN_N
    nval = 256 if smoke else VAL_N
    ntest = 256 if smoke else TEST_N
    pair_n = 96 if smoke else PAIR_N

    train_y = make_memories(ntrain, derive_seed(seed, "train"))
    val_y = make_memories(nval, derive_seed(seed, "val"))
    test_y = make_memories(ntest, derive_seed(seed, "test"))
    val_perm = make_perms(nval, derive_seed(seed, "val_perm"))
    test_perm = make_perms(ntest, derive_seed(seed, "test_perm"))
    bank = make_pair_bank(seed, pair_n)

    states, params, copied, param_ratio = make_initial_states(seed)
    results = {}
    for name in ("B16", "S24", "S32", "P16"):
        results[name] = train_condition(
            seed, name, states[name], train_y, val_y, val_perm, test_y, test_perm, bank, out, smoke=smoke
        )

    if smoke:
        save_json(out / "smoke_summary.json", {
            "status": "ok",
            "seed": seed,
            "parameter_counts": params,
            "P16_S32_relative_parameter_difference": param_ratio,
            "initialization": copied,
        })
        return

    epoch100 = {}
    validity = {}
    for name in ("B16", "S24", "S32", "P16"):
        last = results[name]["records"][-1]
        validity[name] = bool(
            last["validation"]["combined"] >= 0.38 and
            last["validation"]["h0_overall"] >= 0.55
        )
        epoch100[name] = {
            "validation": last["validation"],
            "survival": last["survival"],
            "test": results[name]["test"],
            "winner_gap_test": results[name]["winner_gap_test"],
        }

    summary = {
        "experiment": "R8-M5",
        "seed": int(seed),
        "fresh_seed": bool(seed in FRESH_SEEDS),
        "specs": SPECS,
        "parameter_counts": params,
        "P16_S32_relative_parameter_difference": param_ratio,
        "initialization": copied,
        "condition_validity": validity,
        "all_conditions_valid": bool(all(validity.values())),
        "conditions": results,
        "epoch100": epoch100,
        "environment": {
            "python": __import__("sys").version,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    save_json(out / "seed_summary.json", summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if not args.smoke and args.seed not in FRESH_SEEDS:
        raise SystemExit(f"seed {args.seed} is not preregistered")
    run(args.seed, args.outdir, smoke=args.smoke)


if __name__ == "__main__":
    main()
