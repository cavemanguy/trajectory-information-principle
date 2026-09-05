import argparse
import copy
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

N_REL = 8
N_VAL = 16
EMB = 8
HIDDEN = 32
LATENT = 16
STEPS = 12
EPS = 1e-8
FRESH_SEEDS = (16, 26, 39, 52, 64, 78, 93, 109, 122, 136, 148, 163)
CHECKPOINTS = (20, 25, 30, 35, 40, 50, 60, 80, 100)
TRAIN_N = 20000
VAL_N = 2500
TEST_N = 5000
PAIR_N = 2048
BATCH = 256
AUX_N = 64
LR = 1e-3
WD = 1e-4
CLIP = 1.0
LAMBDA = 0.50


def derive_seed(seed, name):
    h = hashlib.sha256(f"R8-M4|{seed}|{name}".encode()).digest()
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
    def __init__(self):
        super().__init__()
        self.rel_emb = nn.Embedding(N_REL, EMB)
        self.val_emb = nn.Embedding(N_VAL, EMB)
        self.enc = nn.GRU(EMB * 2, HIDDEN, batch_first=True)
        self.to_h = nn.Linear(HIDDEN, LATENT)
        self.F = nn.Sequential(
            nn.Linear(LATENT, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, LATENT),
            nn.Tanh(),
        )
        self.head0 = nn.ModuleList([nn.Linear(LATENT, N_VAL) for _ in range(N_REL)])
        self.headT = nn.ModuleList([nn.Linear(LATENT, N_VAL) for _ in range(N_REL)])
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


def make_aux_batch(train_y, seed, epoch, step):
    g = torch.Generator().manual_seed(derive_seed(seed, f"aux_index_{epoch}_{step}"))
    ix = torch.randint(0, len(train_y), (AUX_N,), generator=g)
    base = train_y[ix].clone()
    perms = make_perms(AUX_N, derive_seed(seed, f"aux_perm_{epoch}_{step}"))
    variants = []
    for r in range(N_REL):
        v = base.clone()
        go = torch.Generator().manual_seed(derive_seed(seed, f"aux_offset_{epoch}_{step}_{r}"))
        off = torch.randint(1, N_VAL, (AUX_N,), generator=go)
        v[:, r] = (v[:, r] + off) % N_VAL
        variants.append(v)
    return base, torch.stack(variants, dim=0), perms


def relation_log_survival(model, base, variants, perms):
    h0 = model.encode(base, perms)
    h12 = model.trajectory(h0)[:, -1]
    z = []
    for r in range(N_REL):
        hv0 = model.encode(variants[r], perms)
        hv12 = model.trajectory(hv0)[:, -1]
        d0 = torch.linalg.vector_norm(h0 - hv0, dim=-1).mean()
        dT = torch.linalg.vector_norm(h12 - hv12, dim=-1).mean()
        ratio = dT / (d0 + EPS)
        z.append(torch.log(ratio + EPS))
    return torch.stack(z)


def fork_target_m20(model, train_y, seed):
    base, variants, perms = make_aux_batch(train_y, seed, 20, -1)
    with torch.no_grad():
        return float(relation_log_survival(model, base, variants, perms).mean())


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


def save_checkpoint(model, path, epoch, condition):
    torch.save({"epoch": epoch, "condition": condition, "state_dict": clone_state(model.state_dict())}, path)


def shared_pretrain(seed, train_y, val_y, val_perm, out, smoke=False):
    set_seed(derive_seed(seed, "init"))
    model = Core()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    max_ep = 2 if smoke else 20
    for ep in range(1, max_ep + 1):
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
    return model, opt


def continue_condition(seed, condition, start_state, opt_state, train_y, val_y, val_perm, test_y, test_perm, bank, m20, out, smoke=False):
    model = Core()
    model.load_state_dict(clone_state(start_state))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    opt.load_state_dict(copy.deepcopy(opt_state))
    records = []
    start_ep = 2 if smoke else 20
    end_ep = 4 if smoke else 100
    checkpoint_set = {start_ep, end_ep} if smoke else set(CHECKPOINTS)

    if start_ep in checkpoint_set:
        save_checkpoint(model, out / f"{condition}_epoch_{start_ep:03d}.pt", start_ep, condition)

    step_global = 0
    for ep in range(start_ep + 1, end_ep + 1):
        model.train()
        train_perm = make_perms(len(train_y), derive_seed(seed, f"presentation_{ep}"))
        g = torch.Generator().manual_seed(derive_seed(seed, f"order_{ep}"))
        order = torch.randperm(len(train_y), generator=g)
        for a in range(0, len(train_y), BATCH):
            ix = order[a:a+BATCH]
            opt.zero_grad(set_to_none=True)
            loss = task_loss(model, train_y[ix], train_perm[ix])
            reg = torch.tensor(0.0)
            if condition in ("E", "M"):
                base, variants, perms = make_aux_batch(train_y, seed, ep, step_global)
                z = relation_log_survival(model, base, variants, perms)
                if condition == "E":
                    reg = torch.mean((z - z.mean()) ** 2)
                else:
                    reg = (z.mean() - float(m20)) ** 2
                loss = loss + LAMBDA * reg
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            step_global += 1

        if ep in checkpoint_set:
            val = eval_model(model, val_y, val_perm)
            surv = survival_summary(model, bank)
            records.append({"epoch": ep, "validation": val, "survival": surv})
            save_checkpoint(model, out / f"{condition}_epoch_{ep:03d}.pt", ep, condition)
            print(f"seed={seed} cond={condition} ep={ep} h0={val['h0_overall']:.4f} h12={val['h12_overall']:.4f} G={surv['G']:.4f}", flush=True)

    test = eval_model(model, test_y, test_perm)
    return model, records, test


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

    pre, pre_opt = shared_pretrain(seed, train_y, val_y, val_perm, out, smoke=smoke)
    start_state = clone_state(pre.state_dict())
    start_hash = sha_state_dict(start_state)
    opt_state = copy.deepcopy(pre_opt.state_dict())
    m20 = fork_target_m20(pre, train_y, seed)

    results = {}
    final_models = {}
    for cond in ("B", "E", "M"):
        model, records, test = continue_condition(
            seed, cond, start_state, opt_state, train_y, val_y, val_perm, test_y, test_perm, bank, m20, out, smoke=smoke
        )
        results[cond] = {"records": records, "test": test, "final_state_sha256": sha_state_dict(model.state_dict())}
        final_models[cond] = model

    fork_hashes = {}
    start_ep = 2 if smoke else 20
    for cond in ("B", "E", "M"):
        ck = torch.load(out / f"{cond}_epoch_{start_ep:03d}.pt", map_location="cpu", weights_only=False)
        fork_hashes[cond] = sha_state_dict(ck["state_dict"])
    if len(set(fork_hashes.values())) != 1 or next(iter(fork_hashes.values())) != start_hash:
        raise RuntimeError("fork checkpoints are not bit-identical")

    if smoke:
        save_json(out / "smoke_summary.json", {"status": "ok", "seed": seed, "fork_hashes": fork_hashes, "m20": m20})
        return

    def last(cond):
        return results[cond]["records"][-1]

    baseline_valid = bool(last("B")["validation"]["combined"] >= 0.38 and last("B")["validation"]["h0_overall"] >= 0.55)
    summary = {
        "experiment": "R8-M4",
        "seed": seed,
        "fresh_seed": seed in FRESH_SEEDS,
        "m20": m20,
        "fork_hashes": fork_hashes,
        "baseline_valid": baseline_valid,
        "conditions": results,
        "epoch100": {
            c: {
                "validation": last(c)["validation"],
                "survival": last(c)["survival"],
                "test": results[c]["test"],
            } for c in ("B", "E", "M")
        },
        "environment": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__},
    }
    save_json(out / "seed_summary.json", summary)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run(args.seed, args.outdir, args.smoke)
