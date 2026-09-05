import argparse
import copy
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

N_REL = 8
N_VAL = 16
EMB = 8
ENC_HIDDEN = 32
LATENT = 16
F_HIDDEN = 32
STEPS = 12
EPS = 1e-8
FRESH_SEEDS = (207, 223, 239, 254, 271, 288, 304, 321, 337, 354, 371, 389)
TRAIN_N = 20000
VAL_N = 2500
TEST_N = 5000
PAIR_N = 2048
BATCH = 256
LR = 1e-3
WD = 1e-4
CLIP = 1.0
PRETRAIN_EPOCHS = 60
PHASE_LEN = 20
TARGET_WEIGHT = 4.0
CONDITIONS = ("SWITCH", "FIX", "H0SWITCH")


def derive_seed(seed, name):
    h = hashlib.sha256(f"R8-M7|{seed}|{name}".encode()).digest()
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
        self.enc = nn.GRU(EMB * 2, ENC_HIDDEN, batch_first=True)
        self.to_h = nn.Linear(ENC_HIDDEN, LATENT)
        self.F = nn.Sequential(
            nn.Linear(LATENT, F_HIDDEN),
            nn.GELU(),
            nn.Linear(F_HIDDEN, LATENT),
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


def weighted_mean_losses(losses, target=None):
    w = torch.ones(N_REL, dtype=losses[0].dtype, device=losses[0].device)
    if target is not None:
        w[int(target)] = TARGET_WEIGHT
    vals = torch.stack(losses)
    return (vals * w).sum() / w.sum()


def task_loss(model, y, perms, h0_target=None, h12_target=None):
    ce = nn.CrossEntropyLoss()
    _, l0, lT = model(y, perms)
    z0 = [ce(l0[r], y[:, r]) for r in range(N_REL)]
    zT = [ce(lT[r], y[:, r]) for r in range(N_REL)]
    return 0.5 * (weighted_mean_losses(z0, h0_target) + weighted_mean_losses(zT, h12_target))


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
        base = make_memories(n, derive_seed(seed, f"pair_base_{r}"))
        alt = base.clone()
        g = torch.Generator().manual_seed(derive_seed(seed, f"pair_offset_{r}"))
        off = torch.randint(1, N_VAL, (n,), generator=g)
        alt[:, r] = (alt[:, r] + off) % N_VAL
        perms = make_perms(n, derive_seed(seed, f"pair_perm_{r}"))
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
    align = []
    for r in range(N_REL):
        align.append(float(logs[r] - np.delete(logs, r).mean()))
    return {
        "terminal_survival": arr.tolist(),
        "terminal_log_survival": logs.tolist(),
        "alignment": align,
        "G": float(np.std(logs, ddof=0)),
        "C": float(np.mean(logs)),
        "winner_relation": int(np.argmax(arr)),
        "loser_relation": int(np.argmin(arr)),
    }


def checkpoint_record(model, val_y, val_perm, bank, epoch, A, B):
    perf = eval_model(model, val_y, val_perm)
    surv = survival_summary(model, bank)
    q = None
    if A is not None and B is not None:
        q = float(surv["alignment"][B] - surv["alignment"][A])
    return {"epoch": int(epoch), "validation": perf, "survival": surv, "Q": q}


def train_epochs(model, opt, seed, train_y, start_ep, end_ep, condition, A, B):
    for ep in range(start_ep, end_ep + 1):
        phase = 1 + (ep - (PRETRAIN_EPOCHS + 1)) // PHASE_LEN if ep > PRETRAIN_EPOCHS else 0
        h0_target = None
        h12_target = None
        if phase > 0:
            if condition == "SWITCH":
                target = A if phase in (1, 3) else B
                h12_target = target
            elif condition == "FIX":
                h12_target = A
            elif condition == "H0SWITCH":
                target = A if phase in (1, 3) else B
                h0_target = target
        model.train()
        train_perm = make_perms(len(train_y), derive_seed(seed, f"presentation_{ep}"))
        g = torch.Generator().manual_seed(derive_seed(seed, f"order_{ep}"))
        order = torch.randperm(len(train_y), generator=g)
        for a in range(0, len(train_y), BATCH):
            ix = order[a:a+BATCH]
            opt.zero_grad(set_to_none=True)
            loss = task_loss(model, train_y[ix], train_perm[ix], h0_target=h0_target, h12_target=h12_target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss seed={seed} condition={condition} epoch={ep}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()


def run(seed, outdir, smoke=False):
    global PRETRAIN_EPOCHS, PHASE_LEN
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    ntrain = 512 if smoke else TRAIN_N
    nval = 256 if smoke else VAL_N
    ntest = 256 if smoke else TEST_N
    pair_n = 96 if smoke else PAIR_N
    pretrain_epochs = 2 if smoke else PRETRAIN_EPOCHS
    phase_len = 1 if smoke else PHASE_LEN

    old_pre, old_phase = PRETRAIN_EPOCHS, PHASE_LEN
    PRETRAIN_EPOCHS, PHASE_LEN = pretrain_epochs, phase_len

    try:
        train_y = make_memories(ntrain, derive_seed(seed, "train"))
        val_y = make_memories(nval, derive_seed(seed, "val"))
        test_y = make_memories(ntest, derive_seed(seed, "test"))
        val_perm = make_perms(nval, derive_seed(seed, "val_perm"))
        test_perm = make_perms(ntest, derive_seed(seed, "test_perm"))
        bank = make_pair_bank(seed, pair_n)

        set_seed(derive_seed(seed, "init"))
        model = Core()
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

        train_epochs(model, opt, seed, train_y, 1, pretrain_epochs, "BASE", None, None)
        baseline_surv = survival_summary(model, bank)
        A = int(baseline_surv["winner_relation"])
        B = int(baseline_surv["loser_relation"])
        if A == B:
            raise RuntimeError("baseline winner and loser are identical")
        baseline = checkpoint_record(model, val_y, val_perm, bank, pretrain_epochs, A, B)

        base_model_state = clone_state(model.state_dict())
        base_opt_state = copy.deepcopy(opt.state_dict())
        base_sha = sha_state_dict(base_model_state)

        conditions = {}
        phase_ends = [pretrain_epochs + phase_len, pretrain_epochs + 2*phase_len, pretrain_epochs + 3*phase_len]
        for condition in CONDITIONS:
            m = Core()
            m.load_state_dict(clone_state(base_model_state))
            o = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
            o.load_state_dict(copy.deepcopy(base_opt_state))
            if sha_state_dict(m.state_dict()) != base_sha:
                raise RuntimeError(f"fork state mismatch for {condition}")
            records = []
            start = pretrain_epochs + 1
            for end in phase_ends:
                train_epochs(m, o, seed, train_y, start, end, condition, A, B)
                rec = checkpoint_record(m, val_y, val_perm, bank, end, A, B)
                records.append(rec)
                start = end + 1
            conditions[condition] = {
                "records": records,
                "test": eval_model(m, test_y, test_perm),
                "final_state_sha256": sha_state_dict(m.state_dict()),
            }

        if smoke:
            save_json(out / "smoke_summary.json", {
                "status": "ok",
                "seed": int(seed),
                "A": A,
                "B": B,
                "baseline_Q": baseline["Q"],
                "base_state_sha256": base_sha,
                "phase_ends": phase_ends,
                "condition_Q": {c: [r["Q"] for r in conditions[c]["records"]] for c in CONDITIONS},
            })
            return

        valid_baseline = bool(
            baseline["validation"]["combined"] >= 0.38 and
            baseline["validation"]["h0_overall"] >= 0.55
        )
        finite = True
        for c in CONDITIONS:
            for r in conditions[c]["records"]:
                vals = [r["Q"], r["validation"]["h0_overall"], r["validation"]["h12_overall"], r["survival"]["G"]]
                finite = finite and all(math.isfinite(float(v)) for v in vals)
        summary = {
            "experiment": "R8-M7",
            "seed": int(seed),
            "fresh_seed": bool(seed in FRESH_SEEDS),
            "A_baseline_winner": A,
            "B_baseline_loser": B,
            "baseline": baseline,
            "baseline_state_sha256": base_sha,
            "conditions": conditions,
            "validity": {
                "baseline_gate": valid_baseline,
                "finite": bool(finite),
                "complete": all(len(conditions[c]["records"]) == 3 for c in CONDITIONS),
            },
            "environment": {
                "python": __import__("sys").version,
                "torch": torch.__version__,
                "numpy": np.__version__,
            },
        }
        summary["all_valid"] = bool(all(summary["validity"].values()))
        save_json(out / "seed_summary.json", summary)
    finally:
        PRETRAIN_EPOCHS, PHASE_LEN = old_pre, old_phase


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
