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
FRESH_SEEDS = (214, 230, 247, 263, 279, 296, 313, 329, 346, 362, 378, 397)
TRAIN_N = 20000
VAL_N = 2500
TEST_N = 5000
PAIR_N = 2048
BATCH = 256
LR = 1e-3
WD = 1e-4
CLIP = 1.0
CHECK_EVERY = 10
FIRST_RECORDED_CHECK = 40
FIRST_ELIGIBLE_MATURITY = 60
MAX_BASELINE_EPOCH = 400
PHASE_LEN = 40
TARGET_WEIGHT = 4.0
CONDITIONS = ("SWITCH", "FIX", "H0SWITCH")


def derive_seed(seed, name):
    h = hashlib.sha256(f"R8-M7R|{seed}|{name}".encode()).digest()
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


def q_from_survival(surv, A, B):
    return float(surv["alignment"][int(B)] - surv["alignment"][int(A)])


def checkpoint_record(model, val_y, val_perm, bank, epoch, A=None, B=None):
    perf = eval_model(model, val_y, val_perm)
    surv = survival_summary(model, bank)
    competent = bool(perf["combined"] >= 0.38 and perf["h0_overall"] >= 0.55)
    q = None if A is None or B is None else q_from_survival(surv, A, B)
    return {
        "epoch": int(epoch),
        "validation": perf,
        "survival": surv,
        "competent": competent,
        "Q": q,
    }


def maturity_from_history(history):
    if len(history) < 3:
        return None
    cur, prev, prev2 = history[-1], history[-2], history[-3]
    if cur["epoch"] < FIRST_ELIGIBLE_MATURITY:
        return None
    consecutive = (cur["epoch"] - prev["epoch"] == CHECK_EVERY and prev["epoch"] - prev2["epoch"] == CHECK_EVERY)
    competent3 = bool(cur["competent"] and prev["competent"] and prev2["competent"])
    stable_ab = bool(
        cur["survival"]["winner_relation"] == prev["survival"]["winner_relation"] and
        cur["survival"]["loser_relation"] == prev["survival"]["loser_relation"]
    )
    if consecutive and competent3 and stable_ab:
        return int(cur["epoch"])
    return None


def train_one_epoch(model, opt, seed, train_y, ep, h0_target=None, h12_target=None):
    model.train()
    train_perm = make_perms(len(train_y), derive_seed(seed, f"presentation_{ep}"))
    g = torch.Generator().manual_seed(derive_seed(seed, f"order_{ep}"))
    order = torch.randperm(len(train_y), generator=g)
    for a in range(0, len(train_y), BATCH):
        ix = order[a:a+BATCH]
        opt.zero_grad(set_to_none=True)
        loss = task_loss(model, train_y[ix], train_perm[ix], h0_target=h0_target, h12_target=h12_target)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss seed={seed} epoch={ep}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()


def post_targets(condition, phase, A, B):
    if condition == "SWITCH":
        return None, (A if phase in (1, 3) else B)
    if condition == "FIX":
        return None, A
    if condition == "H0SWITCH":
        return (A if phase in (1, 3) else B), None
    raise ValueError(condition)


def finite_record(r):
    vals = [r["Q"], r["validation"]["h0_overall"], r["validation"]["h12_overall"], r["survival"]["G"]]
    return all(v is not None and math.isfinite(float(v)) for v in vals)


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

    set_seed(derive_seed(seed, "init"))
    model = Core()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    if smoke:
        for ep in range(1, 3):
            train_one_epoch(model, opt, seed, train_y, ep)
        base_surv = survival_summary(model, bank)
        A = int(base_surv["winner_relation"])
        B = int(base_surv["loser_relation"])
        M = 2
        base_model_state = clone_state(model.state_dict())
        base_opt_state = copy.deepcopy(opt.state_dict())
        base_sha = sha_state_dict(base_model_state)
        conditions = {}
        for condition in CONDITIONS:
            m = Core(); m.load_state_dict(clone_state(base_model_state))
            o = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD); o.load_state_dict(copy.deepcopy(base_opt_state))
            assert sha_state_dict(m.state_dict()) == base_sha
            records = []
            ep = M
            for phase in (1, 2, 3):
                h0_target, h12_target = post_targets(condition, phase, A, B)
                ep += 1
                train_one_epoch(m, o, seed, train_y, ep, h0_target, h12_target)
                records.append(checkpoint_record(m, val_y, val_perm, bank, ep, A, B))
            conditions[condition] = {"records": records}
        fake = []
        for ep in (40, 50, 60):
            fake.append({"epoch": ep, "competent": True, "survival": {"winner_relation": 2, "loser_relation": 5}})
        assert maturity_from_history(fake) == 60
        save_json(out / "smoke_summary.json", {
            "status": "ok", "seed": int(seed), "M": M, "A": A, "B": B,
            "base_state_sha256": base_sha,
            "condition_Q": {c: [r["Q"] for r in conditions[c]["records"]] for c in CONDITIONS},
        })
        return

    history = []
    M = None
    for ep in range(1, MAX_BASELINE_EPOCH + 1):
        train_one_epoch(model, opt, seed, train_y, ep)
        if ep >= FIRST_RECORDED_CHECK and ep % CHECK_EVERY == 0:
            rec = checkpoint_record(model, val_y, val_perm, bank, ep)
            history.append(rec)
            print(
                f"seed={seed} baseline ep={ep} combined={rec['validation']['combined']:.4f} "
                f"h0={rec['validation']['h0_overall']:.4f} winner={rec['survival']['winner_relation']} "
                f"loser={rec['survival']['loser_relation']} competent={rec['competent']}",
                flush=True,
            )
            M = maturity_from_history(history)
            if M is not None:
                break

    if M is None:
        summary = {
            "experiment": "R8-M7R",
            "seed": int(seed),
            "fresh_seed": bool(seed in FRESH_SEEDS),
            "maturity_reached": False,
            "maturity_epoch": None,
            "baseline_history": history,
            "validity": {"maturity": False, "fork_identity": False, "finite": True, "complete": False},
            "all_valid": False,
            "environment": {"python": __import__("sys").version, "torch": torch.__version__, "numpy": np.__version__},
        }
        save_json(out / "seed_summary.json", summary)
        return

    A = int(history[-1]["survival"]["winner_relation"])
    B = int(history[-1]["survival"]["loser_relation"])
    if A == B:
        raise RuntimeError("maturity winner and loser identical")
    baseline = checkpoint_record(model, val_y, val_perm, bank, M, A, B)
    base_model_state = clone_state(model.state_dict())
    base_opt_state = copy.deepcopy(opt.state_dict())
    base_sha = sha_state_dict(base_model_state)

    conditions = {}
    fork_identity = True
    phase_ends = [M + PHASE_LEN, M + 2 * PHASE_LEN, M + 3 * PHASE_LEN]
    for condition in CONDITIONS:
        m = Core()
        m.load_state_dict(clone_state(base_model_state))
        o = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
        o.load_state_dict(copy.deepcopy(base_opt_state))
        same = sha_state_dict(m.state_dict()) == base_sha
        fork_identity = fork_identity and same
        if not same:
            raise RuntimeError(f"fork state mismatch for {condition}")
        records = []
        start = M + 1
        for phase, end in zip((1, 2, 3), phase_ends):
            h0_target, h12_target = post_targets(condition, phase, A, B)
            for ep in range(start, end + 1):
                train_one_epoch(m, o, seed, train_y, ep, h0_target, h12_target)
            rec = checkpoint_record(m, val_y, val_perm, bank, end, A, B)
            records.append(rec)
            print(
                f"seed={seed} cond={condition} phase={phase} ep={end} Q={rec['Q']:.4f} "
                f"winner={rec['survival']['winner_relation']} h12={rec['validation']['h12_overall']:.4f}",
                flush=True,
            )
            start = end + 1
        conditions[condition] = {
            "records": records,
            "test": eval_model(m, test_y, test_perm),
            "final_state_sha256": sha_state_dict(m.state_dict()),
        }

    finite = all(finite_record(r) for c in CONDITIONS for r in conditions[c]["records"])
    complete = all(len(conditions[c]["records"]) == 3 for c in CONDITIONS)
    summary = {
        "experiment": "R8-M7R",
        "seed": int(seed),
        "fresh_seed": bool(seed in FRESH_SEEDS),
        "maturity_reached": True,
        "maturity_epoch": int(M),
        "A_baseline_winner": A,
        "B_baseline_loser": B,
        "baseline": baseline,
        "baseline_history": history,
        "baseline_state_sha256": base_sha,
        "phase_ends": phase_ends,
        "conditions": conditions,
        "validity": {
            "maturity": True,
            "fork_identity": bool(fork_identity),
            "finite": bool(finite),
            "complete": bool(complete),
        },
        "environment": {"python": __import__("sys").version, "torch": torch.__version__, "numpy": np.__version__},
    }
    summary["all_valid"] = bool(all(summary["validity"].values()))
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
