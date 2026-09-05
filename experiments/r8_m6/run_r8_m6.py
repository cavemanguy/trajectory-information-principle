import argparse
import copy
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
H0_DIM = 16
STEPS = 12
EPS = 1e-8
FRESH_SEEDS = (22, 36, 49, 67, 82, 97, 113, 129, 144, 159, 177, 193)
CHECKPOINTS = (20, 25, 30, 35, 40, 50, 60, 80, 100)
TRAIN_N = 20000
VAL_N = 2500
TEST_N = 5000
PAIR_N = 2048
BATCH = 256
LR = 1e-3
WD = 1e-4
CLIP = 1.0
DORMANT_SCALE = 0.02
P_HIDDEN = 126
FORK_TOL = 1e-6
EXTRA_ZERO_TOL = 1e-7


def derive_seed(seed, name):
    h = hashlib.sha256(f"R8-M6|{seed}|{name}".encode()).digest()
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
    def __init__(self, state_dim=16, f_hidden=32):
        super().__init__()
        self.state_dim = int(state_dim)
        self.f_hidden = int(f_hidden)
        self.rel_emb = nn.Embedding(N_REL, EMB)
        self.val_emb = nn.Embedding(N_VAL, EMB)
        self.enc = nn.GRU(EMB * 2, ENC_HIDDEN, batch_first=True)
        self.to_h = nn.Linear(ENC_HIDDEN, H0_DIM)
        self.F = nn.Sequential(
            nn.Linear(self.state_dim, self.f_hidden),
            nn.GELU(),
            nn.Linear(self.f_hidden, self.state_dim),
            nn.Tanh(),
        )
        self.head0 = nn.ModuleList([nn.Linear(H0_DIM, N_VAL) for _ in range(N_REL)])
        self.headT = nn.ModuleList([nn.Linear(self.state_dim, N_VAL) for _ in range(N_REL)])
        self.apply(init_linear_embedding)
        init_gru(self.enc)

    def encode(self, y, perms):
        b = len(y)
        rel = torch.arange(N_REL, device=y.device).expand(b, -1).gather(1, perms)
        val = y.gather(1, perms)
        x = torch.cat([self.rel_emb(rel), self.val_emb(val)], dim=-1)
        _, s = self.enc(x)
        return torch.tanh(self.to_h(s[-1]))

    def lift(self, h0):
        if self.state_dim == H0_DIM:
            return h0
        if self.state_dim == 32:
            return torch.cat([h0, torch.zeros_like(h0)], dim=-1)
        raise RuntimeError(f"unsupported state_dim {self.state_dim}")

    def trajectory(self, h0):
        z = self.lift(h0)
        zs = [z]
        for _ in range(STEPS):
            z = self.F(z)
            zs.append(z)
        return torch.stack(zs, dim=1)

    def forward(self, y, perms):
        h0 = self.encode(y, perms)
        tr = self.trajectory(h0)
        return h0, tr, [h(h0) for h in self.head0], [h(tr[:, -1]) for h in self.headT]


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def task_loss(model, y, perms):
    ce = nn.CrossEntropyLoss()
    _, _, l0, lT = model(y, perms)
    return sum(ce(l0[r], y[:, r]) + ce(lT[r], y[:, r]) for r in range(N_REL)) / N_REL


def eval_model(model, y, perms):
    model.eval()
    c0 = np.zeros(N_REL, dtype=np.int64)
    cT = np.zeros(N_REL, dtype=np.int64)
    with torch.no_grad():
        for a in range(0, len(y), BATCH):
            yy = y[a:a+BATCH]
            pp = perms[a:a+BATCH]
            _, _, l0, lT = model(yy, pp)
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


def winner_gap(test_or_val, winner):
    a = np.asarray(test_or_val["h12_per_relation"], dtype=np.float64)
    rest = np.delete(a, winner)
    return float(a[winner] - rest.mean())


def added_energy_summary(model, y, perms, n=512):
    if model.state_dim != 32:
        return None
    n = min(n, len(y))
    model.eval()
    with torch.no_grad():
        h0 = model.encode(y[:n], perms[:n])
        tr = model.trajectory(h0)
        total = (tr * tr).sum(-1)
        extra = (tr[:, :, 16:] * tr[:, :, 16:]).sum(-1)
        frac = extra / (total + EPS)
        return torch.median(frac, dim=0).values.cpu().tolist()


def copy_common_encoder_and_h0(base, dst):
    dst.rel_emb.load_state_dict(base.rel_emb.state_dict())
    dst.val_emb.load_state_dict(base.val_emb.state_dict())
    dst.enc.load_state_dict(base.enc.state_dict())
    dst.to_h.load_state_dict(base.to_h.state_dict())
    dst.head0.load_state_dict(base.head0.state_dict())


def dormant_noise(shape, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g) * DORMANT_SCALE


def expand_x32(base, seed):
    x = Core(state_dim=32, f_hidden=32)
    copy_common_encoder_and_h0(base, x)
    with torch.no_grad():
        x.F[0].weight.zero_()
        x.F[0].weight[:, :16].copy_(base.F[0].weight)
        x.F[0].weight[:, 16:].copy_(dormant_noise((32, 16), derive_seed(seed, "x32_dormant_recurrent_in")))
        x.F[0].bias.copy_(base.F[0].bias)

        x.F[2].weight.zero_()
        x.F[2].weight[:16, :].copy_(base.F[2].weight)
        x.F[2].bias.zero_()
        x.F[2].bias[:16].copy_(base.F[2].bias)

        for r in range(N_REL):
            x.headT[r].weight.zero_()
            x.headT[r].weight[:, :16].copy_(base.headT[r].weight)
            x.headT[r].bias.copy_(base.headT[r].bias)
    return x


def expand_p16(base, seed):
    p = Core(state_dim=16, f_hidden=P_HIDDEN)
    copy_common_encoder_and_h0(base, p)
    p.headT.load_state_dict(base.headT.state_dict())
    with torch.no_grad():
        p.F[0].weight.zero_()
        p.F[0].weight[:32, :].copy_(base.F[0].weight)
        p.F[0].bias.zero_()
        p.F[0].bias[:32].copy_(base.F[0].bias)

        p.F[2].weight.zero_()
        p.F[2].weight[:, :32].copy_(base.F[2].weight)
        p.F[2].weight[:, 32:].copy_(dormant_noise((16, P_HIDDEN - 32), derive_seed(seed, "p16_dormant_out")))
        p.F[2].bias.copy_(base.F[2].bias)
    return p


def pretrain_common(seed, train_y, smoke=False):
    set_seed(derive_seed(seed, "init"))
    model = Core(state_dim=16, f_hidden=32)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    end_ep = 2 if smoke else 20
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
    return model


def logits_and_tr(model, y, perms):
    model.eval()
    with torch.no_grad():
        h0, tr, l0, lT = model(y, perms)
        return h0, tr, torch.stack(l0, 1), torch.stack(lT, 1)


def dormant_gradient_norms(x32, p16, y, perms):
    x32.zero_grad(set_to_none=True)
    lx = task_loss(x32, y, perms)
    lx.backward()
    gx = float(torch.linalg.vector_norm(x32.F[2].weight.grad[16:, :]).detach())
    x32.zero_grad(set_to_none=True)

    p16.zero_grad(set_to_none=True)
    lp = task_loss(p16, y, perms)
    lp.backward()
    gp = float(torch.linalg.vector_norm(p16.F[0].weight.grad[32:, :]).detach())
    p16.zero_grad(set_to_none=True)
    return gx, gp


def fork_equivalence(base, x32, p16, diag_y, diag_perm, bank):
    hb, tb, l0b, lTb = logits_and_tr(base, diag_y, diag_perm)
    hx, tx, l0x, lTx = logits_and_tr(x32, diag_y, diag_perm)
    hp, tp, l0p, lTp = logits_and_tr(p16, diag_y, diag_perm)

    sb = survival_summary(base, bank)
    sx = survival_summary(x32, bank)
    sp = survival_summary(p16, bank)

    surv_b = np.asarray(sb["terminal_survival"])
    surv_x = np.asarray(sx["terminal_survival"])
    surv_p = np.asarray(sp["terminal_survival"])

    n_x = count_params(x32)
    n_p = count_params(p16)
    param_rel = abs(n_x - n_p) / max(n_x, n_p)
    grad_x, grad_p = dormant_gradient_norms(x32, p16, diag_y[: min(64, len(diag_y))], diag_perm[: min(64, len(diag_perm))])

    checks = {
        "h0_logits_x_vs_b_maxabs": float((l0x - l0b).abs().max()),
        "h0_logits_p_vs_b_maxabs": float((l0p - l0b).abs().max()),
        "h12_logits_x_vs_b_maxabs": float((lTx - lTb).abs().max()),
        "h12_logits_p_vs_b_maxabs": float((lTp - lTb).abs().max()),
        "x_first16_traj_vs_b_maxabs": float((tx[:, :, :16] - tb).abs().max()),
        "x_extra_traj_maxabs": float(tx[:, :, 16:].abs().max()),
        "p_traj_vs_b_maxabs": float((tp - tb).abs().max()),
        "survival_x_vs_b_maxabs": float(np.max(np.abs(surv_x - surv_b))),
        "survival_p_vs_b_maxabs": float(np.max(np.abs(surv_p - surv_b))),
        "G_x_vs_b_abs": float(abs(sx["G"] - sb["G"])),
        "G_p_vs_b_abs": float(abs(sp["G"] - sb["G"])),
        "params_B16": count_params(base),
        "params_X32": n_x,
        "params_P16": n_p,
        "X32_P16_param_relative_difference": float(param_rel),
        "X32_added_state_output_gradient_norm": grad_x,
        "P16_added_hidden_input_gradient_norm": grad_p,
    }
    checks["passed"] = bool(
        checks["h0_logits_x_vs_b_maxabs"] <= FORK_TOL and
        checks["h0_logits_p_vs_b_maxabs"] <= FORK_TOL and
        checks["h12_logits_x_vs_b_maxabs"] <= FORK_TOL and
        checks["h12_logits_p_vs_b_maxabs"] <= FORK_TOL and
        checks["x_first16_traj_vs_b_maxabs"] <= FORK_TOL and
        checks["x_extra_traj_maxabs"] <= EXTRA_ZERO_TOL and
        checks["p_traj_vs_b_maxabs"] <= FORK_TOL and
        checks["survival_x_vs_b_maxabs"] <= FORK_TOL and
        checks["survival_p_vs_b_maxabs"] <= FORK_TOL and
        checks["G_x_vs_b_abs"] <= FORK_TOL and
        checks["G_p_vs_b_abs"] <= FORK_TOL and
        param_rel < 0.05 and
        grad_x > 0.0 and grad_p > 0.0
    )
    return checks, {"B16": sb, "X32": sx, "P16": sp}


def evaluate_record(model, epoch, val_y, val_perm, bank):
    val = eval_model(model, val_y, val_perm)
    surv = survival_summary(model, bank)
    return {
        "epoch": int(epoch),
        "validation": val,
        "survival": surv,
        "D_validation": winner_gap(val, surv["winner_relation"]),
        "added_energy_median_fraction": added_energy_summary(model, val_y, val_perm),
    }


def continue_model(seed, name, model, train_y, val_y, val_perm, bank, start_ep, end_ep, smoke=False):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    checkpoint_set = {start_ep, end_ep} if smoke else set(CHECKPOINTS)
    records = [evaluate_record(model, start_ep, val_y, val_perm, bank)]
    for ep in range(start_ep + 1, end_ep + 1):
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
        if ep in checkpoint_set and ep != start_ep:
            rec = evaluate_record(model, ep, val_y, val_perm, bank)
            records.append(rec)
            print(
                f"seed={seed} cond={name} ep={ep} h0={rec['validation']['h0_overall']:.4f} "
                f"h12={rec['validation']['h12_overall']:.4f} G={rec['survival']['G']:.4f} D={rec['D_validation']:.4f}",
                flush=True,
            )
    return model, records


def run(seed, outdir, smoke=False):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    ntrain = 512 if smoke else TRAIN_N
    nval = 256 if smoke else VAL_N
    ntest = 256 if smoke else TEST_N
    pair_n = 96 if smoke else PAIR_N
    start_ep = 2 if smoke else 20
    end_ep = 4 if smoke else 100

    train_y = make_memories(ntrain, derive_seed(seed, "train"))
    val_y = make_memories(nval, derive_seed(seed, "val"))
    test_y = make_memories(ntest, derive_seed(seed, "test"))
    val_perm = make_perms(nval, derive_seed(seed, "val_perm"))
    test_perm = make_perms(ntest, derive_seed(seed, "test_perm"))
    bank = make_pair_bank(seed, pair_n)

    common = pretrain_common(seed, train_y, smoke=smoke)
    b16 = copy.deepcopy(common)
    x32 = expand_x32(common, seed)
    p16 = expand_p16(common, seed)

    diag_n = min(256, nval)
    fork_checks, fork_surv = fork_equivalence(b16, x32, p16, val_y[:diag_n], val_perm[:diag_n], bank)
    if smoke and not fork_checks["passed"]:
        raise RuntimeError(f"fork equivalence failed: {fork_checks}")

    models = {"B16": b16, "X32": x32, "P16": p16}
    results = {}
    for name, model in models.items():
        final_model, records = continue_model(
            seed, name, model, train_y, val_y, val_perm, bank, start_ep, end_ep, smoke=smoke
        )
        test = eval_model(final_model, test_y, test_perm)
        surv = survival_summary(final_model, bank)
        results[name] = {
            "records": records,
            "test": test,
            "test_survival": surv,
            "D_test": winner_gap(test, surv["winner_relation"]),
            "final_state_sha256": sha_state_dict(final_model.state_dict()),
            "parameter_count": count_params(final_model),
            "added_energy_test_median_fraction": added_energy_summary(final_model, test_y, test_perm),
        }

    if smoke:
        save_json(out / "smoke_summary.json", {
            "status": "ok",
            "seed": seed,
            "fork_equivalence": fork_checks,
            "fork_survival": fork_surv,
            "results": {k: {"test": v["test"], "D_test": v["D_test"]} for k, v in results.items()},
        })
        print(json.dumps({"smoke": "ok", "fork_equivalence": fork_checks}, indent=2))
        return

    validity = {}
    for name, d in results.items():
        last_val = d["records"][-1]["validation"]
        validity[name] = bool(last_val["combined"] >= 0.38 and last_val["h0_overall"] >= 0.55)

    summary = {
        "experiment": "R8-M6",
        "seed": int(seed),
        "fresh_seed": seed in FRESH_SEEDS,
        "fork_equivalence": fork_checks,
        "training_validity": validity,
        "all_training_valid": bool(all(validity.values())),
        "conditions": results,
        "epoch100": {
            name: {
                "validation": d["records"][-1]["validation"],
                "survival": d["records"][-1]["survival"],
                "D_validation": d["records"][-1]["D_validation"],
                "test": d["test"],
                "test_survival": d["test_survival"],
                "D_test": d["D_test"],
                "parameter_count": d["parameter_count"],
            }
            for name, d in results.items()
        },
    }
    save_json(out / "seed_summary.json", summary)
    print(json.dumps({
        "seed": seed,
        "fork_ok": fork_checks["passed"],
        "validity": validity,
        "epoch100": summary["epoch100"],
    }, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run(args.seed, args.outdir, smoke=args.smoke)


if __name__ == "__main__":
    main()
