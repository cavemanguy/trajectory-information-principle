import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Import the byte-frozen R8-M10 lineage engine directly.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r8_m10"))
import m7r_base as base  # noqa: E402

FRESH_SEEDS = (1341, 1359, 1376, 1394, 1412, 1429, 1447, 1465, 1483, 1500, 1518, 1536)
CHECK_EVERY = 10
FIRST_RECORDED_CHECK = 40
MAX_BASELINE_EPOCH = 400
PAIR_N = 2048
SUSC_N = 2048
A_HISTORY = ((0.00, 60), (0.25, 30), (0.50, 30))
B_HISTORY = ((1.00, 60), (0.75, 30), (0.50, 30))
MIDPOINT_POST_EPOCH = 120
LINEAGES = (
    ("TRUE_A", "AB", A_HISTORY),
    ("TRUE_B", "AB", B_HISTORY),
    ("NULL_C", "CD", A_HISTORY),
    ("NULL_D", "CD", B_HISTORY),
)
EPS = 1e-12


def derive_seed(seed, name):
    h = hashlib.sha256(f"R8-M12|{seed}|{name}".encode()).digest()
    return int.from_bytes(h[:4], "big")


base.derive_seed = derive_seed


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def clone_state(sd):
    return {k: v.detach().cpu().clone() for k, v in sd.items()}


def _hash_obj(h, obj):
    if torch.is_tensor(obj):
        t = obj.detach().cpu().contiguous()
        h.update(b"T")
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    elif isinstance(obj, dict):
        h.update(b"D")
        for k in sorted(obj.keys(), key=lambda x: str(x)):
            h.update(str(k).encode())
            _hash_obj(h, obj[k])
    elif isinstance(obj, (list, tuple)):
        h.update(b"L")
        for x in obj:
            _hash_obj(h, x)
    else:
        h.update(b"S")
        h.update(repr(obj).encode())


def sha_optimizer_state(state):
    h = hashlib.sha256()
    _hash_obj(h, state)
    return h.hexdigest()


def pick_off_axis_pair(surv, A, B):
    arr = np.asarray(surv["terminal_survival"], dtype=np.float64)
    order = [int(r) for r in np.argsort(arr, kind="stable")]
    rest = [r for r in order if r not in (int(A), int(B))]
    mid = len(rest) // 2
    C, D = sorted(rest[max(0, mid - 1):max(0, mid - 1) + 2])
    if C == D or C in (A, B) or D in (A, B):
        raise RuntimeError("off-axis pair not disjoint")
    return int(C), int(D)


def task_loss_lambda(model, y, perms, P, Qr, lam):
    ce = nn.CrossEntropyLoss()
    _, l0, lT = model(y, perms)
    z0 = torch.stack([ce(l0[r], y[:, r]) for r in range(base.N_REL)])
    zT = torch.stack([ce(lT[r], y[:, r]) for r in range(base.N_REL)])
    w = torch.ones(base.N_REL, dtype=zT.dtype, device=zT.device)
    w[int(P)] = 1.0 + 3.0 * (1.0 - float(lam))
    w[int(Qr)] = 1.0 + 3.0 * float(lam)
    return 0.5 * (z0.mean() + (zT * w).sum() / w.sum())


def _group_named_params(model):
    E = []
    for modname in ("rel_emb", "val_emb", "enc", "to_h"):
        E.extend(list(getattr(model, modname).parameters()))
    return {
        "F1": list(model.F[0].parameters()),
        "F2": list(model.F[2].parameters()),
        "E": E,
        "R": list(model.head0.parameters()) + list(model.headT.parameters()),
    }


def _flat_grads(grads):
    return torch.cat([g.detach().reshape(-1).double().cpu() for g in grads])


def susceptibility_axis(model, y, perms, P, Qr):
    groups = _group_named_params(model)
    before = base.sha_state_dict(model.state_dict())
    model.train()

    params = list(model.parameters())
    loss_p = task_loss_lambda(model, y, perms, P, Qr, 0.0)
    gp_all = torch.autograd.grad(loss_p, params, retain_graph=False, create_graph=False)
    map_p = {id(p): g for p, g in zip(params, gp_all)}

    loss_q = task_loss_lambda(model, y, perms, P, Qr, 1.0)
    gq_all = torch.autograd.grad(loss_q, params, retain_graph=False, create_graph=False)
    map_q = {id(p): g for p, g in zip(params, gq_all)}

    out = {}
    for name, ps in groups.items():
        gp = _flat_grads([map_p[id(p)] for p in ps])
        gq = _flat_grads([map_q[id(p)] for p in ps])
        np_ = float(torch.linalg.vector_norm(gp))
        nq = float(torch.linalg.vector_norm(gq))
        diff = float(torch.linalg.vector_norm(gq - gp))
        denom = 0.5 * (np_ + nq) + EPS
        cosine = float(torch.dot(gp, gq) / (torch.linalg.vector_norm(gp) * torch.linalg.vector_norm(gq) + EPS))
        out[name] = {
            "grad_P_norm": np_,
            "grad_Q_norm": nq,
            "contrast_norm": diff,
            "relative_contrast": float(diff / denom),
            "cosine": cosine,
        }

    after = base.sha_state_dict(model.state_dict())
    if before != after:
        raise RuntimeError("susceptibility diagnostic mutated model state")
    return out


def _flat_params(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def new_meter():
    return {"steps": 0, "grad_norm_sum": 0.0, "update_norm_sum": 0.0, "clipped_steps": 0}


def train_one_epoch_lambda(model, opt, seed, train_y, ep, P, Qr, lam, meter):
    model.train()
    train_perm = base.make_perms(len(train_y), derive_seed(seed, f"presentation_{ep}"))
    g = torch.Generator().manual_seed(derive_seed(seed, f"order_{ep}"))
    order = torch.randperm(len(train_y), generator=g)
    for a in range(0, len(train_y), base.BATCH):
        ix = order[a:a + base.BATCH]
        opt.zero_grad(set_to_none=True)
        loss = task_loss_lambda(model, train_y[ix], train_perm[ix], P, Qr, lam)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite lineage loss")
        loss.backward()
        gn = float(nn.utils.clip_grad_norm_(model.parameters(), base.CLIP))
        before = _flat_params(model).clone()
        opt.step()
        after = _flat_params(model)
        meter["steps"] += 1
        meter["grad_norm_sum"] += gn
        meter["update_norm_sum"] += float(torch.linalg.vector_norm(after - before))
        if gn > base.CLIP:
            meter["clipped_steps"] += 1


def finalize_meter(m, model, fork_flat):
    n = max(1, int(m["steps"]))
    return {
        "steps": int(m["steps"]),
        "grad_norm_mean": float(m["grad_norm_sum"] / n),
        "update_norm_mean": float(m["update_norm_sum"] / n),
        "clip_fraction": float(m["clipped_steps"] / n),
        "param_distance_from_fork": float(torch.linalg.vector_norm(_flat_params(model) - fork_flat)),
    }


def dual_axis_record(model, val_y, val_perm, bank, epoch, A, B, C, D, lam=None, post=None):
    r = base.checkpoint_record(model, val_y, val_perm, bank, epoch, A, B)
    surv = r["survival"]
    r["Q_AB"] = base.q_from_survival(surv, A, B)
    r["Q_CD"] = base.q_from_survival(surv, C, D)
    if lam is not None:
        r["lambda"] = float(lam)
    if post is not None:
        r["post_maturity_epoch"] = int(post)
    return r


def run_lineage(seed, name, axis, schedule, start_state, start_opt, start_sha, start_opt_sha,
                M, A, B, C, D, train_y, val_y, val_perm, bank, expected_post=120):
    P, Qr = (A, B) if axis == "AB" else (C, D)
    model = base.Core()
    model.load_state_dict(clone_state(start_state))
    opt = torch.optim.AdamW(model.parameters(), lr=base.LR, weight_decay=base.WD)
    opt.load_state_dict(copy.deepcopy(start_opt))
    if base.sha_state_dict(model.state_dict()) != start_sha:
        raise RuntimeError(f"lineage model fork mismatch {name}")
    if sha_optimizer_state(opt.state_dict()) != start_opt_sha:
        raise RuntimeError(f"lineage optimizer fork mismatch {name}")

    fork_flat = _flat_params(model).clone()
    meter = new_meter()
    records = []
    post = 0
    for lam, duration in schedule:
        for _ in range(int(duration)):
            post += 1
            train_one_epoch_lambda(model, opt, seed, train_y, M + post, P, Qr, lam, meter)
        records.append(dual_axis_record(model, val_y, val_perm, bank, M + post, A, B, C, D, lam, post))

    if post != int(expected_post):
        raise RuntimeError(f"lineage length mismatch {name}")
    return {
        "name": name,
        "axis": axis,
        "records": records,
        "fork_identity": True,
        "finite": all(base.finite_record(x) for x in records),
        "optimization_diagnostics": finalize_meter(meter, model, fork_flat),
    }


def terminal(lin, key):
    return float(lin["records"][-1][key])


def make_susceptibility_probe(seed, n):
    y = base.make_memories(n, derive_seed(seed, "susceptibility_y"))
    p = base.make_perms(n, derive_seed(seed, "susceptibility_perm"))
    return y, p


def run(seed, outdir, smoke=False):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    if smoke:
        train_n, val_n, pair_n = 512, 256, 96
        max_ep = 2
        schedules = {
            "TRUE_A": ((0.0, 1), (0.25, 1), (0.5, 1)),
            "TRUE_B": ((1.0, 1), (0.75, 1), (0.5, 1)),
            "NULL_C": ((0.0, 1), (0.25, 1), (0.5, 1)),
            "NULL_D": ((1.0, 1), (0.75, 1), (0.5, 1)),
        }
        expected_post = 3
        susc_n = 128
    else:
        train_n, val_n, pair_n = base.TRAIN_N, base.VAL_N, PAIR_N
        max_ep = MAX_BASELINE_EPOCH
        schedules = {k: s for k, _, s in LINEAGES}
        expected_post = MIDPOINT_POST_EPOCH
        susc_n = SUSC_N

    train_y = base.make_memories(train_n, derive_seed(seed, "train"))
    val_y = base.make_memories(val_n, derive_seed(seed, "val"))
    val_perm = base.make_perms(val_n, derive_seed(seed, "val_perm"))
    bank = base.make_pair_bank(seed, pair_n)

    base.set_seed(derive_seed(seed, "init"))
    model = base.Core()
    opt = torch.optim.AdamW(model.parameters(), lr=base.LR, weight_decay=base.WD)
    history = []
    M = None

    if smoke:
        for ep in range(1, max_ep + 1):
            base.train_one_epoch(model, opt, seed, train_y, ep)
        surv = base.survival_summary(model, bank)
        A = int(surv["winner_relation"])
        B = int(surv["loser_relation"])
        M = max_ep
        history.append(base.checkpoint_record(model, val_y, val_perm, bank, M))
    else:
        for ep in range(1, max_ep + 1):
            base.train_one_epoch(model, opt, seed, train_y, ep)
            if ep >= FIRST_RECORDED_CHECK and ep % CHECK_EVERY == 0:
                r = base.checkpoint_record(model, val_y, val_perm, bank, ep)
                history.append(r)
                M = base.maturity_from_history(history)
                if M is not None:
                    break
        if M is None:
            save_json(out / "seed_summary.json", {
                "experiment": "R8-M12", "seed": int(seed), "maturity_reached": False,
                "validity": {"maturity": False, "complete": False}, "all_valid": False,
            })
            return
        A = int(history[-1]["survival"]["winner_relation"])
        B = int(history[-1]["survival"]["loser_relation"])

    C, D = pick_off_axis_pair(history[-1]["survival"], A, B)
    pairs_disjoint = bool(len({A, B, C, D}) == 4)
    baseline = dual_axis_record(model, val_y, val_perm, bank, M, A, B, C, D)

    start_state = clone_state(model.state_dict())
    start_opt = copy.deepcopy(opt.state_dict())
    start_sha = base.sha_state_dict(start_state)
    start_opt_sha = sha_optimizer_state(start_opt)

    sy, sp = make_susceptibility_probe(seed, susc_n)
    diagnostic_before = base.sha_state_dict(model.state_dict())
    susc = {
        "AB": susceptibility_axis(model, sy, sp, A, B),
        "CD": susceptibility_axis(model, sy, sp, C, D),
    }
    diagnostic_after = base.sha_state_dict(model.state_dict())
    diagnostic_immutable = bool(diagnostic_before == diagnostic_after == start_sha)

    lineages = {}
    for name, axis, _ in LINEAGES:
        lineages[name] = run_lineage(
            seed, name, axis, schedules[name], start_state, start_opt, start_sha, start_opt_sha,
            M, A, B, C, D, train_y, val_y, val_perm, bank, expected_post=expected_post
        )

    effects = {
        "H_true_AB": terminal(lineages["TRUE_B"], "Q_AB") - terminal(lineages["TRUE_A"], "Q_AB"),
        "H_null_AB": terminal(lineages["NULL_D"], "Q_AB") - terminal(lineages["NULL_C"], "Q_AB"),
        "H_null_CD": terminal(lineages["NULL_D"], "Q_CD") - terminal(lineages["NULL_C"], "Q_CD"),
        "H_true_CD": terminal(lineages["TRUE_B"], "Q_CD") - terminal(lineages["TRUE_A"], "Q_CD"),
    }
    effects["SPECIFICITY"] = effects["H_true_AB"] - effects["H_null_AB"]

    fork_ok = all(x["fork_identity"] for x in lineages.values())
    finite_ok = all(x["finite"] for x in lineages.values())
    length_ok = all(len(x["records"]) == 3 for x in lineages.values())

    summary = {
        "experiment": "R8-M12",
        "seed": int(seed),
        "fresh_seed": bool(seed in FRESH_SEEDS),
        "maturity_reached": True,
        "maturity_epoch": int(M),
        "A_baseline_winner": int(A), "B_baseline_loser": int(B),
        "C_off_axis": int(C), "D_off_axis": int(D),
        "baseline": baseline,
        "susceptibility": susc,
        "diagnostic_state_immutable": diagnostic_immutable,
        "lineages": lineages,
        "effects": effects,
        "optimization_diagnostics": {k: v["optimization_diagnostics"] for k, v in lineages.items()},
        "validity": {
            "maturity": True,
            "pairs_disjoint": pairs_disjoint,
            "diagnostic_immutable": diagnostic_immutable,
            "fork_identity": bool(fork_ok),
            "lineage_length": bool(length_ok),
            "finite": bool(finite_ok),
            "complete": True,
        },
    }
    summary["all_valid"] = bool(all(summary["validity"].values()))
    save_json(out / ("smoke_summary.json" if smoke else "seed_summary.json"), summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run(args.seed, args.outdir, args.smoke)


if __name__ == "__main__":
    main()
