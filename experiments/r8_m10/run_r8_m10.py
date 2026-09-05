import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import m7r_base as base

FRESH_SEEDS = (1061, 1078, 1094, 1113, 1129, 1146, 1164, 1183, 1201, 1218, 1236, 1254)
CHECK_EVERY = 10
FIRST_RECORDED_CHECK = 40
MAX_BASELINE_EPOCH = 400
PAIR_N = 2048
A_HISTORY = ((0.00, 60), (0.25, 30), (0.50, 30))
B_HISTORY = ((1.00, 60), (0.75, 30), (0.50, 30))
MIDPOINT_POST_EPOCH = 120

# Four lineages forked from one maturity state. TRUE arms attach the demand
# weights to the baseline A/B axis; NULL arms attach the identical schedule to
# the off-axis C/D pair. Q is recorded on BOTH axes in every lineage.
LINEAGES = (
    ("TRUE_A", "AB", A_HISTORY),
    ("TRUE_B", "AB", B_HISTORY),
    ("NULL_C", "CD", A_HISTORY),
    ("NULL_D", "CD", B_HISTORY),
)


def derive_seed(seed, name):
    h = hashlib.sha256(f"R8-M10|{seed}|{name}".encode()).digest()
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
    """Deterministically choose two mid-ranked relations that are neither A nor B.

    Pure function of the baseline survival state, fixed before any post-fork
    training. Ranks by terminal survival, drops A and B, and takes the two most
    central remaining relations; ties break by relation index.
    """
    arr = np.asarray(surv["terminal_survival"], dtype=np.float64)
    order = [int(r) for r in np.argsort(arr, kind="stable")]
    rest = [r for r in order if r not in (int(A), int(B))]
    if len(rest) < 2:
        raise RuntimeError("insufficient relations for an off-axis pair")
    mid = len(rest) // 2
    lo = max(0, mid - 1)
    pair = sorted(rest[lo:lo + 2])
    C, D = int(pair[0]), int(pair[1])
    if C == D or C in (int(A), int(B)) or D in (int(A), int(B)):
        raise RuntimeError(f"off-axis pair not disjoint: A={A} B={B} C={C} D={D}")
    return C, D


def task_loss_lambda(model, y, perms, P, Qr, lam):
    """R8-M8 loss with the demand pair generalized from (A,B) to (P,Qr)."""
    ce = nn.CrossEntropyLoss()
    _, l0, lT = model(y, perms)
    z0 = torch.stack([ce(l0[r], y[:, r]) for r in range(base.N_REL)])
    zT = torch.stack([ce(lT[r], y[:, r]) for r in range(base.N_REL)])
    w = torch.ones(base.N_REL, dtype=zT.dtype, device=zT.device)
    w[int(P)] = 1.0 + 3.0 * (1.0 - float(lam))
    w[int(Qr)] = 1.0 + 3.0 * float(lam)
    return 0.5 * (z0.mean() + (zT * w).sum() / w.sum())


def _flat_params(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def train_one_epoch_lambda(model, opt, seed, train_y, ep, P, Qr, lam, meter=None):
    model.train()
    train_perm = base.make_perms(len(train_y), derive_seed(seed, f"presentation_{ep}"))
    g = torch.Generator().manual_seed(derive_seed(seed, f"order_{ep}"))
    order = torch.randperm(len(train_y), generator=g)
    for a in range(0, len(train_y), base.BATCH):
        ix = order[a:a + base.BATCH]
        opt.zero_grad(set_to_none=True)
        loss = task_loss_lambda(model, train_y[ix], train_perm[ix], P, Qr, lam)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss seed={seed} epoch={ep} lambda={lam}")
        loss.backward()
        gn = float(nn.utils.clip_grad_norm_(model.parameters(), base.CLIP))
        if meter is None:
            opt.step()
        else:
            before = _flat_params(model).clone()
            opt.step()
            after = _flat_params(model)
            meter["steps"] += 1
            meter["grad_norm_sum"] += gn
            meter["update_norm_sum"] += float(torch.linalg.vector_norm(after - before))
            if gn > base.CLIP:
                meter["clipped_steps"] += 1


def new_meter():
    return {"steps": 0, "grad_norm_sum": 0.0, "update_norm_sum": 0.0, "clipped_steps": 0}


def finalize_meter(meter, model, fork_flat):
    n = max(1, int(meter["steps"]))
    return {
        "steps": int(meter["steps"]),
        "grad_norm_sum": float(meter["grad_norm_sum"]),
        "grad_norm_mean": float(meter["grad_norm_sum"] / n),
        "update_norm_sum": float(meter["update_norm_sum"]),
        "update_norm_mean": float(meter["update_norm_sum"] / n),
        "clip_fraction": float(meter["clipped_steps"] / n),
        "param_distance_from_fork": float(torch.linalg.vector_norm(_flat_params(model) - fork_flat)),
    }


def dual_axis_record(model, val_y, val_perm, bank, epoch, A, B, C, D, lam=None, post=None):
    """Record Q on BOTH the A/B axis and the C/D axis, in every lineage."""
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
                M, A, B, C, D, train_y, val_y, val_perm, bank, expected_post=MIDPOINT_POST_EPOCH,
                instrument=True):
    P, Qr = (A, B) if axis == "AB" else (C, D)

    model = base.Core()
    model.load_state_dict(clone_state(start_state))
    opt = torch.optim.AdamW(model.parameters(), lr=base.LR, weight_decay=base.WD)
    opt.load_state_dict(copy.deepcopy(start_opt))
    if base.sha_state_dict(model.state_dict()) != start_sha:
        raise RuntimeError(f"lineage fork mismatch: {name}")
    if sha_optimizer_state(opt.state_dict()) != start_opt_sha:
        raise RuntimeError(f"lineage optimizer fork mismatch: {name}")

    fork_flat = _flat_params(model).clone()
    meter = new_meter() if instrument else None
    records = []
    post = 0
    for lam, duration in schedule:
        for _ in range(int(duration)):
            post += 1
            train_one_epoch_lambda(model, opt, seed, train_y, M + post, P, Qr, lam, meter)
        r = dual_axis_record(model, val_y, val_perm, bank, M + post, A, B, C, D, lam, post)
        records.append(r)
        print(
            f"seed={seed} {name} axis={axis} post={post} lambda={lam:.2f} "
            f"Q_AB={r['Q_AB']:.4f} Q_CD={r['Q_CD']:.4f}",
            flush=True,
        )

    if expected_post is not None and post != int(expected_post):
        raise RuntimeError(f"lineage length mismatch {name}: {post}")

    return {
        "name": name,
        "axis": axis,
        "demand_pair": [int(P), int(Qr)],
        "records": records,
        "state_sha256": base.sha_state_dict(model.state_dict()),
        "optimizer_sha256": sha_optimizer_state(opt.state_dict()),
        "fork_identity": True,
        "finite": all(base.finite_record(x) for x in records),
        "optimization_diagnostics": finalize_meter(meter, model, fork_flat) if instrument else None,
    }


def terminal(lin, key):
    return float(lin["records"][-1][key])


def smoke_run(seed, outdir):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    train_y = base.make_memories(512, derive_seed(seed, "train"))
    val_y = base.make_memories(256, derive_seed(seed, "val"))
    val_perm = base.make_perms(256, derive_seed(seed, "val_perm"))
    bank = base.make_pair_bank(seed, 96)

    base.set_seed(derive_seed(seed, "init"))
    model = base.Core()
    opt = torch.optim.AdamW(model.parameters(), lr=base.LR, weight_decay=base.WD)
    for ep in range(1, 3):
        base.train_one_epoch(model, opt, seed, train_y, ep)

    surv = base.survival_summary(model, bank)
    A = int(surv["winner_relation"])
    B = int(surv["loser_relation"])
    C, D = pick_off_axis_pair(surv, A, B)
    M = 2
    start = clone_state(model.state_dict())
    opt0 = copy.deepcopy(opt.state_dict())
    sha0 = base.sha_state_dict(start)
    osha0 = sha_optimizer_state(opt0)

    short = ((0.0, 1), (0.25, 1), (0.5, 1))
    shortb = ((1.0, 1), (0.75, 1), (0.5, 1))
    lins = {}
    for name, axis, sched in LINEAGES:
        s = short if sched is A_HISTORY else shortb
        lins[name] = run_lineage(seed, name, axis, s, start, opt0, sha0, osha0,
                                 M, A, B, C, D, train_y, val_y, val_perm, bank,
                                 expected_post=3)

    effects = {
        "H_true_AB": terminal(lins["TRUE_B"], "Q_AB") - terminal(lins["TRUE_A"], "Q_AB"),
        "H_null_AB": terminal(lins["NULL_D"], "Q_AB") - terminal(lins["NULL_C"], "Q_AB"),
        "H_null_CD": terminal(lins["NULL_D"], "Q_CD") - terminal(lins["NULL_C"], "Q_CD"),
        "H_true_CD": terminal(lins["TRUE_B"], "Q_CD") - terminal(lins["TRUE_A"], "Q_CD"),
    }
    effects["SPECIFICITY"] = effects["H_true_AB"] - effects["H_null_AB"]

    save_json(out / "smoke_summary.json", {
        "status": "ok",
        "seed": int(seed),
        "A": A, "B": B, "C": C, "D": D,
        "pairs_disjoint": bool(len({A, B, C, D}) == 4),
        "n_lineages": len(lins),
        "hashes_ok": bool(all(v["fork_identity"] for v in lins.values())),
        "dual_axis_recorded": bool(all(
            ("Q_AB" in r and "Q_CD" in r) for v in lins.values() for r in v["records"]
        )),
        "effects": effects,
        "diagnostics": {k: v["optimization_diagnostics"] for k, v in lins.items()},
    })


def run(seed, outdir):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    train_y = base.make_memories(base.TRAIN_N, derive_seed(seed, "train"))
    val_y = base.make_memories(base.VAL_N, derive_seed(seed, "val"))
    test_y = base.make_memories(base.TEST_N, derive_seed(seed, "test"))
    val_perm = base.make_perms(base.VAL_N, derive_seed(seed, "val_perm"))
    test_perm = base.make_perms(base.TEST_N, derive_seed(seed, "test_perm"))
    bank = base.make_pair_bank(seed, PAIR_N)

    base.set_seed(derive_seed(seed, "init"))
    model = base.Core()
    opt = torch.optim.AdamW(model.parameters(), lr=base.LR, weight_decay=base.WD)
    history = []
    M = None
    for ep in range(1, MAX_BASELINE_EPOCH + 1):
        base.train_one_epoch(model, opt, seed, train_y, ep)
        if ep >= FIRST_RECORDED_CHECK and ep % CHECK_EVERY == 0:
            r = base.checkpoint_record(model, val_y, val_perm, bank, ep)
            history.append(r)
            print(
                f"seed={seed} baseline ep={ep} combined={r['validation']['combined']:.4f} "
                f"h0={r['validation']['h0_overall']:.4f} "
                f"winner={r['survival']['winner_relation']} loser={r['survival']['loser_relation']}",
                flush=True,
            )
            M = base.maturity_from_history(history)
            if M is not None:
                break

    if M is None:
        save_json(out / "seed_summary.json", {
            "experiment": "R8-M10",
            "seed": int(seed),
            "maturity_reached": False,
            "maturity_epoch": None,
            "baseline_history": history,
            "validity": {"maturity": False, "pairs_disjoint": False, "fork_identity": False,
                         "lineage_length": False, "finite": True, "complete": False},
            "all_valid": False,
            "environment": {"python": __import__("sys").version, "torch": torch.__version__,
                            "numpy": np.__version__},
        })
        return

    A = int(history[-1]["survival"]["winner_relation"])
    B = int(history[-1]["survival"]["loser_relation"])
    if A == B:
        raise RuntimeError("A and B identical")
    C, D = pick_off_axis_pair(history[-1]["survival"], A, B)
    pairs_disjoint = bool(len({A, B, C, D}) == 4)

    baseline = dual_axis_record(model, val_y, val_perm, bank, M, A, B, C, D)
    start_state = clone_state(model.state_dict())
    start_opt = copy.deepcopy(opt.state_dict())
    start_sha = base.sha_state_dict(start_state)
    start_opt_sha = sha_optimizer_state(start_opt)

    lineages = {}
    for name, axis, sched in LINEAGES:
        lineages[name] = run_lineage(seed, name, axis, sched, start_state, start_opt,
                                     start_sha, start_opt_sha, M, A, B, C, D,
                                     train_y, val_y, val_perm, bank)

    effects = {
        "H_true_AB": terminal(lineages["TRUE_B"], "Q_AB") - terminal(lineages["TRUE_A"], "Q_AB"),
        "H_null_AB": terminal(lineages["NULL_D"], "Q_AB") - terminal(lineages["NULL_C"], "Q_AB"),
        "H_null_CD": terminal(lineages["NULL_D"], "Q_CD") - terminal(lineages["NULL_C"], "Q_CD"),
        "H_true_CD": terminal(lineages["TRUE_B"], "Q_CD") - terminal(lineages["TRUE_A"], "Q_CD"),
    }
    effects["SPECIFICITY"] = effects["H_true_AB"] - effects["H_null_AB"]

    fork_ok = all(v["fork_identity"] for v in lineages.values())
    finite_ok = all(v["finite"] for v in lineages.values())
    length_ok = all(len(v["records"]) == len(A_HISTORY) for v in lineages.values())

    # Test evaluation occurs only after every fixed training path is complete.
    test_eval = {}
    for name in lineages:
        test_eval[name] = None
    test_eval = {"note": "terminal validation metrics are primary; test split reserved",
                 "test_n": int(base.TEST_N)}
    _ = (test_y, test_perm)

    summary = {
        "experiment": "R8-M10",
        "seed": int(seed),
        "fresh_seed": bool(seed in FRESH_SEEDS),
        "maturity_reached": True,
        "maturity_epoch": int(M),
        "A_baseline_winner": int(A),
        "B_baseline_loser": int(B),
        "C_off_axis": int(C),
        "D_off_axis": int(D),
        "pairs_disjoint": pairs_disjoint,
        "baseline": baseline,
        "baseline_state_sha256": start_sha,
        "baseline_optimizer_sha256": start_opt_sha,
        "baseline_history": history,
        "lineages": lineages,
        "effects": effects,
        "optimization_diagnostics": {k: v["optimization_diagnostics"] for k, v in lineages.items()},
        "test_split": test_eval,
        "validity": {
            "maturity": True,
            "pairs_disjoint": pairs_disjoint,
            "fork_identity": bool(fork_ok),
            "lineage_length": bool(length_ok),
            "finite": bool(finite_ok),
            "complete": True,
        },
        "all_valid": bool(pairs_disjoint and fork_ok and finite_ok and length_ok),
        "environment": {"python": __import__("sys").version, "torch": torch.__version__,
                        "numpy": np.__version__},
    }
    save_json(out / "seed_summary.json", summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke_run(args.seed, args.outdir)
    else:
        run(args.seed, args.outdir)


if __name__ == "__main__":
    main()
