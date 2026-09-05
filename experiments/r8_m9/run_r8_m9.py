import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import m7r_base as base

# R8-M9 uses its own fresh seed set. No R8-M8 family is reused.
FRESH_SEEDS = (631, 648, 664, 683, 701, 718, 736, 754, 771, 789, 806, 824)
CHECK_EVERY = 10
FIRST_RECORDED_CHECK = 40
FIRST_ELIGIBLE_MATURITY = 60
MAX_BASELINE_EPOCH = 400
PAIR_N = 2048
MIDPOINT_POST_EPOCH = 120
HOLD_EPOCHS = 120
HOLD_CHECKS = (30, 60, 90, 120)

# TRUE arm: the R8-M8 demand schedules, reproduced exactly.
SWEEP_LEVELS_A = ((0.00, 60), (0.25, 30), (0.50, 30), (0.75, 30), (1.00, 30))
SWEEP_LEVELS_B = ((1.00, 60), (0.75, 30), (0.50, 30), (0.25, 30), (0.00, 30))

# NULL arm: identical lambda schedules, but lambda is routed to two relations
# that are neither the survival winner A nor the loser B. The optimizer sees the
# same weighting magnitudes on the same epochs; only the scientific meaning of
# the history differs. Q is still measured on the untouched A/B axis.
NULL_SWEEP_LEVELS_C = SWEEP_LEVELS_A
NULL_SWEEP_LEVELS_D = SWEEP_LEVELS_B


def derive_seed(seed, name):
    h = hashlib.sha256(f"R8-M9|{seed}|{name}".encode()).digest()
    return int.from_bytes(h[:4], "big")


# Reuse the frozen M7R/M8 lineage engine under a fresh R8-M9 deterministic namespace.
base.derive_seed = derive_seed


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def clone_state(sd):
    return {k: v.detach().cpu().clone() for k, v in sd.items()}


def pick_null_pair(surv, A, B, seed):
    """Deterministically choose two mid-ranked relations that are neither A nor B.

    Selecting from the middle of the survival ranking avoids handing the null arm
    either the established specialist or the weakest relation, so the null history
    is matched in optimizer magnitude while carrying no A/B demand meaning.
    """
    order = list(np.argsort(np.asarray(surv["terminal_survival"], dtype=np.float64)))
    mid = [int(r) for r in order if int(r) not in (int(A), int(B))]
    if len(mid) < 2:
        raise RuntimeError("insufficient non-A/B relations for null pair")
    g = np.random.default_rng(derive_seed(seed, "null_pair"))
    lo = len(mid) // 2 - 1
    cand = mid[max(lo, 0):max(lo, 0) + 2]
    if len(cand) < 2:
        cand = mid[:2]
    c, d = int(cand[0]), int(cand[1])
    if g.integers(0, 2) == 1:
        c, d = d, c
    return c, d


def task_loss_lambda(model, y, perms, P, Qr, lam):
    """R8-M8 loss with the demand pair generalized from (A,B) to (P,Qr)."""
    ce = nn.CrossEntropyLoss()
    _, l0, lT = model(y, perms)
    z0 = torch.stack([ce(l0[r], y[:, r]) for r in range(base.N_REL)])
    zT = torch.stack([ce(lT[r], y[:, r]) for r in range(base.N_REL)])
    w = torch.ones(base.N_REL, dtype=zT.dtype, device=zT.device)
    w[int(P)] = 1.0 + 3.0 * (1.0 - float(lam))
    w[int(Qr)] = 1.0 + 3.0 * float(lam)
    h0_loss = z0.mean()
    h12_loss = (zT * w).sum() / w.sum()
    return 0.5 * (h0_loss + h12_loss)


def train_one_epoch_lambda(model, opt, seed, train_y, ep, P, Qr, lam):
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
        nn.utils.clip_grad_norm_(model.parameters(), base.CLIP)
        opt.step()


def checkpoint_with_lambda(model, val_y, val_perm, bank, epoch, A, B, lam, post_epoch, arm):
    """Q is always measured on the baseline A/B axis, in both arms."""
    rec = base.checkpoint_record(model, val_y, val_perm, bank, epoch, A, B)
    rec["lambda"] = float(lam)
    rec["post_maturity_epoch"] = int(post_epoch)
    rec["arm"] = arm
    return rec


def run_sweep(seed, name, arm, schedule, base_model_state, base_opt_state, base_sha,
              M, A, B, P, Qr, train_y, val_y, val_perm, bank):
    model = base.Core()
    model.load_state_dict(clone_state(base_model_state))
    opt = torch.optim.AdamW(model.parameters(), lr=base.LR, weight_decay=base.WD)
    opt.load_state_dict(copy.deepcopy(base_opt_state))
    if base.sha_state_dict(model.state_dict()) != base_sha:
        raise RuntimeError(f"baseline fork-state mismatch in {name}")

    records = []
    post = 0
    midpoint = None
    for lam, duration in schedule:
        for _ in range(int(duration)):
            post += 1
            train_one_epoch_lambda(model, opt, seed, train_y, M + post, P, Qr, lam)
        rec = checkpoint_with_lambda(model, val_y, val_perm, bank, M + post, A, B, lam, post, arm)
        records.append(rec)
        print(
            f"seed={seed} arm={arm} branch={name} post={post} lambda={lam:.2f} "
            f"Q={rec['Q']:.4f} winner={rec['survival']['winner_relation']} "
            f"h12={rec['validation']['h12_overall']:.4f}",
            flush=True,
        )
        if abs(float(lam) - 0.5) < 1e-12:
            if post != MIDPOINT_POST_EPOCH:
                raise RuntimeError(f"midpoint timing mismatch in {name}: {post}")
            midpoint = {
                "model_state": clone_state(model.state_dict()),
                "optimizer_state": copy.deepcopy(opt.state_dict()),
                "state_sha256": base.sha_state_dict(model.state_dict()),
                "record": copy.deepcopy(rec),
            }

    if midpoint is None:
        raise RuntimeError(f"missing midpoint snapshot in {name}")
    return {
        "records": records,
        "final_state_sha256": base.sha_state_dict(model.state_dict()),
        "fork_identity": True,
        "midpoint": midpoint,
    }


def run_hold(seed, name, arm, midpoint, M, A, B, P, Qr, train_y, val_y, val_perm, bank):
    model = base.Core()
    model.load_state_dict(clone_state(midpoint["model_state"]))
    opt = torch.optim.AdamW(model.parameters(), lr=base.LR, weight_decay=base.WD)
    opt.load_state_dict(copy.deepcopy(midpoint["optimizer_state"]))
    if base.sha_state_dict(model.state_dict()) != midpoint["state_sha256"]:
        raise RuntimeError(f"midpoint hold fork mismatch in {name}")

    records = []
    for hold_ep in range(1, HOLD_EPOCHS + 1):
        absolute_post = MIDPOINT_POST_EPOCH + hold_ep
        train_one_epoch_lambda(model, opt, seed, train_y, M + absolute_post, P, Qr, 0.5)
        if hold_ep in HOLD_CHECKS:
            rec = checkpoint_with_lambda(
                model, val_y, val_perm, bank, M + absolute_post, A, B, 0.5, absolute_post, arm,
            )
            rec["hold_epoch"] = int(hold_ep)
            records.append(rec)
            print(
                f"seed={seed} arm={arm} branch={name} hold={hold_ep} Q={rec['Q']:.4f} "
                f"winner={rec['survival']['winner_relation']} h12={rec['validation']['h12_overall']:.4f}",
                flush=True,
            )

    return {
        "records": records,
        "fork_identity": True,
        "final_state_sha256": base.sha_state_dict(model.state_dict()),
    }


def run_arm(seed, arm, sched_first, sched_second, first_name, second_name,
            base_model_state, base_opt_state, base_sha, M, A, B, P, Qr,
            train_y, val_y, val_perm, bank):
    s1 = run_sweep(seed, first_name, arm, sched_first, base_model_state, base_opt_state,
                   base_sha, M, A, B, P, Qr, train_y, val_y, val_perm, bank)
    s2 = run_sweep(seed, second_name, arm, sched_second, base_model_state, base_opt_state,
                   base_sha, M, A, B, P, Qr, train_y, val_y, val_perm, bank)
    h1 = run_hold(seed, f"{first_name}_HOLD", arm, s1["midpoint"], M, A, B, P, Qr,
                  train_y, val_y, val_perm, bank)
    h2 = run_hold(seed, f"{second_name}_HOLD", arm, s2["midpoint"], M, A, B, P, Qr,
                  train_y, val_y, val_perm, bank)

    def public(s):
        return {
            "records": s["records"],
            "final_state_sha256": s["final_state_sha256"],
            "fork_identity": s["fork_identity"],
            "midpoint_state_sha256": s["midpoint"]["state_sha256"],
            "midpoint_record": s["midpoint"]["record"],
        }

    return {
        "sweeps": {first_name: public(s1), second_name: public(s2)},
        "holds": {f"{first_name}_HOLD": h1, f"{second_name}_HOLD": h2},
        "raw_midpoints": {first_name: s1["midpoint"], second_name: s2["midpoint"]},
        "complete": bool(
            len(s1["records"]) == 5 and len(s2["records"]) == 5
            and len(h1["records"]) == 4 and len(h2["records"]) == 4
        ),
        "finite": bool(all(
            base.finite_record(r)
            for r in s1["records"] + s2["records"] + h1["records"] + h2["records"]
        )),
    }


def midpoint_latent_distance(state_a, state_b, y, perms):
    ma = base.Core(); ma.load_state_dict(clone_state(state_a)); ma.eval()
    mb = base.Core(); mb.load_state_dict(clone_state(state_b)); mb.eval()
    h0_vals, h12_vals = [], []
    with torch.no_grad():
        for a in range(0, len(y), base.BATCH):
            yy = y[a:a + base.BATCH]
            pp = perms[a:a + base.BATCH]
            h0a = ma.encode(yy, pp)
            h0b = mb.encode(yy, pp)
            ta = ma.trajectory(h0a)
            tb = mb.trajectory(h0b)
            h0_vals.append(torch.linalg.vector_norm(h0a - h0b, dim=1).cpu())
            h12_vals.append(torch.linalg.vector_norm(ta[:, -1] - tb[:, -1], dim=1).cpu())
    return {
        "mean_h0_distance": float(torch.cat(h0_vals).mean()),
        "mean_h12_distance": float(torch.cat(h12_vals).mean()),
    }


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
            rec = base.checkpoint_record(model, val_y, val_perm, bank, ep)
            history.append(rec)
            print(
                f"seed={seed} baseline ep={ep} combined={rec['validation']['combined']:.4f} "
                f"h0={rec['validation']['h0_overall']:.4f} winner={rec['survival']['winner_relation']} "
                f"loser={rec['survival']['loser_relation']} competent={rec['competent']}",
                flush=True,
            )
            M = base.maturity_from_history(history)
            if M is not None:
                break

    if M is None:
        save_json(out / "seed_summary.json", {
            "experiment": "R8-M9",
            "seed": int(seed),
            "fresh_seed": bool(seed in FRESH_SEEDS),
            "maturity_reached": False,
            "maturity_epoch": None,
            "baseline_history": history,
            "validity": {"maturity": False, "baseline_fork_identity": False,
                         "midpoint_hold_fork_identity": False, "finite": True, "complete": False},
            "all_valid": False,
            "environment": {"python": __import__("sys").version,
                            "torch": torch.__version__, "numpy": np.__version__},
        })
        return

    surv = history[-1]["survival"]
    A = int(surv["winner_relation"])
    B = int(surv["loser_relation"])
    if A == B:
        raise RuntimeError("maturity winner and loser identical")
    C, D = pick_null_pair(surv, A, B, seed)

    baseline = base.checkpoint_record(model, val_y, val_perm, bank, M, A, B)
    base_model_state = clone_state(model.state_dict())
    base_opt_state = copy.deepcopy(opt.state_dict())
    base_sha = base.sha_state_dict(base_model_state)

    true_arm = run_arm(
        seed, "TRUE", SWEEP_LEVELS_A, SWEEP_LEVELS_B, "A_SWEEP", "B_SWEEP",
        base_model_state, base_opt_state, base_sha, M, A, B, A, B,
        train_y, val_y, val_perm, bank,
    )
    null_arm = run_arm(
        seed, "NULL", NULL_SWEEP_LEVELS_C, NULL_SWEEP_LEVELS_D, "C_SWEEP", "D_SWEEP",
        base_model_state, base_opt_state, base_sha, M, A, B, C, D,
        train_y, val_y, val_perm, bank,
    )

    true_dist = midpoint_latent_distance(
        true_arm["raw_midpoints"]["A_SWEEP"]["model_state"],
        true_arm["raw_midpoints"]["B_SWEEP"]["model_state"], val_y, val_perm)
    null_dist = midpoint_latent_distance(
        null_arm["raw_midpoints"]["C_SWEEP"]["model_state"],
        null_arm["raw_midpoints"]["D_SWEEP"]["model_state"], val_y, val_perm)

    for arm in (true_arm, null_arm):
        arm.pop("raw_midpoints", None)

    summary = {
        "experiment": "R8-M9",
        "seed": int(seed),
        "fresh_seed": bool(seed in FRESH_SEEDS),
        "maturity_reached": True,
        "maturity_epoch": int(M),
        "A_baseline_winner": A,
        "B_baseline_loser": B,
        "C_null_first": C,
        "D_null_second": D,
        "baseline": baseline,
        "baseline_history": history,
        "baseline_state_sha256": base_sha,
        "arms": {"TRUE": true_arm, "NULL": null_arm},
        "midpoint_latent_distance": {"TRUE": true_dist, "NULL": null_dist},
        "test_reference": base.eval_model(model, test_y, test_perm),
        "validity": {
            "maturity": True,
            "baseline_fork_identity": True,
            "midpoint_hold_fork_identity": True,
            "finite": bool(true_arm["finite"] and null_arm["finite"]),
            "complete": bool(true_arm["complete"] and null_arm["complete"]),
        },
        "environment": {"python": __import__("sys").version,
                        "torch": torch.__version__, "numpy": np.__version__},
    }
    summary["all_valid"] = bool(all(summary["validity"].values()))
    save_json(out / "seed_summary.json", summary)


def smoke_run(seed, outdir):
    """Tiny end-to-end path. Not scientific: durations and data sizes are reduced."""
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
    C, D = pick_null_pair(surv, A, B, seed)
    M = 2
    state = clone_state(model.state_dict())
    opt_state = copy.deepcopy(opt.state_dict())

    sched = ((0.0, 1), (0.5, 1), (1.0, 1))
    qs = {}
    for arm, P, Qr in (("TRUE", A, B), ("NULL", C, D)):
        for name, sc in (("FWD", sched), ("REV", tuple(reversed(sched)))):
            m = base.Core(); m.load_state_dict(clone_state(state))
            o = torch.optim.AdamW(m.parameters(), lr=base.LR, weight_decay=base.WD)
            o.load_state_dict(copy.deepcopy(opt_state))
            post = 0
            for lam, dur in sc:
                for _ in range(dur):
                    post += 1
                    train_one_epoch_lambda(m, o, seed, train_y, M + post, P, Qr, lam)
            qs[f"{arm}_{name}_Q"] = base.checkpoint_record(
                m, val_y, val_perm, bank, M + post, A, B)["Q"]

    save_json(out / "smoke_summary.json", {
        "status": "ok", "seed": int(seed), "A": A, "B": B, "C": C, "D": D,
        "H_true_smoke": qs["TRUE_REV_Q"] - qs["TRUE_FWD_Q"],
        "H_null_smoke": qs["NULL_REV_Q"] - qs["NULL_FWD_Q"],
        **qs,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke_run(args.seed, args.outdir)
        return
    if args.seed not in FRESH_SEEDS:
        raise SystemExit(f"seed {args.seed} is not preregistered for R8-M9")
    run(args.seed, args.outdir)


if __name__ == "__main__":
    main()
