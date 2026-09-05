import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import m7r_base as base

FRESH_SEEDS = (421, 438, 454, 471, 489, 506, 523, 541, 558, 574, 592, 611)
CHECK_EVERY = 10
FIRST_RECORDED_CHECK = 40
FIRST_ELIGIBLE_MATURITY = 60
MAX_BASELINE_EPOCH = 400
PAIR_N = 2048
SWEEP_LEVELS_A = ((0.00, 60), (0.25, 30), (0.50, 30), (0.75, 30), (1.00, 30))
SWEEP_LEVELS_B = ((1.00, 60), (0.75, 30), (0.50, 30), (0.25, 30), (0.00, 30))
MIDPOINT_POST_EPOCH = 120
HOLD_EPOCHS = 120
HOLD_CHECKS = (30, 60, 90, 120)


def derive_seed(seed, name):
    h = hashlib.sha256(f"R8-M8|{seed}|{name}".encode()).digest()
    return int.from_bytes(h[:4], "big")


# Reuse the frozen M7R lineage engine, but with a fresh R8-M8 deterministic namespace.
base.derive_seed = derive_seed


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def clone_state(sd):
    return {k: v.detach().cpu().clone() for k, v in sd.items()}


def task_loss_lambda(model, y, perms, A, B, lam):
    ce = nn.CrossEntropyLoss()
    _, l0, lT = model(y, perms)
    z0 = torch.stack([ce(l0[r], y[:, r]) for r in range(base.N_REL)])
    zT = torch.stack([ce(lT[r], y[:, r]) for r in range(base.N_REL)])
    w = torch.ones(base.N_REL, dtype=zT.dtype, device=zT.device)
    w[int(A)] = 1.0 + 3.0 * (1.0 - float(lam))
    w[int(B)] = 1.0 + 3.0 * float(lam)
    h0_loss = z0.mean()
    h12_loss = (zT * w).sum() / w.sum()
    return 0.5 * (h0_loss + h12_loss)


def train_one_epoch_lambda(model, opt, seed, train_y, ep, A, B, lam):
    model.train()
    train_perm = base.make_perms(len(train_y), derive_seed(seed, f"presentation_{ep}"))
    g = torch.Generator().manual_seed(derive_seed(seed, f"order_{ep}"))
    order = torch.randperm(len(train_y), generator=g)
    for a in range(0, len(train_y), base.BATCH):
        ix = order[a:a + base.BATCH]
        opt.zero_grad(set_to_none=True)
        loss = task_loss_lambda(model, train_y[ix], train_perm[ix], A, B, lam)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss seed={seed} epoch={ep} lambda={lam}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), base.CLIP)
        opt.step()


def checkpoint_with_lambda(model, val_y, val_perm, bank, epoch, A, B, lam, post_epoch):
    rec = base.checkpoint_record(model, val_y, val_perm, bank, epoch, A, B)
    rec["lambda"] = float(lam)
    rec["post_maturity_epoch"] = int(post_epoch)
    return rec


def midpoint_latent_distance(state_a, state_b, y, perms):
    ma = base.Core(); ma.load_state_dict(clone_state(state_a)); ma.eval()
    mb = base.Core(); mb.load_state_dict(clone_state(state_b)); mb.eval()
    h0_vals = []
    h12_vals = []
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


def run_sweep(seed, name, schedule, base_model_state, base_opt_state, base_sha,
              M, A, B, train_y, val_y, val_perm, bank):
    model = base.Core()
    model.load_state_dict(clone_state(base_model_state))
    opt = torch.optim.AdamW(model.parameters(), lr=base.LR, weight_decay=base.WD)
    opt.load_state_dict(copy.deepcopy(base_opt_state))
    fork_ok = base.sha_state_dict(model.state_dict()) == base_sha
    if not fork_ok:
        raise RuntimeError(f"baseline fork-state mismatch in {name}")

    records = []
    post = 0
    midpoint = None
    for lam, duration in schedule:
        for _ in range(int(duration)):
            post += 1
            train_one_epoch_lambda(model, opt, seed, train_y, M + post, A, B, lam)
        rec = checkpoint_with_lambda(model, val_y, val_perm, bank, M + post, A, B, lam, post)
        records.append(rec)
        print(
            f"seed={seed} branch={name} post={post} lambda={lam:.2f} "
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
        "fork_identity": bool(fork_ok),
        "midpoint": midpoint,
    }


def run_hold(seed, name, midpoint, M, A, B, train_y, val_y, val_perm, bank):
    model = base.Core()
    model.load_state_dict(clone_state(midpoint["model_state"]))
    opt = torch.optim.AdamW(model.parameters(), lr=base.LR, weight_decay=base.WD)
    opt.load_state_dict(copy.deepcopy(midpoint["optimizer_state"]))
    fork_ok = base.sha_state_dict(model.state_dict()) == midpoint["state_sha256"]
    if not fork_ok:
        raise RuntimeError(f"midpoint hold fork mismatch in {name}")

    records = []
    for hold_ep in range(1, HOLD_EPOCHS + 1):
        absolute_post = MIDPOINT_POST_EPOCH + hold_ep
        train_one_epoch_lambda(model, opt, seed, train_y, M + absolute_post, A, B, 0.5)
        if hold_ep in HOLD_CHECKS:
            rec = checkpoint_with_lambda(
                model, val_y, val_perm, bank,
                M + absolute_post, A, B, 0.5, absolute_post,
            )
            rec["hold_epoch"] = int(hold_ep)
            records.append(rec)
            print(
                f"seed={seed} branch={name} hold={hold_ep} Q={rec['Q']:.4f} "
                f"winner={rec['survival']['winner_relation']} h12={rec['validation']['h12_overall']:.4f}",
                flush=True,
            )

    return {
        "records": records,
        "fork_identity": bool(fork_ok),
        "final_state_sha256": base.sha_state_dict(model.state_dict()),
    }


def finite_records(records):
    return all(base.finite_record(r) for r in records)


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
    M = 2
    state = clone_state(model.state_dict())
    opt_state = copy.deepcopy(opt.state_dict())
    sha = base.sha_state_dict(state)

    smoke_sched_a = ((0.0, 1), (0.25, 1), (0.5, 1), (0.75, 1), (1.0, 1))
    smoke_sched_b = tuple(reversed(smoke_sched_a))

    # Tiny direct path exercising continuous weighting without using scientific durations.
    branches = {}
    for name, schedule in (("A_SWEEP", smoke_sched_a), ("B_SWEEP", smoke_sched_b)):
        m = base.Core(); m.load_state_dict(clone_state(state))
        o = torch.optim.AdamW(m.parameters(), lr=base.LR, weight_decay=base.WD)
        o.load_state_dict(copy.deepcopy(opt_state))
        recs = []
        post = 0
        midpoint = None
        for lam, duration in schedule:
            for _ in range(duration):
                post += 1
                train_one_epoch_lambda(m, o, seed, train_y, M + post, A, B, lam)
            r = checkpoint_with_lambda(m, val_y, val_perm, bank, M + post, A, B, lam, post)
            recs.append(r)
            if abs(lam - 0.5) < 1e-12:
                midpoint = {
                    "model_state": clone_state(m.state_dict()),
                    "optimizer_state": copy.deepcopy(o.state_dict()),
                    "state_sha256": base.sha_state_dict(m.state_dict()),
                }
        branches[name] = {"records": recs, "midpoint": midpoint}

    for name, mid in (("A_HOLD", branches["A_SWEEP"]["midpoint"]), ("B_HOLD", branches["B_SWEEP"]["midpoint"])):
        m = base.Core(); m.load_state_dict(clone_state(mid["model_state"]))
        o = torch.optim.AdamW(m.parameters(), lr=base.LR, weight_decay=base.WD)
        o.load_state_dict(copy.deepcopy(mid["optimizer_state"]))
        train_one_epoch_lambda(m, o, seed, train_y, M + 10, A, B, 0.5)
        branches[name] = {
            "Q": base.checkpoint_record(m, val_y, val_perm, bank, M + 10, A, B)["Q"],
            "fork_identity": base.sha_state_dict(mid["model_state"]) == mid["state_sha256"],
        }

    save_json(out / "smoke_summary.json", {
        "status": "ok",
        "seed": int(seed),
        "A": A,
        "B": B,
        "baseline_state_sha256": sha,
        "A_mid_Q": branches["A_SWEEP"]["records"][2]["Q"],
        "B_mid_Q": branches["B_SWEEP"]["records"][2]["Q"],
        "A_hold_Q": branches["A_HOLD"]["Q"],
        "B_hold_Q": branches["B_HOLD"]["Q"],
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
        summary = {
            "experiment": "R8-M8",
            "seed": int(seed),
            "fresh_seed": True,
            "maturity_reached": False,
            "maturity_epoch": None,
            "baseline_history": history,
            "validity": {
                "maturity": False,
                "baseline_fork_identity": False,
                "midpoint_hold_fork_identity": False,
                "finite": True,
                "complete": False,
            },
            "all_valid": False,
            "environment": {"python": __import__("sys").version, "torch": torch.__version__, "numpy": np.__version__},
        }
        save_json(out / "seed_summary.json", summary)
        return

    A = int(history[-1]["survival"]["winner_relation"])
    B = int(history[-1]["survival"]["loser_relation"])
    if A == B:
        raise RuntimeError("maturity winner and loser identical")

    baseline = base.checkpoint_record(model, val_y, val_perm, bank, M, A, B)
    base_model_state = clone_state(model.state_dict())
    base_opt_state = copy.deepcopy(opt.state_dict())
    base_sha = base.sha_state_dict(base_model_state)

    sweep_a = run_sweep(
        seed, "A_SWEEP", SWEEP_LEVELS_A, base_model_state, base_opt_state, base_sha,
        M, A, B, train_y, val_y, val_perm, bank,
    )
    sweep_b = run_sweep(
        seed, "B_SWEEP", SWEEP_LEVELS_B, base_model_state, base_opt_state, base_sha,
        M, A, B, train_y, val_y, val_perm, bank,
    )

    midpoint_distance = midpoint_latent_distance(
        sweep_a["midpoint"]["model_state"], sweep_b["midpoint"]["model_state"], val_y, val_perm,
    )

    hold_a = run_hold(
        seed, "A_HOLD", sweep_a["midpoint"], M, A, B, train_y, val_y, val_perm, bank,
    )
    hold_b = run_hold(
        seed, "B_HOLD", sweep_b["midpoint"], M, A, B, train_y, val_y, val_perm, bank,
    )

    baseline_fork_identity = bool(sweep_a["fork_identity"] and sweep_b["fork_identity"])
    midpoint_hold_fork_identity = bool(hold_a["fork_identity"] and hold_b["fork_identity"])
    finite = bool(
        finite_records(sweep_a["records"]) and finite_records(sweep_b["records"])
        and finite_records(hold_a["records"]) and finite_records(hold_b["records"])
    )
    complete = bool(
        len(sweep_a["records"]) == 5 and len(sweep_b["records"]) == 5
        and len(hold_a["records"]) == 4 and len(hold_b["records"]) == 4
    )

    # Remove raw model/optimizer snapshots from serialized branch records while preserving hashes and midpoint records.
    sweep_a_public = {
        "records": sweep_a["records"],
        "final_state_sha256": sweep_a["final_state_sha256"],
        "fork_identity": sweep_a["fork_identity"],
        "midpoint_state_sha256": sweep_a["midpoint"]["state_sha256"],
        "midpoint_record": sweep_a["midpoint"]["record"],
    }
    sweep_b_public = {
        "records": sweep_b["records"],
        "final_state_sha256": sweep_b["final_state_sha256"],
        "fork_identity": sweep_b["fork_identity"],
        "midpoint_state_sha256": sweep_b["midpoint"]["state_sha256"],
        "midpoint_record": sweep_b["midpoint"]["record"],
    }

    summary = {
        "experiment": "R8-M8",
        "seed": int(seed),
        "fresh_seed": bool(seed in FRESH_SEEDS),
        "maturity_reached": True,
        "maturity_epoch": int(M),
        "A_baseline_winner": A,
        "B_baseline_loser": B,
        "baseline": baseline,
        "baseline_history": history,
        "baseline_state_sha256": base_sha,
        "sweeps": {"A_SWEEP": sweep_a_public, "B_SWEEP": sweep_b_public},
        "holds": {"A_HOLD": hold_a, "B_HOLD": hold_b},
        "midpoint_latent_distance": midpoint_distance,
        "test_reference": eval_model_reference(model, test_y, test_perm),
        "validity": {
            "maturity": True,
            "baseline_fork_identity": baseline_fork_identity,
            "midpoint_hold_fork_identity": midpoint_hold_fork_identity,
            "finite": finite,
            "complete": complete,
        },
        "environment": {"python": __import__("sys").version, "torch": torch.__version__, "numpy": np.__version__},
    }
    summary["all_valid"] = bool(all(summary["validity"].values()))
    save_json(out / "seed_summary.json", summary)


def eval_model_reference(model, y, perms):
    return base.eval_model(model, y, perms)


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
        raise SystemExit(f"seed {args.seed} is not preregistered for R8-M8")
    run(args.seed, args.outdir)


if __name__ == "__main__":
    main()
