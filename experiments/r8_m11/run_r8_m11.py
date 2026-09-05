import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

import m7r_base as base
import m9_base as m9

FRESH_SEEDS = (839, 856, 872, 891, 907, 926, 944, 963, 981, 1003, 1021, 1042)
CHECK_EVERY = 10
FIRST_RECORDED_CHECK = 40
MAX_BASELINE_EPOCH = 400
PAIR_N = 2048
A_HISTORY = ((0.00, 60), (0.25, 30), (0.50, 30))
B_HISTORY = ((1.00, 60), (0.75, 30), (0.50, 30))
MIDPOINT_POST_EPOCH = 120

ENC_PREFIXES = ("rel_emb.", "val_emb.", "enc.", "to_h.")
READER_PREFIXES = ("head0.", "headT.")
F1_KEYS = ("F.0.weight", "F.0.bias")
F2_KEYS = ("F.2.weight", "F.2.bias")


def derive_seed(seed, name):
    h = hashlib.sha256(f"R8-M11|{seed}|{name}".encode()).digest()
    return int.from_bytes(h[:4], "big")


base.derive_seed = derive_seed
m9.derive_seed = derive_seed
m9.base.derive_seed = derive_seed


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def clone_state(sd):
    return {k: v.detach().cpu().clone() for k, v in sd.items()}


def subblock(k):
    if k.startswith(ENC_PREFIXES):
        return "E"
    if k in F1_KEYS:
        return "F1"
    if k in F2_KEYS:
        return "F2"
    if k.startswith(READER_PREFIXES):
        return "R"
    raise RuntimeError(f"unassigned parameter key: {k}")


def subset_state(sd, block):
    return {k: v for k, v in sd.items() if subblock(k) == block}


def sha_subblock(sd, block):
    return base.sha_state_dict(subset_state(sd, block))


def compose_substructure(e_state, f1_state, f2_state, r_state):
    out = {}
    keys = set(e_state.keys())
    if not (keys == set(f1_state.keys()) == set(f2_state.keys()) == set(r_state.keys())):
        raise RuntimeError("state-key mismatch among transplant sources")
    for k in sorted(keys):
        b = subblock(k)
        src = e_state if b == "E" else f1_state if b == "F1" else f2_state if b == "F2" else r_state
        out[k] = src[k].detach().cpu().clone()
    return out


def build_hybrid(states, e_src, f1_src, f2_src, reader_src="A"):
    sd = compose_substructure(states[e_src], states[f1_src], states[f2_src], states[reader_src])
    model = base.Core()
    model.load_state_dict(sd)
    expected = {
        "E": sha_subblock(states[e_src], "E"),
        "F1": sha_subblock(states[f1_src], "F1"),
        "F2": sha_subblock(states[f2_src], "F2"),
        "R": sha_subblock(states[reader_src], "R"),
    }
    got = {b: sha_subblock(model.state_dict(), b) for b in ("E", "F1", "F2", "R")}
    if got != expected:
        raise RuntimeError(f"hybrid hash mismatch expected={expected} got={got}")
    return model, expected, got


def hybrid_name(e, f1, f2):
    return f"E{e}_F1{f1}_F2{f2}"


def param_descriptors(states):
    out = {}
    for label, key in (("F1_weight", "F.0.weight"), ("F2_weight", "F.2.weight")):
        d = states["B"][key].detach().cpu() - states["A"][key].detach().cpu()
        sv = torch.linalg.svdvals(d)
        out[label] = {
            "frobenius_delta": float(torch.linalg.vector_norm(d)),
            "singular_values": [float(x) for x in sv],
        }
    for label, key in (("F1_bias", "F.0.bias"), ("F2_bias", "F.2.bias")):
        d = states["B"][key].detach().cpu() - states["A"][key].detach().cpu()
        out[label] = {"l2_delta": float(torch.linalg.vector_norm(d))}
    return out


def compute_effects(q):
    f1_contrasts = []
    for e in ("A", "B"):
        for f2 in ("A", "B"):
            f1_contrasts.append(q[(e, "B", f2)] - q[(e, "A", f2)])

    f2_contrasts = []
    for e in ("A", "B"):
        for f1 in ("A", "B"):
            f2_contrasts.append(q[(e, f1, "B")] - q[(e, f1, "A")])

    f_total = 0.5 * (
        (q[("A", "B", "B")] - q[("A", "A", "A")]) +
        (q[("B", "B", "B")] - q[("B", "A", "A")])
    )

    i12 = 0.5 * sum(
        (q[(e, "B", "B")] - q[(e, "A", "B")]) -
        (q[(e, "B", "A")] - q[(e, "A", "A")])
        for e in ("A", "B")
    )

    f1 = float(np.mean(f1_contrasts))
    f2 = float(np.mean(f2_contrasts))
    if not math.isclose(f1 + f2, f_total, rel_tol=1e-9, abs_tol=1e-9):
        raise RuntimeError(f"effect decomposition identity failed: {f1}+{f2}!={f_total}")
    return {
        "F_total": float(f_total),
        "F1_effect": f1,
        "F2_effect": f2,
        "I12": float(i12),
        "F1_contrasts": [float(x) for x in f1_contrasts],
        "F2_contrasts": [float(x) for x in f2_contrasts],
        "F1_fraction": None if abs(f_total) < 1e-12 else float(f1 / f_total),
        "F2_fraction": None if abs(f_total) < 1e-12 else float(f2 / f_total),
    }


def evaluate_hybrids(states, y, perms, bank, epoch, A, B):
    records = {}
    q = {}
    hashes_ok = True
    for e in ("A", "B"):
        for f1 in ("A", "B"):
            for f2 in ("A", "B"):
                name = hybrid_name(e, f1, f2)
                model, expected, got = build_hybrid(states, e, f1, f2, "A")
                rec = base.checkpoint_record(model, y, perms, bank, epoch, A, B)
                records[name] = {
                    "sources": {"E": e, "F1": f1, "F2": f2, "R": "A"},
                    "record": rec,
                    "expected_hashes": expected,
                    "block_hashes": got,
                }
                q[(e, f1, f2)] = float(rec["Q"])
                hashes_ok = hashes_ok and (expected == got)
    return records, q, hashes_ok


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
    start_state = clone_state(model.state_dict())
    start_opt = copy.deepcopy(opt.state_dict())
    start_sha = base.sha_state_dict(start_state)

    ha = m9.run_history(
        seed, "A_SMOKE", ((0.0, 1), (0.25, 1), (0.5, 1)),
        start_state, start_opt, start_sha, M, A, B,
        train_y, val_y, val_perm, bank, expected_post=3,
    )
    hb = m9.run_history(
        seed, "B_SMOKE", ((1.0, 1), (0.75, 1), (0.5, 1)),
        start_state, start_opt, start_sha, M, A, B,
        train_y, val_y, val_perm, bank, expected_post=3,
    )

    states = {"A": ha["model_state"], "B": hb["model_state"]}
    hybrids, q, hashes_ok = evaluate_hybrids(states, val_y, val_perm, bank, M + 3, A, B)
    effects = compute_effects(q)

    save_json(out / "smoke_summary.json", {
        "status": "ok",
        "seed": int(seed),
        "A": A,
        "B": B,
        "n_hybrids": len(hybrids),
        "hashes_ok": bool(hashes_ok),
        "effects": effects,
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
                f"h0={rec['validation']['h0_overall']:.4f} "
                f"winner={rec['survival']['winner_relation']} loser={rec['survival']['loser_relation']}",
                flush=True,
            )
            M = base.maturity_from_history(history)
            if M is not None:
                break

    if M is None:
        save_json(out / "seed_summary.json", {
            "experiment": "R8-M11",
            "seed": int(seed),
            "fresh_seed": True,
            "maturity_reached": False,
            "maturity_epoch": None,
            "baseline_history": history,
            "validity": {
                "maturity": False,
                "fork_identity": False,
                "hybrid_hashes": False,
                "finite": True,
                "complete": False,
            },
            "all_valid": False,
            "environment": {
                "python": __import__("sys").version,
                "torch": torch.__version__,
                "numpy": np.__version__,
            },
        })
        return

    A = int(history[-1]["survival"]["winner_relation"])
    B = int(history[-1]["survival"]["loser_relation"])
    if A == B:
        raise RuntimeError("maturity winner and loser identical")

    baseline = base.checkpoint_record(model, val_y, val_perm, bank, M, A, B)
    start_state = clone_state(model.state_dict())
    start_opt = copy.deepcopy(opt.state_dict())
    start_sha = base.sha_state_dict(start_state)

    ha = m9.run_history(
        seed, "A_HISTORY", A_HISTORY, start_state, start_opt, start_sha,
        M, A, B, train_y, val_y, val_perm, bank,
    )
    hb = m9.run_history(
        seed, "B_HISTORY", B_HISTORY, start_state, start_opt, start_sha,
        M, A, B, train_y, val_y, val_perm, bank,
    )

    states = {"A": ha["model_state"], "B": hb["model_state"]}
    h_parent = float(hb["records"][-1]["Q"] - ha["records"][-1]["Q"])

    hybrids_val, q, hashes_ok = evaluate_hybrids(
        states, val_y, val_perm, bank, M + MIDPOINT_POST_EPOCH, A, B
    )
    effects = compute_effects(q)
    effects["H_parent"] = h_parent

    finite = all(base.finite_record(v["record"]) for v in hybrids_val.values())
    complete = len(hybrids_val) == 8
    fork_ok = bool(ha["fork_identity"] and hb["fork_identity"])

    hybrids_test = {}
    for e in ("A", "B"):
        for f1 in ("A", "B"):
            for f2 in ("A", "B"):
                name = hybrid_name(e, f1, f2)
                tm, expected, got = build_hybrid(states, e, f1, f2, "A")
                perf = base.eval_model(tm, test_y, test_perm)
                hybrids_test[name] = {
                    "sources": {"E": e, "F1": f1, "F2": f2, "R": "A"},
                    "h0_overall": perf["h0_overall"],
                    "h12_overall": perf["h12_overall"],
                    "h12_A": float(perf["h12_per_relation"][A]),
                    "h12_B": float(perf["h12_per_relation"][B]),
                    "h12_per_relation": perf["h12_per_relation"],
                    "expected_hashes": expected,
                    "block_hashes": got,
                }

    summary = {
        "experiment": "R8-M11",
        "seed": int(seed),
        "fresh_seed": True,
        "maturity_reached": True,
        "maturity_epoch": int(M),
        "A_baseline_winner": A,
        "B_baseline_loser": B,
        "baseline": baseline,
        "baseline_history": history,
        "baseline_state_sha256": start_sha,
        "histories": {
            "A": {k: v for k, v in ha.items() if k not in ("model_state", "optimizer_state")},
            "B": {k: v for k, v in hb.items() if k not in ("model_state", "optimizer_state")},
        },
        "subblock_hashes": {
            s: {b: sha_subblock(states[s], b) for b in ("E", "F1", "F2", "R")}
            for s in ("A", "B")
        },
        "effects": effects,
        "hybrids_validation": hybrids_val,
        "hybrids_test": hybrids_test,
        "parameter_descriptors": param_descriptors(states),
        "validity": {
            "maturity": True,
            "fork_identity": fork_ok,
            "hybrid_hashes": bool(hashes_ok),
            "finite": bool(finite),
            "complete": bool(complete),
        },
        "environment": {
            "python": __import__("sys").version,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    summary["all_valid"] = bool(all(summary["validity"].values()))
    save_json(out / "seed_summary.json", summary)


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
        raise SystemExit(f"seed {args.seed} is not preregistered for R8-M11")
    run(args.seed, args.outdir)


if __name__ == "__main__":
    main()
