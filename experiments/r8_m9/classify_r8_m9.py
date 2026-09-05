import argparse
import json
from pathlib import Path

import numpy as np

SEEDS = (631, 648, 664, 683, 701, 718, 736, 754, 771, 789, 806, 824)
LEVELS = (0.00, 0.25, 0.50, 0.75, 1.00)
N_BOOT = 5000
BOOT_SEED = 20260905

# Preregistered thresholds. TRUE-arm gates reproduce R8-M8 exactly.
TRUE_H_MIN = 0.50
TRUE_AREA_MIN = 0.25
TRUE_HOLD_MIN = 0.25
# Null arm must be materially smaller than the true arm on the primary statistic.
CONTRAST_MIN = 0.50


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def bootstrap_mean_ci(x, seed):
    """Deterministic paired-family bootstrap, matching the R8-M8 procedure."""
    x = np.asarray(x, dtype=np.float64)
    g = np.random.default_rng(seed)
    draws = np.empty(N_BOOT, dtype=np.float64)
    for i in range(N_BOOT):
        ix = g.integers(0, len(x), len(x))
        draws[i] = x[ix].mean()
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def bootstrap_paired_diff_ci(a, b, seed):
    """CI on mean(a - b) resampling families jointly, preserving the pairing."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    g = np.random.default_rng(seed)
    draws = np.empty(N_BOOT, dtype=np.float64)
    for i in range(N_BOOT):
        ix = g.integers(0, len(a), len(a))
        draws[i] = (a[ix] - b[ix]).mean()
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def load_summaries(root):
    by_seed = {}
    for p in Path(root).rglob("seed_summary.json"):
        d = json.loads(p.read_text())
        if d.get("experiment") == "R8-M9":
            by_seed[int(d["seed"])] = d
    missing = [s for s in SEEDS if s not in by_seed]
    if missing:
        raise RuntimeError(f"missing R8-M9 seed summaries: {missing}")
    return [by_seed[s] for s in SEEDS]


def rec_at_lambda(arm, branch, lam):
    for r in arm["sweeps"][branch]["records"]:
        if abs(float(r["lambda"]) - float(lam)) < 1e-10:
            return r
    raise KeyError((branch, lam))


def hold_rec(arm, branch, hold_epoch):
    for r in arm["holds"][branch]["records"]:
        if int(r["hold_epoch"]) == int(hold_epoch):
            return r
    raise KeyError((branch, hold_epoch))


def arm_stats(arm, fwd, rev):
    """H(lambda) = Q_rev(lambda) - Q_fwd(lambda), where rev is the branch that
    starts at lambda=1. Identical construction in both arms."""
    h_by_level = {}
    for lam in LEVELS:
        q_f = float(rec_at_lambda(arm, fwd, lam)["Q"])
        q_r = float(rec_at_lambda(arm, rev, lam)["Q"])
        h_by_level[f"{lam:.2f}"] = q_r - q_f
    xs = np.asarray(LEVELS, dtype=np.float64)
    ys = np.asarray([h_by_level[f"{l:.2f}"] for l in LEVELS], dtype=np.float64)
    area = float(np.trapezoid(ys, xs)) if hasattr(np, "trapezoid") else float(np.trapz(ys, xs))
    hold = {}
    for he in (30, 60, 90, 120):
        q_f = float(hold_rec(arm, f"{fwd}_HOLD", he)["Q"])
        q_r = float(hold_rec(arm, f"{rev}_HOLD", he)["Q"])
        hold[str(he)] = q_r - q_f
    return {
        "H_mid": h_by_level["0.50"],
        "H_by_lambda": h_by_level,
        "AREA": area,
        "H_hold120": hold["120"],
        "hold_curve": hold,
    }


def classify(summaries):
    maturity_bad, execution_bad = [], []
    for d in summaries:
        v = d.get("validity", {})
        if not bool(v.get("maturity")):
            maturity_bad.append({"seed": int(d["seed"]), "validity": v})
            continue
        other = [v.get("baseline_fork_identity"), v.get("midpoint_hold_fork_identity"),
                 v.get("finite"), v.get("complete")]
        if not all(bool(x) for x in other):
            execution_bad.append({"seed": int(d["seed"]), "validity": v})

    if maturity_bad:
        return {"classification": "V0 — maturity validity failure",
                "all_valid": False, "maturity_failures": maturity_bad}
    if execution_bad:
        return {"classification": "V1 — post-maturity execution validity failure",
                "all_valid": False, "execution_failures": execution_bad}

    rows = []
    for d in summaries:
        t = arm_stats(d["arms"]["TRUE"], "A_SWEEP", "B_SWEEP")
        n = arm_stats(d["arms"]["NULL"], "C_SWEEP", "D_SWEEP")
        rows.append({
            "seed": int(d["seed"]),
            "M": int(d["maturity_epoch"]),
            "A": int(d["A_baseline_winner"]),
            "B": int(d["B_baseline_loser"]),
            "C": int(d["C_null_first"]),
            "D": int(d["D_null_second"]),
            "baseline_Q": float(d["baseline"]["Q"]),
            "TRUE": t,
            "NULL": n,
            "CONTRAST_mid": t["H_mid"] - n["H_mid"],
            "CONTRAST_hold120": t["H_hold120"] - n["H_hold120"],
        })

    t_mid = [r["TRUE"]["H_mid"] for r in rows]
    n_mid = [r["NULL"]["H_mid"] for r in rows]
    t_area = [r["TRUE"]["AREA"] for r in rows]
    n_area = [r["NULL"]["AREA"] for r in rows]
    t_hold = [r["TRUE"]["H_hold120"] for r in rows]
    n_hold = [r["NULL"]["H_hold120"] for r in rows]
    contrast = [r["CONTRAST_mid"] for r in rows]

    stats = {
        "TRUE_H_mid": {"mean": float(np.mean(t_mid)), "median": float(np.median(t_mid)),
                       "ci95": bootstrap_mean_ci(t_mid, BOOT_SEED)},
        "NULL_H_mid": {"mean": float(np.mean(n_mid)), "median": float(np.median(n_mid)),
                       "ci95": bootstrap_mean_ci(n_mid, BOOT_SEED + 1)},
        "TRUE_AREA": {"mean": float(np.mean(t_area)), "median": float(np.median(t_area)),
                      "ci95": bootstrap_mean_ci(t_area, BOOT_SEED + 2)},
        "NULL_AREA": {"mean": float(np.mean(n_area)), "median": float(np.median(n_area)),
                      "ci95": bootstrap_mean_ci(n_area, BOOT_SEED + 3)},
        "TRUE_H_hold120": {"mean": float(np.mean(t_hold)), "median": float(np.median(t_hold)),
                           "ci95": bootstrap_mean_ci(t_hold, BOOT_SEED + 4)},
        "NULL_H_hold120": {"mean": float(np.mean(n_hold)), "median": float(np.median(n_hold)),
                           "ci95": bootstrap_mean_ci(n_hold, BOOT_SEED + 5)},
        "CONTRAST_mid": {"mean": float(np.mean(contrast)), "median": float(np.median(contrast)),
                         "ci95": bootstrap_paired_diff_ci(t_mid, n_mid, BOOT_SEED + 6)},
    }

    # Distribution reporting is mandatory and does not depend on the classification.
    # R8-M8 produced a bimodal H_mid whose mean described no observed family.
    def dist(x):
        x = np.asarray(x, dtype=np.float64)
        return {
            "n_positive": int((x > 0).sum()),
            "n": int(x.size),
            "min": float(x.min()),
            "max": float(x.max()),
            "per_seed": [float(v) for v in x],
        }

    distribution = {
        "TRUE_H_mid": dist(t_mid),
        "NULL_H_mid": dist(n_mid),
        "CONTRAST_mid": dist(contrast),
    }

    true_replicates = bool(
        stats["TRUE_H_mid"]["mean"] >= TRUE_H_MIN
        and stats["TRUE_H_mid"]["ci95"][0] > 0
        and stats["TRUE_AREA"]["mean"] >= TRUE_AREA_MIN
        and stats["TRUE_AREA"]["ci95"][0] > 0
        and stats["TRUE_H_hold120"]["mean"] >= TRUE_HOLD_MIN
        and stats["TRUE_H_hold120"]["ci95"][0] > 0
    )
    null_is_flat = bool(stats["NULL_H_mid"]["ci95"][0] <= 0)
    contrast_supported = bool(
        stats["CONTRAST_mid"]["mean"] >= CONTRAST_MIN
        and stats["CONTRAST_mid"]["ci95"][0] > 0
    )

    if not true_replicates:
        classification = "N0 — true-arm replication failure; null contrast uninterpretable"
    elif contrast_supported and null_is_flat:
        classification = "N3 — history specificity supported; M8 effect is not optimizer inertia"
    elif contrast_supported:
        classification = "N2 — partial specificity; true arm exceeds a nonzero null baseline"
    else:
        classification = "N1 — null reproduces the effect; M8 separation is consistent with optimizer inertia"

    return {
        "classification": classification,
        "all_valid": True,
        "true_arm_replicates_m8": true_replicates,
        "null_is_flat": null_is_flat,
        "contrast_supported": contrast_supported,
        "stats": stats,
        "distribution": distribution,
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    result = classify(load_summaries(args.root))
    save_json(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2, default=str))


if __name__ == "__main__":
    main()
