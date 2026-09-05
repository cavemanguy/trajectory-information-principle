import argparse
import json
from pathlib import Path

import numpy as np

SEEDS = (1061, 1078, 1094, 1113, 1129, 1146, 1164, 1183, 1201, 1218, 1236, 1254)
N_BOOT = 5000

# Frozen gates, per PREREGISTRATION.md section 5.
REPLICATION_MIN = 0.50      # R  : mean H_true_AB
MANIP_CHECK_MIN = 0.50      # MC : mean H_null_CD
SEPARATION_MIN = 0.50       # SEP: mean SPECIFICITY
EQUIV_DELTA = 0.25          # FLAT: entire CI of H_null_AB inside (-DELTA, +DELTA)

PRIMARY = ("H_true_AB", "H_null_AB", "H_null_CD", "H_true_CD", "SPECIFICITY")


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def bootstrap_mean_ci(x, seed):
    """Resample families. Arms are paired within family and never resampled apart."""
    x = np.asarray(x, dtype=np.float64)
    g = np.random.default_rng(seed)
    draws = np.empty(N_BOOT, dtype=np.float64)
    for i in range(N_BOOT):
        ix = g.integers(0, len(x), len(x))
        draws[i] = x[ix].mean()
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def distribution(x):
    """Mandatory per PREREGISTRATION.md section 6. M8's mean described no observed family."""
    a = np.asarray(x, dtype=np.float64)
    return {
        "values": [float(v) for v in a],
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "min": float(a.min()),
        "max": float(a.max()),
        "n_positive": int(np.sum(a > 0)),
        "n": int(a.size),
    }


def stat(x, seed):
    d = distribution(x)
    d["ci95"] = bootstrap_mean_ci(x, seed)
    return d


def load_summaries(root):
    by_seed = {}
    for p in Path(root).rglob("seed_summary.json"):
        d = json.loads(p.read_text())
        if d.get("experiment") == "R8-M10":
            by_seed[int(d["seed"])] = d
    missing = [s for s in SEEDS if s not in by_seed]
    if missing:
        raise RuntimeError(f"missing M10 seed summaries: {missing}")
    return [by_seed[s] for s in SEEDS]


def is_flat(ci, delta=EQUIV_DELTA):
    """Two-sided equivalence: the ENTIRE interval must lie inside the margin.

    A CI-lower-bound-below-zero rule would call a clearly negative effect flat.
    This does not.
    """
    return bool(ci[0] > -float(delta) and ci[1] < float(delta))


def classify(summaries):
    maturity_bad = []
    execution_bad = []
    for d in summaries:
        v = d.get("validity", {})
        if not bool(v.get("maturity")):
            maturity_bad.append(int(d["seed"]))
        elif not all(bool(v.get(k)) for k in
                     ("pairs_disjoint", "fork_identity", "lineage_length", "finite", "complete")):
            execution_bad.append({"seed": int(d["seed"]), "validity": v})
    if maturity_bad:
        return {"classification": "V0 — maturity validity failure",
                "all_valid": False, "maturity_failures": maturity_bad}
    if execution_bad:
        return {"classification": "V1 — post-maturity lineage execution failure",
                "all_valid": False, "execution_failures": execution_bad}

    rows = []
    for d in summaries:
        e = d["effects"]
        row = {
            "seed": int(d["seed"]),
            "M": int(d["maturity_epoch"]),
            "A": int(d["A_baseline_winner"]),
            "B": int(d["B_baseline_loser"]),
            "C": int(d["C_off_axis"]),
            "D": int(d["D_off_axis"]),
        }
        for k in PRIMARY:
            row[k] = float(e[k])
        diag = d.get("optimization_diagnostics") or {}
        for name in ("TRUE_A", "TRUE_B", "NULL_C", "NULL_D"):
            dd = diag.get(name) or {}
            row[f"update_mean_{name}"] = float(dd.get("update_norm_mean", float("nan")))
            row[f"grad_mean_{name}"] = float(dd.get("grad_norm_mean", float("nan")))
            row[f"dist_{name}"] = float(dd.get("param_distance_from_fork", float("nan")))
        rows.append(row)

    stats = {k: stat([r[k] for r in rows], 84100 + i) for i, k in enumerate(PRIMARY)}

    # Secondary diagnostics — reported, never gating.
    diagnostics = {}
    for i, name in enumerate(("TRUE_A", "TRUE_B", "NULL_C", "NULL_D")):
        diagnostics[name] = {
            "update_norm_mean": stat([r[f"update_mean_{name}"] for r in rows], 84200 + i * 3),
            "grad_norm_mean": stat([r[f"grad_mean_{name}"] for r in rows], 84201 + i * 3),
            "param_distance_from_fork": stat([r[f"dist_{name}"] for r in rows], 84202 + i * 3),
        }
    diagnostics["note"] = (
        "Identical loss weights on different relation pairs do not guarantee identical "
        "optimization pressure. These diagnostics let a reader judge whether an observed "
        "null-arm flatness could instead reflect the null arm being pushed less far. "
        "They are not gates."
    )

    R = bool(stats["H_true_AB"]["mean"] >= REPLICATION_MIN and stats["H_true_AB"]["ci95"][0] > 0)
    if not R:
        return {
            "classification": "S0 — true-arm replication failure; off-axis contrast uninterpretable",
            "all_valid": True, "R_supported": False,
            "gates": {"R": False},
            "stats": stats, "diagnostics": diagnostics, "rows": rows,
            "claim_boundary": (
                "No specificity claim is promoted because the fresh A/B history effect missed "
                "the frozen replication gate."
            ),
        }

    MC = bool(stats["H_null_CD"]["mean"] >= MANIP_CHECK_MIN and stats["H_null_CD"]["ci95"][0] > 0)
    if not MC:
        return {
            "classification": "S1 — off-axis manipulation ineffective; specificity untestable",
            "all_valid": True, "R_supported": True, "MC_supported": False,
            "gates": {"R": True, "MC": False},
            "stats": stats, "diagnostics": diagnostics, "rows": rows,
            "claim_boundary": (
                "The C/D demand history did not reorganize the C/D axis, so a flat A/B response "
                "in the null arm cannot be read as axis specificity."
            ),
        }

    SEP = bool(stats["SPECIFICITY"]["mean"] >= SEPARATION_MIN and stats["SPECIFICITY"]["ci95"][0] > 0)
    FLAT = is_flat(stats["H_null_AB"]["ci95"])

    if SEP and FLAT:
        classification = "S2 — strong axis specificity supported"
        boundary = (
            "Within this synthetic recurrent system, persistent A/B reorganization under matched "
            "present demand is specific to the historically demanded axis: an equally weighted "
            "off-axis demand history reorganized its own axis without producing a comparable A/B "
            "effect. This does not establish bistability, hysteresis, or generalization beyond "
            "this system."
        )
    elif SEP:
        classification = "S3 — partial specificity; true arm exceeds a nonzero off-axis baseline"
        boundary = (
            "The demanded axis reorganizes more than an arbitrary off-axis history does, but the "
            "off-axis history is not flat on A/B, so part of the effect is generic post-fork "
            "divergence."
        )
    else:
        classification = "S4 — no specificity detected; off-axis history produces comparable A/B reorganization"
        boundary = (
            "Arbitrary off-axis demand history produced A/B reorganization comparable to the "
            "demanded-axis history. This substantially weakens the axis-specific reading of R8-M8 "
            "and is a frozen outcome, not to be repaired or re-run."
        )

    return {
        "classification": classification,
        "all_valid": True,
        "R_supported": True,
        "MC_supported": True,
        "SEP_supported": SEP,
        "NULL_flat_by_equivalence": FLAT,
        "equivalence_margin": EQUIV_DELTA,
        "gates": {"R": True, "MC": True, "SEP": SEP, "FLAT": FLAT},
        "stats": stats,
        "diagnostics": diagnostics,
        "rows": rows,
        "claim_boundary": boundary,
    }


def write_md(path, r):
    lines = ["# R8-M10 Final Result — Off-Axis History Specificity Control", "",
             f"**Primary classification:** {r['classification']}", ""]
    if not r.get("all_valid", False):
        lines.append(f"- Validity record: {r}")
    else:
        g = r.get("gates", {})
        lines += ["## Frozen gates", ""]
        lines.append(f"- R (A/B replication, mean >= {REPLICATION_MIN}): **{g.get('R')}**")
        if "MC" in g:
            lines.append(f"- MC (C/D manipulation check, mean >= {MANIP_CHECK_MIN}): **{g.get('MC')}**")
        if "SEP" in g:
            lines.append(f"- SEP (specificity, mean >= {SEPARATION_MIN}): **{g.get('SEP')}**")
            lines.append(
                f"- FLAT (entire H_null_AB CI within +/-{EQUIV_DELTA}): **{g.get('FLAT')}**"
            )
        lines += ["", "## Frozen primary statistics", ""]
        for k in PRIMARY:
            v = r["stats"].get(k)
            if v is None:
                continue
            lines.append(
                f"- `{k}`: mean {v['mean']:.6f}; median {v['median']:.6f}; "
                f"95% CI {v['ci95']}; range [{v['min']:.6f}, {v['max']:.6f}]; "
                f"positive {v['n_positive']}/{v['n']}"
            )
        lines += ["", "## Per-family distribution", "",
                  "Reported because R8-M8's mean described no observed family.", "",
                  "| seed | A | B | C | D | " + " | ".join(f"`{k}`" for k in PRIMARY) + " |",
                  "|---|---|---|---|---|" + "---|" * len(PRIMARY)]
        for row in r.get("rows", []):
            vals = " | ".join(f"{row[k]:+.4f}" for k in PRIMARY)
            lines.append(
                f"| {row['seed']} | {row['A']} | {row['B']} | {row['C']} | {row['D']} | {vals} |"
            )
        d = r.get("diagnostics", {})
        if d:
            lines += ["", "## Secondary optimization diagnostics (not gates)", ""]
            for name in ("TRUE_A", "TRUE_B", "NULL_C", "NULL_D"):
                dd = d.get(name)
                if not dd:
                    continue
                lines.append(
                    f"- {name}: update_norm_mean {dd['update_norm_mean']['mean']:.6f}; "
                    f"grad_norm_mean {dd['grad_norm_mean']['mean']:.6f}; "
                    f"fork_distance {dd['param_distance_from_fork']['mean']:.6f}"
                )
            if d.get("note"):
                lines += ["", d["note"]]
    lines += ["", "## Claim boundary", "", r.get("claim_boundary", ""), ""]
    Path(path).write_text("\n".join(lines))


def self_check():
    assert bootstrap_mean_ci([1.0, 1.0, 1.0], 1)[0] > 0
    assert bootstrap_mean_ci([-1.0, -1.0, -1.0], 2)[1] < 0

    # The equivalence test must reject a clearly negative effect that the old
    # CI-lower-bound-below-zero rule would have called flat.
    assert is_flat([-0.10, 0.10]) is True
    assert is_flat([-0.90, -0.60]) is False, "negative effect must not count as flat"
    assert is_flat([-0.50, 0.05]) is False, "wide interval must not count as flat"
    assert is_flat([0.05, 0.30]) is False, "interval escaping the upper margin is not flat"

    # SPECIFICITY must decompose exactly.
    ht, hn = 1.20, 0.15
    assert abs((ht - hn) - 1.05) < 1e-12

    d = distribution([0.05, 3.10])
    assert d["median"] == 1.575 and d["n_positive"] == 2 and d["min"] == 0.05
    print("R8-M10 classifier self-check ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root")
    ap.add_argument("--outdir")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        self_check()
        return
    if not args.root or not args.outdir:
        raise SystemExit("--root and --outdir required")
    summaries = load_summaries(args.root)
    result = classify(summaries)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "FINAL_RESULT.json", result)
    write_md(out / "FINAL_RESULT.md", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
