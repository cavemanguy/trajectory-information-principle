import argparse
import json
import math
from pathlib import Path

import numpy as np

SEEDS = (1341, 1359, 1376, 1394, 1412, 1429, 1447, 1465, 1483, 1500, 1518, 1536)
N_BOOT = 5000
N_PERM = 20000
REPLICATION_MIN = 0.50
MANIP_MIN = 0.50
SEPARATION_MIN = 0.50
EQUIV_DELTA = 0.25
RHO_MIN = 0.60
LOCALIZATION_MARGIN = 0.20
PRIMARY_EFFECTS = ("H_true_AB", "H_null_AB", "H_null_CD", "H_true_CD", "SPECIFICITY")


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def ranks(a):
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    out = np.empty(len(a), dtype=np.float64)
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and a[order[j]] == a[order[i]]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        out[order[i:j]] = rank
        i = j
    return out


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def bootstrap_mean_ci(x, seed):
    x = np.asarray(x, dtype=np.float64)
    g = np.random.default_rng(seed)
    vals = []
    for _ in range(N_BOOT):
        ix = g.integers(0, len(x), len(x))
        vals.append(float(x[ix].mean()))
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def distribution(x, seed):
    a = np.asarray(x, dtype=np.float64)
    return {
        "values": [float(v) for v in a],
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "min": float(a.min()),
        "max": float(a.max()),
        "n_positive": int(np.sum(a > 0)),
        "n": int(len(a)),
        "ci95": bootstrap_mean_ci(a, seed),
    }


def bootstrap_corr_ci(x, y, seed):
    x, y = np.asarray(x), np.asarray(y)
    g = np.random.default_rng(seed)
    vals = []
    for _ in range(N_BOOT):
        ix = g.integers(0, len(x), len(x))
        r = spearman(x[ix], y[ix])
        if math.isfinite(r):
            vals.append(r)
    if not vals:
        return [float("nan"), float("nan")]
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def bootstrap_corr_diff_ci(x1, x2, y, seed):
    x1, x2, y = np.asarray(x1), np.asarray(x2), np.asarray(y)
    g = np.random.default_rng(seed)
    vals = []
    for _ in range(N_BOOT):
        ix = g.integers(0, len(y), len(y))
        r1, r2 = spearman(x1[ix], y[ix]), spearman(x2[ix], y[ix])
        if math.isfinite(r1) and math.isfinite(r2):
            vals.append(r1 - r2)
    if not vals:
        return [float("nan"), float("nan")]
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def permutation_p(x, y, seed):
    x, y = np.asarray(x), np.asarray(y)
    obs = abs(spearman(x, y))
    g = np.random.default_rng(seed)
    hit = 0
    for _ in range(N_PERM):
        yp = y[g.permutation(len(y))]
        if abs(spearman(x, yp)) >= obs - 1e-15:
            hit += 1
    return float((hit + 1) / (N_PERM + 1))


def is_flat(ci):
    return bool(ci[0] > -EQUIV_DELTA and ci[1] < EQUIV_DELTA)


def load_summaries(root):
    by_seed = {}
    for p in Path(root).rglob("seed_summary.json"):
        d = json.loads(p.read_text())
        if d.get("experiment") == "R8-M12":
            by_seed[int(d["seed"])] = d
    missing = [s for s in SEEDS if s not in by_seed]
    if missing:
        raise RuntimeError(f"missing R8-M12 summaries: {missing}")
    return [by_seed[s] for s in SEEDS]


def classify(ds):
    bad = []
    for d in ds:
        v = d.get("validity", {})
        needed = ("maturity", "pairs_disjoint", "diagnostic_immutable", "fork_identity",
                  "lineage_length", "finite", "complete")
        if not all(bool(v.get(k)) for k in needed):
            bad.append({"seed": d.get("seed"), "validity": v})
    if bad:
        return {"classification": "V0 — R8-M12 validity failure", "all_valid": False, "failures": bad}

    rows = []
    for d in ds:
        b = d["baseline"]
        s = d["susceptibility"]
        e = d["effects"]
        row = {
            "seed": int(d["seed"]),
            "M": int(d["maturity_epoch"]),
            "A": int(d["A_baseline_winner"]),
            "B": int(d["B_baseline_loser"]),
            "C": int(d["C_off_axis"]),
            "D": int(d["D_off_axis"]),
            "S_F1_AB": float(s["AB"]["F1"]["relative_contrast"]),
            "S_F2_AB": float(s["AB"]["F2"]["relative_contrast"]),
            "S_E_AB": float(s["AB"]["E"]["relative_contrast"]),
            "S_R_AB": float(s["AB"]["R"]["relative_contrast"]),
            "S_F1_CD": float(s["CD"]["F1"]["relative_contrast"]),
            "F1_cos_AB": float(s["AB"]["F1"]["cosine"]),
            "F1_raw_AB": float(s["AB"]["F1"]["contrast_norm"]),
            "baseline_combined": float(b["validation"]["combined"]),
            "baseline_h0": float(b["validation"]["h0_overall"]),
            "baseline_Q_AB": float(b["Q_AB"]),
            "baseline_G": float(b["survival"]["G"]),
            "baseline_C": float(b["survival"]["C"]),
        }
        for k in PRIMARY_EFFECTS:
            row[k] = float(e[k])
        rows.append(row)

    stats = {k: distribution([r[k] for r in rows], 91200 + i)
             for i, k in enumerate(PRIMARY_EFFECTS)}

    R = bool(stats["H_true_AB"]["mean"] >= REPLICATION_MIN and stats["H_true_AB"]["ci95"][0] > 0)
    MC = bool(stats["H_null_CD"]["mean"] >= MANIP_MIN and stats["H_null_CD"]["ci95"][0] > 0)
    FLAT = is_flat(stats["H_null_AB"]["ci95"])
    SEP = bool(stats["SPECIFICITY"]["mean"] >= SEPARATION_MIN and stats["SPECIFICITY"]["ci95"][0] > 0)
    parent = bool(R and MC and FLAT and SEP)

    y = np.asarray([r["SPECIFICITY"] for r in rows])
    predictors = ("S_F1_AB", "S_F2_AB", "S_E_AB", "S_R_AB", "S_F1_CD",
                  "M", "baseline_combined", "baseline_h0", "baseline_Q_AB", "baseline_G", "baseline_C")
    corrs = {}
    for i, k in enumerate(predictors):
        x = np.asarray([r[k] for r in rows], dtype=np.float64)
        corrs[k] = {
            "rho": spearman(x, y),
            "ci95": bootstrap_corr_ci(x, y, 91300 + i),
            "permutation_p_two_sided": permutation_p(x, y, 91400 + i),
        }

    f1 = corrs["S_F1_AB"]
    P = bool(f1["rho"] >= RHO_MIN and f1["ci95"][0] > 0 and f1["permutation_p_two_sided"] < 0.05)

    x1 = np.asarray([r["S_F1_AB"] for r in rows])
    x2 = np.asarray([r["S_F2_AB"] for r in rows])
    xe = np.asarray([r["S_E_AB"] for r in rows])
    d_f2 = float(corrs["S_F1_AB"]["rho"] - corrs["S_F2_AB"]["rho"])
    d_e = float(corrs["S_F1_AB"]["rho"] - corrs["S_E_AB"]["rho"])
    d_f2_ci = bootstrap_corr_diff_ci(x1, x2, y, 91501)
    d_e_ci = bootstrap_corr_diff_ci(x1, xe, y, 91502)
    LOC = bool(P and d_f2 >= LOCALIZATION_MARGIN and d_f2_ci[0] > 0
               and d_e >= LOCALIZATION_MARGIN and d_e_ci[0] > 0)

    if not parent:
        classification = "U0 — parent axis-specific phenomenon not fully replicated; predictor not promoted"
        boundary = "At least one frozen M10 parent gate failed on the fresh R8-M12 families, so no heterogeneity predictor is promoted."
    elif not P:
        classification = "U1 — no preregistered F1 susceptibility predictor"
        boundary = "The parent axis-specific phenomenon replicated, but pre-history F1 relative gradient contrast did not meet the frozen prediction gate."
    elif LOC:
        classification = "U3 — F1-localized pre-history susceptibility predictor supported"
        boundary = ("Pre-history F1 relative gradient contrast predicts later axis-specific persistent reorganization and exceeds the "
                    "matched F2 and encoder predictor correlations by the frozen margins. This is predictive localization, not causal proof.")
    else:
        classification = "U2 — pre-history F1 susceptibility predicts later specificity"
        boundary = ("Pre-history F1 relative gradient contrast predicts later axis-specific persistent reorganization, but the frozen "
                    "localization comparison does not establish that this predictive signal is specific to F1.")

    return {
        "classification": classification,
        "all_valid": True,
        "parent_gates": {"R": R, "MC": MC, "FLAT": FLAT, "SEP": SEP, "all": parent},
        "predictor_gate": {"supported": P, "rho_min": RHO_MIN},
        "localization_gate": {
            "supported": LOC,
            "margin": LOCALIZATION_MARGIN,
            "rho_F1_minus_F2": d_f2,
            "rho_F1_minus_F2_ci95": d_f2_ci,
            "rho_F1_minus_E": d_e,
            "rho_F1_minus_E_ci95": d_e_ci,
        },
        "stats": stats,
        "correlations": corrs,
        "rows": rows,
        "claim_boundary": boundary,
    }


def write_md(path, r):
    lines = ["# R8-M12 Final Result — Pre-History Susceptibility Predictor", "",
             f"**Primary classification:** {r['classification']}", ""]
    if not r.get("all_valid"):
        lines += ["## Validity", "", json.dumps(r, indent=2)]
    else:
        lines += ["## Parent M10 gates", ""]
        for k, v in r["parent_gates"].items():
            lines.append(f"- {k}: **{v}**")
        lines += ["", "## Primary predictor", ""]
        f1 = r["correlations"]["S_F1_AB"]
        lines.append(f"- Spearman rho(S_F1_AB, SPECIFICITY): {f1['rho']:+.6f}")
        lines.append(f"- bootstrap 95% CI: {f1['ci95']}")
        lines.append(f"- two-sided permutation p: {f1['permutation_p_two_sided']:.6g}")
        lines.append(f"- predictor gate: **{r['predictor_gate']['supported']}**")
        lines += ["", "## Localization descriptor", ""]
        lg = r["localization_gate"]
        lines.append(f"- rho_F1 - rho_F2: {lg['rho_F1_minus_F2']:+.6f}; CI {lg['rho_F1_minus_F2_ci95']}")
        lines.append(f"- rho_F1 - rho_E: {lg['rho_F1_minus_E']:+.6f}; CI {lg['rho_F1_minus_E_ci95']}")
        lines.append(f"- localization gate: **{lg['supported']}**")
        lines += ["", "## Parent primary statistics", ""]
        for k, v in r["stats"].items():
            lines.append(f"- `{k}` mean {v['mean']:+.6f}; median {v['median']:+.6f}; 95% CI {v['ci95']}; "
                         f"range [{v['min']:+.6f}, {v['max']:+.6f}]; positive {v['n_positive']}/{v['n']}")
        lines += ["", "## All preregistered correlations", ""]
        for k, v in r["correlations"].items():
            lines.append(f"- `{k}`: rho {v['rho']:+.6f}; CI {v['ci95']}; p={v['permutation_p_two_sided']:.6g}")
        lines += ["", "## Per-family values", "",
                  "| seed | M | S_F1_AB | S_F2_AB | S_E_AB | H_true_AB | H_null_AB | SPECIFICITY |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for x in r["rows"]:
            lines.append(f"| {x['seed']} | {x['M']} | {x['S_F1_AB']:.4f} | {x['S_F2_AB']:.4f} | "
                         f"{x['S_E_AB']:.4f} | {x['H_true_AB']:+.4f} | {x['H_null_AB']:+.4f} | {x['SPECIFICITY']:+.4f} |")
    lines += ["", "## Claim boundary", "", r.get("claim_boundary", ""), ""]
    Path(path).write_text("\n".join(lines))


def self_check():
    x = np.arange(12, dtype=np.float64)
    y = x * 2.0 + 1.0
    assert abs(spearman(x, y) - 1.0) < 1e-12
    assert abs(spearman(x, -y) + 1.0) < 1e-12
    assert bootstrap_corr_ci(x, y, 1)[0] > 0.9
    assert permutation_p(x, y, 2) < 0.01
    assert is_flat([-0.10, 0.10])
    assert not is_flat([-0.50, 0.10])
    print("R8-M12 classifier self-check ok")


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
    result = classify(load_summaries(args.root))
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "FINAL_RESULT.json", result)
    write_md(out / "FINAL_RESULT.md", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
