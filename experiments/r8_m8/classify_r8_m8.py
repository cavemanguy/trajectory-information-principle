import argparse
import json
from pathlib import Path

import numpy as np

SEEDS = (421, 438, 454, 471, 489, 506, 523, 541, 558, 574, 592, 611)
LEVELS = (0.00, 0.25, 0.50, 0.75, 1.00)
N_BOOT = 5000


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def bootstrap_mean_ci(x, seed):
    x = np.asarray(x, dtype=np.float64)
    g = np.random.default_rng(seed)
    draws = np.empty(N_BOOT, dtype=np.float64)
    for i in range(N_BOOT):
        ix = g.integers(0, len(x), len(x))
        draws[i] = x[ix].mean()
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def load_summaries(root):
    by_seed = {}
    for p in Path(root).rglob("seed_summary.json"):
        d = json.loads(p.read_text())
        if d.get("experiment") == "R8-M8":
            by_seed[int(d["seed"])] = d
    missing = [s for s in SEEDS if s not in by_seed]
    if missing:
        raise RuntimeError(f"missing R8-M8 seed summaries: {missing}")
    return [by_seed[s] for s in SEEDS]


def rec_at_lambda(d, branch, lam):
    for r in d["sweeps"][branch]["records"]:
        if abs(float(r["lambda"]) - float(lam)) < 1e-10:
            return r
    raise KeyError((d["seed"], branch, lam))


def hold_rec(d, branch, hold_epoch):
    for r in d["holds"][branch]["records"]:
        if int(r["hold_epoch"]) == int(hold_epoch):
            return r
    raise KeyError((d["seed"], branch, hold_epoch))


def functional_d(rec):
    vals = np.asarray(rec["validation"]["h12_per_relation"], dtype=np.float64)
    winner = int(rec["survival"]["winner_relation"])
    return float(vals[winner] - np.delete(vals, winner).mean())


def classify(summaries):
    maturity_bad = []
    execution_bad = []
    for d in summaries:
        v = d.get("validity", {})
        if not bool(v.get("maturity")):
            maturity_bad.append({"seed": int(d["seed"]), "validity": v})
        other = [v.get("baseline_fork_identity"), v.get("midpoint_hold_fork_identity"), v.get("finite"), v.get("complete")]
        if bool(v.get("maturity")) and not all(bool(x) for x in other):
            execution_bad.append({"seed": int(d["seed"]), "validity": v})

    if maturity_bad:
        return {
            "classification": "V0 — maturity validity failure",
            "all_valid": False,
            "maturity_failures": maturity_bad,
        }
    if execution_bad:
        return {
            "classification": "V1 — post-maturity execution validity failure",
            "all_valid": False,
            "execution_failures": execution_bad,
        }

    rows = []
    for d in summaries:
        seed = int(d["seed"])
        A = int(d["A_baseline_winner"])
        B = int(d["B_baseline_loser"])
        h_by_level = {}
        q_a = {}
        q_b = {}
        sweep_detail = {}
        for lam in LEVELS:
            ra = rec_at_lambda(d, "A_SWEEP", lam)
            rb = rec_at_lambda(d, "B_SWEEP", lam)
            qa = float(ra["Q"])
            qb = float(rb["Q"])
            q_a[str(lam)] = qa
            q_b[str(lam)] = qb
            h_by_level[str(lam)] = qb - qa
            sweep_detail[str(lam)] = {
                "A_SWEEP_Q": qa,
                "B_SWEEP_Q": qb,
                "H": qb - qa,
                "A_SWEEP_winner": int(ra["survival"]["winner_relation"]),
                "B_SWEEP_winner": int(rb["survival"]["winner_relation"]),
                "A_SWEEP_G": float(ra["survival"]["G"]),
                "B_SWEEP_G": float(rb["survival"]["G"]),
                "A_SWEEP_D": functional_d(ra),
                "B_SWEEP_D": functional_d(rb),
                "A_SWEEP_h12_A": float(ra["validation"]["h12_per_relation"][A]),
                "A_SWEEP_h12_B": float(ra["validation"]["h12_per_relation"][B]),
                "B_SWEEP_h12_A": float(rb["validation"]["h12_per_relation"][A]),
                "B_SWEEP_h12_B": float(rb["validation"]["h12_per_relation"][B]),
            }
        hs = np.asarray([h_by_level[str(l)] for l in LEVELS], dtype=np.float64)
        area = float(np.trapezoid(hs, np.asarray(LEVELS, dtype=np.float64)))
        h_mid = float(h_by_level[str(0.5)])
        ha120 = hold_rec(d, "A_HOLD", 120)
        hb120 = hold_rec(d, "B_HOLD", 120)
        h_hold120 = float(hb120["Q"] - ha120["Q"])

        hold_curve = {}
        for ep in (30, 60, 90, 120):
            ha = hold_rec(d, "A_HOLD", ep)
            hb = hold_rec(d, "B_HOLD", ep)
            hold_curve[str(ep)] = float(hb["Q"] - ha["Q"])

        ra_mid = rec_at_lambda(d, "A_SWEEP", 0.5)
        rb_mid = rec_at_lambda(d, "B_SWEEP", 0.5)
        exact_opposite_mid = bool(
            int(ra_mid["survival"]["winner_relation"]) == A
            and int(rb_mid["survival"]["winner_relation"]) == B
        )
        functional_mid_contrast = float(
            (rb_mid["validation"]["h12_per_relation"][B] - rb_mid["validation"]["h12_per_relation"][A])
            - (ra_mid["validation"]["h12_per_relation"][B] - ra_mid["validation"]["h12_per_relation"][A])
        )

        rows.append({
            "seed": seed,
            "M": int(d["maturity_epoch"]),
            "A": A,
            "B": B,
            "baseline_Q": float(d["baseline"]["Q"]),
            "H_mid": h_mid,
            "AREA": area,
            "H_hold120": h_hold120,
            "H_by_lambda": h_by_level,
            "hold_curve": hold_curve,
            "Q_A_SWEEP": q_a,
            "Q_B_SWEEP": q_b,
            "sweep_detail": sweep_detail,
            "exact_opposite_midpoint_winners": exact_opposite_mid,
            "functional_midpoint_contrast": functional_mid_contrast,
            "midpoint_mean_h0_distance": float(d["midpoint_latent_distance"]["mean_h0_distance"]),
            "midpoint_mean_h12_distance": float(d["midpoint_latent_distance"]["mean_h12_distance"]),
        })

    def arr(k):
        return np.asarray([r[k] for r in rows], dtype=np.float64)

    stats = {}
    for k, sd in (("H_mid", 88101), ("AREA", 88102), ("H_hold120", 88103), ("functional_midpoint_contrast", 88104)):
        x = arr(k)
        stats[k] = {"mean": float(x.mean()), "ci95": bootstrap_mean_ci(x, sd)}

    pointwise = {}
    for i, lam in enumerate(LEVELS):
        x = np.asarray([r["H_by_lambda"][str(lam)] for r in rows], dtype=np.float64)
        pointwise[str(lam)] = {"mean": float(x.mean()), "ci95": bootstrap_mean_ci(x, 88200 + i)}

    hold_curve_stats = {}
    for i, ep in enumerate((30, 60, 90, 120)):
        x = np.asarray([r["hold_curve"][str(ep)] for r in rows], dtype=np.float64)
        hold_curve_stats[str(ep)] = {"mean": float(x.mean()), "ci95": bootstrap_mean_ci(x, 88300 + i)}

    H = bool(stats["H_mid"]["mean"] >= 0.50 and stats["H_mid"]["ci95"][0] > 0)
    L = bool(stats["AREA"]["mean"] >= 0.25 and stats["AREA"]["ci95"][0] > 0)
    P = bool(stats["H_hold120"]["mean"] >= 0.25 and stats["H_hold120"]["ci95"][0] > 0)

    if not H:
        classification = "Y0 — matched history dependence not supported"
    elif not L:
        classification = "Y1 — matched midpoint history dependence supported"
    elif not P:
        classification = "Y2 — hysteresis-like sweep separation supported"
    else:
        classification = "Y3 — persistent history-dependent regime separation supported"

    h_mid_mean = stats["H_mid"]["mean"]
    retention = None if abs(h_mid_mean) < 1e-12 else float(stats["H_hold120"]["mean"] / h_mid_mean)

    secondary = {
        "pointwise_H": pointwise,
        "hold_H_curve": hold_curve_stats,
        "retention_fraction_of_mean_midpoint_separation": retention,
        "exact_opposite_midpoint_winner_count": int(sum(r["exact_opposite_midpoint_winners"] for r in rows)),
        "maturity_epoch": {
            "mean": float(np.mean([r["M"] for r in rows])),
            "min": int(min(r["M"] for r in rows)),
            "max": int(max(r["M"] for r in rows)),
            "per_seed": {str(r["seed"]): int(r["M"]) for r in rows},
        },
        "midpoint_latent_distance": {
            "mean_h0": float(np.mean([r["midpoint_mean_h0_distance"] for r in rows])),
            "mean_h12": float(np.mean([r["midpoint_mean_h12_distance"] for r in rows])),
        },
    }

    return {
        "classification": classification,
        "all_valid": True,
        "H_supported": H,
        "L_supported": L,
        "P_supported": P,
        "stats": stats,
        "secondary": secondary,
        "rows": rows,
    }


def write_md(path, r):
    lines = ["# R8-M8 Final Result", "", f"**Primary classification:** {r['classification']}", ""]
    if not r.get("all_valid", False):
        lines.append(f"- Validity record: {r}")
    else:
        lines += [
            "- Cross-family maturity/execution validity: **True**",
            f"- H matched-midpoint history effect: **{r['H_supported']}**",
            f"- L signed sweep-loop separation: **{r['L_supported']}**",
            f"- P +120 identical-demand persistence: **{r['P_supported']}**",
            "",
            "## Frozen primary statistics",
            "",
        ]
        for k in ("H_mid", "AREA", "H_hold120"):
            v = r["stats"][k]
            lines.append(f"- {k}: mean {v['mean']:.6f}; 95% CI {v['ci95']}")
        lines += ["", "## Secondary descriptors", ""]
        lines.append(f"- pointwise H(lambda): {r['secondary']['pointwise_H']}")
        lines.append(f"- hold H curve: {r['secondary']['hold_H_curve']}")
        lines.append(f"- retention fraction: {r['secondary']['retention_fraction_of_mean_midpoint_separation']}")
        lines.append(f"- exact opposite midpoint winners: {r['secondary']['exact_opposite_midpoint_winner_count']}/12")
        lines.append(f"- maturity epochs: {r['secondary']['maturity_epoch']}")
        lines.append(f"- midpoint latent distance: {r['secondary']['midpoint_latent_distance']}")
        v = r["stats"]["functional_midpoint_contrast"]
        lines.append(f"- functional midpoint contrast: mean {v['mean']:.6f}; 95% CI {v['ci95']}")
    lines += [
        "",
        "## Claim boundary",
        "",
        "R8-M8 tests operational history dependence in native dynamical organization under matched current demand. Even Y3 does not establish mathematical bistability, formal thermodynamic hysteresis, conscious choice, universal trajectory computation, essential chronology, information beyond the complete state, or generalization beyond this synthetic system.",
        "",
    ]
    Path(path).write_text("\n".join(lines))


def self_check():
    assert bootstrap_mean_ci([1.0, 1.0, 1.0], 1)[0] > 0
    assert bootstrap_mean_ci([-1.0, -1.0, -1.0], 2)[1] < 0
    x = np.asarray([1.0, 1.0, 1.0, 1.0, 1.0])
    assert abs(float(np.trapezoid(x, np.asarray(LEVELS))) - 1.0) < 1e-12
    assert 0.50 >= 0.50 and 0.25 >= 0.25
    print("R8-M8 classifier self-check ok")


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
