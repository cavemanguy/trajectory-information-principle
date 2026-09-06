import argparse
import json
from pathlib import Path

import numpy as np

EXPECTED = (1782, 1801, 1819, 1838, 1856, 1875, 1893, 1912, 1930, 1949, 1967, 1986)
BOOT_SEED = 86061
BOOT_N = 20000
MIN_REC_EFFECT = 0.05
MIN_FUNC_EFFECT = 0.005
MIN_POSITIVE = 10


def load_rows(root):
    return [json.loads(p.read_text()) for p in sorted(Path(root).rglob("seed_summary.json"))]


def bootstrap_ci(vals):
    vals = np.asarray(vals, dtype=np.float64)
    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.integers(0, len(vals), size=(BOOT_N, len(vals)))
    means = vals[draws].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def stats(vals):
    vals = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "positive_lineages": int((vals > 0).sum()),
        "bootstrap_95_ci": bootstrap_ci(vals),
        "n": int(len(vals)),
    }


def condition_values(rows, selector, key):
    vals = []
    for row in rows:
        for c in row["diagnostic"]["conditions"]:
            if selector(c):
                vals.append(float(c[key]))
    return vals


def recovery_at(c, k):
    rec = c["recovery"]
    tan_total = 0.5 * (
        rec["TPLUS"][k]["total_retention"] + rec["TMINUS"][k]["total_retention"]
    )
    trans_total = float(np.mean([
        rec[f"P{j}"][k]["total_retention"] for j in range(4)
    ]))
    tan_parallel = 0.5 * (
        rec["TPLUS"][k]["parallel_retention"] + rec["TMINUS"][k]["parallel_retention"]
    )
    tan_transverse = 0.5 * (
        rec["TPLUS"][k]["transverse_retention"] + rec["TMINUS"][k]["transverse_retention"]
    )
    trans_parallel = float(np.mean([
        rec[f"P{j}"][k]["parallel_retention"] for j in range(4)
    ]))
    trans_transverse = float(np.mean([
        rec[f"P{j}"][k]["transverse_retention"] for j in range(4)
    ]))
    phase_ret = 0.5 * (
        rec["TPLUS"][k]["signed_progress"] - rec["TMINUS"][k]["signed_progress"]
    )
    return {
        "tan_total": tan_total,
        "trans_total": trans_total,
        "d_total": tan_total - trans_total,
        "tan_parallel": tan_parallel,
        "tan_transverse": tan_transverse,
        "trans_parallel": trans_parallel,
        "trans_transverse": trans_transverse,
        "phase_ret": phase_ret,
    }


def summarize_secondary(rows):
    by_time = {}
    by_rho = {}
    curves = {}
    by_regime = {}

    for t in (2, 4, 6, 8):
        dr = condition_values(rows, lambda c, tt=t: int(c["t"]) == tt, "d_rec")
        df = condition_values(rows, lambda c, tt=t: int(c["t"]) == tt, "d_func")
        by_time[str(t)] = {
            "mean_d_rec": float(np.mean(dr)),
            "mean_d_func": float(np.mean(df)),
        }

    for rho in (0.05, 0.10, 0.20):
        dr = condition_values(rows, lambda c, rr=rho: abs(float(c["rho"]) - rr) < 1e-12, "d_rec")
        df = condition_values(rows, lambda c, rr=rho: abs(float(c["rho"]) - rr) < 1e-12, "d_func")
        by_rho[f"{rho:.2f}"] = {
            "mean_d_rec": float(np.mean(dr)),
            "mean_d_func": float(np.mean(df)),
        }

    for k in range(5):
        vals = []
        for row in rows:
            for c in row["diagnostic"]["conditions"]:
                vals.append(recovery_at(c, k))
        curves[str(k)] = {
            key: float(np.mean([v[key] for v in vals]))
            for key in (
                "tan_total", "trans_total", "d_total", "tan_parallel", "tan_transverse",
                "trans_parallel", "trans_transverse", "phase_ret"
            )
        }

    groups = {}
    for row in rows:
        regime = row["diagnostic"].get("ad2_regime_secondary") or "UNKNOWN"
        groups.setdefault(regime, {"d_rec": [], "d_func": [], "phase_ret": []})
        p = row["diagnostic"]["primary"]
        groups[regime]["d_rec"].append(p["d_rec"])
        groups[regime]["d_func"].append(p["d_func"])
        groups[regime]["phase_ret"].append(p["phase_ret"])
    for regime, g in groups.items():
        by_regime[regime] = {
            "n": len(g["d_rec"]),
            "mean_d_rec": float(np.mean(g["d_rec"])),
            "mean_d_func": float(np.mean(g["d_func"])),
            "mean_phase_ret": float(np.mean(g["phase_ret"])),
        }

    cell_tan_ret = []
    cell_drop_tan = []
    for row in rows:
        for c in row["diagnostic"]["conditions"]:
            cell_tan_ret.append(float(c["ret_tan_k3"]))
            cell_drop_tan.append(float(c["drop_tan"]))
    corr = float(np.corrcoef(cell_tan_ret, cell_drop_tan)[0, 1]) if np.std(cell_tan_ret) > 0 and np.std(cell_drop_tan) > 0 else 0.0

    return {
        "by_time": by_time,
        "by_rho": by_rho,
        "recovery_curves": curves,
        "by_ad2_regime": by_regime,
        "exploratory_condition_cell_corr_tan_retention_vs_damage": corr,
    }


def classify(rows):
    seeds = [int(r.get("seed", -1)) for r in rows]
    exact = sorted(seeds) == list(EXPECTED) and len(seeds) == len(set(seeds)) == len(EXPECTED)
    valid = bool(exact and all(r.get("all_valid") is True for r in rows))

    if not valid:
        return {
            "experiment": "R8-AD6",
            "classification": "V0 — invalid",
            "n_valid": int(sum(bool(r.get("all_valid")) for r in rows)),
            "expected_seeds": list(EXPECTED),
            "observed_seeds": seeds,
            "validity": {"exact_seed_set": exact, "all_valid": False},
            "gates": {"G_RECOVERY": False, "G_FUNCTION": False},
        }

    rows = sorted(rows, key=lambda r: int(r["seed"]))
    d_rec = [r["diagnostic"]["primary"]["d_rec"] for r in rows]
    d_func = [r["diagnostic"]["primary"]["d_func"] for r in rows]
    d_func_ce = [r["diagnostic"]["primary"]["d_func_ce"] for r in rows]
    phase_ret = [r["diagnostic"]["primary"]["phase_ret"] for r in rows]
    ret_tan = [r["diagnostic"]["primary"]["ret_tan"] for r in rows]
    ret_trans = [r["diagnostic"]["primary"]["ret_trans"] for r in rows]
    drop_tan = [r["diagnostic"]["primary"]["drop_tan"] for r in rows]
    drop_trans = [r["diagnostic"]["primary"]["drop_trans"] for r in rows]

    s_rec = stats(d_rec)
    s_func = stats(d_func)
    g_rec = bool(
        s_rec["mean"] >= MIN_REC_EFFECT
        and s_rec["bootstrap_95_ci"][0] > 0
        and s_rec["positive_lineages"] >= MIN_POSITIVE
    )
    g_func = bool(
        s_func["mean"] >= MIN_FUNC_EFFECT
        and s_func["bootstrap_95_ci"][0] > 0
        and s_func["positive_lineages"] >= MIN_POSITIVE
    )

    if g_rec and g_func:
        label = "P1 — tangent persistence with functional asymmetry supported"
    elif g_rec:
        label = "P2 — tangent persistence only"
    elif g_func:
        label = "P3 — functional tangent sensitivity only"
    else:
        label = "P0 — tangent/phase account not supported by primary gates"

    out_rows = []
    for r in rows:
        d = r["diagnostic"]
        p = d["primary"]
        out_rows.append({
            "seed": int(r["seed"]),
            "M": int(r["maturity_epoch"]),
            "ad2_regime": d.get("ad2_regime_secondary"),
            "native_accuracy": d["native"]["accuracy"],
            "ret_tan": p["ret_tan"],
            "ret_trans": p["ret_trans"],
            "d_rec": p["d_rec"],
            "phase_ret": p["phase_ret"],
            "drop_tan": p["drop_tan"],
            "drop_trans": p["drop_trans"],
            "d_func": p["d_func"],
            "d_func_ce": p["d_func_ce"],
            "max_geometry_error": d["max_geometry_error"],
        })

    return {
        "experiment": "R8-AD6",
        "classification": label,
        "n_valid": len(rows),
        "expected_seeds": list(EXPECTED),
        "validity": {"exact_seed_set": True, "all_valid": True},
        "frozen_thresholds": {
            "min_recovery_effect": MIN_REC_EFFECT,
            "min_function_effect_accuracy": MIN_FUNC_EFFECT,
            "min_positive_lineages": MIN_POSITIVE,
            "bootstrap_seed": BOOT_SEED,
            "bootstrap_resamples": BOOT_N,
        },
        "gates": {"G_RECOVERY": g_rec, "G_FUNCTION": g_func},
        "summaries": {
            "D_REC": s_rec,
            "D_FUNC": s_func,
            "D_FUNC_CE": stats(d_func_ce),
            "PHASE_RET": stats(phase_ret),
            "RET_TAN": stats(ret_tan),
            "RET_TRANS": stats(ret_trans),
            "DROP_TAN": stats(drop_tan),
            "DROP_TRANS": stats(drop_trans),
            "MAX_GEOMETRY_ERROR": {
                "max": float(max(r["max_geometry_error"] for r in out_rows)),
                "mean": float(np.mean([r["max_geometry_error"] for r in out_rows])),
            },
        },
        "secondary": summarize_secondary(rows),
        "rows": out_rows,
        "claim_boundary": (
            "R8-AD6 tests the preregistered tangent/phase explanation generated after AD5. "
            "It measures norm-matched tangent versus transverse recovery from the same native state and "
            "their downstream task consequences. Even a positive result does not establish information beyond "
            "the complete Markov state-plus-map, an independent hidden phase variable, essential chronology, "
            "a universal trajectory code, or generalization beyond the tested synthetic recurrent system."
        ),
    }


def render_md(d):
    if d["classification"].startswith("V0"):
        return "# R8-AD6 Final Result — Tangent vs Transverse Error Dynamics\n\n**Primary classification:** V0 — invalid\n\nRequired validity checks failed. No scientific gate is promoted.\n"

    s = d["summaries"]
    lines = [
        "# R8-AD6 Final Result — Tangent vs Transverse Error Dynamics",
        "",
        f"**Primary classification:** {d['classification']}",
        "",
        "## Frozen gates",
        "",
        f"- G_RECOVERY: **{d['gates']['G_RECOVERY']}**",
        f"- G_FUNCTION: **{d['gates']['G_FUNCTION']}**",
        "",
        "## Primary summaries",
        "",
        f"- D_REC: mean {s['D_REC']['mean']:+.6f}; 95% CI [{s['D_REC']['bootstrap_95_ci'][0]:+.6f}, {s['D_REC']['bootstrap_95_ci'][1]:+.6f}]; positive {s['D_REC']['positive_lineages']}/12",
        f"- tangent k=3 retention: mean {s['RET_TAN']['mean']:.6f}",
        f"- transverse k=3 retention: mean {s['RET_TRANS']['mean']:.6f}",
        f"- PHASE_RET k=3: mean {s['PHASE_RET']['mean']:+.6f}; 95% CI [{s['PHASE_RET']['bootstrap_95_ci'][0]:+.6f}, {s['PHASE_RET']['bootstrap_95_ci'][1]:+.6f}]",
        f"- D_FUNC: mean {s['D_FUNC']['mean']:+.6f}; 95% CI [{s['D_FUNC']['bootstrap_95_ci'][0]:+.6f}, {s['D_FUNC']['bootstrap_95_ci'][1]:+.6f}]; positive {s['D_FUNC']['positive_lineages']}/12",
        f"- tangent terminal accuracy drop: mean {s['DROP_TAN']['mean']:+.6f}",
        f"- transverse terminal accuracy drop: mean {s['DROP_TRANS']['mean']:+.6f}",
        f"- D_FUNC_CE: mean {s['D_FUNC_CE']['mean']:+.6f}",
        f"- maximum frozen geometry mismatch: {s['MAX_GEOMETRY_ERROR']['max']:.3e}",
        "",
        "## Per-lineage values",
        "",
        "| seed | M | AD2 regime | ret tan | ret trans | D_REC | phase ret | drop tan | drop trans | D_FUNC |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in d["rows"]:
        lines.append(
            f"| {r['seed']} | {r['M']} | {r['ad2_regime']} | {r['ret_tan']:.4f} | {r['ret_trans']:.4f} | "
            f"{r['d_rec']:+.4f} | {r['phase_ret']:+.4f} | {r['drop_tan']:+.4f} | {r['drop_trans']:+.4f} | {r['d_func']:+.4f} |"
        )
    lines += ["", "## Claim boundary", "", d["claim_boundary"], ""]
    return "\n".join(lines)


def self_check():
    def fake(seed, dr, df):
        rec_arm = [{"total_retention": 1.0, "parallel_retention": 1.0, "transverse_retention": 0.0, "parallel_fraction": 1.0, "signed_progress": 1.0} for _ in range(5)]
        rec_neg = [{**x, "signed_progress": -1.0} for x in rec_arm]
        rec_p = [{"total_retention": 0.5, "parallel_retention": 0.1, "transverse_retention": 0.49, "parallel_fraction": 0.2, "signed_progress": 0.0} for _ in range(5)]
        c = {
            "t": 2, "rho": 0.1, "d_rec": dr, "d_func": df, "ret_tan_k3": 1.0,
            "ret_trans_k3": 1.0-dr, "phase_ret_k3": 1.0, "drop_tan": 0.03,
            "drop_trans": 0.03-df,
            "recovery": {"TPLUS": rec_arm, "TMINUS": rec_neg, "P0": rec_p, "P1": rec_p, "P2": rec_p, "P3": rec_p},
        }
        return {
            "seed": seed, "maturity_epoch": 100, "all_valid": True,
            "diagnostic": {
                "native": {"accuracy": 0.5}, "ad2_regime_secondary": "TEST", "max_geometry_error": 1e-7,
                "primary": {"ret_tan": 1.0, "ret_trans": 1.0-dr, "d_rec": dr, "phase_ret": 1.0,
                            "drop_tan": 0.03, "drop_trans": 0.03-df, "d_func": df, "d_func_ce": df},
                "conditions": [c],
            },
        }
    assert classify([fake(s, 0.10, 0.02) for s in EXPECTED])["classification"].startswith("P1")
    assert classify([fake(s, 0.10, -0.01) for s in EXPECTED])["classification"].startswith("P2")
    assert classify([fake(s, -0.01, 0.02) for s in EXPECTED])["classification"].startswith("P3")
    assert classify([fake(s, -0.01, -0.01) for s in EXPECTED])["classification"].startswith("P0")
    print("R8-AD6 classifier self-check ok")


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
    result = classify(load_rows(args.root))
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "FINAL_RESULT.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    (out / "FINAL_RESULT.md").write_text(render_md(result))
    print(result["classification"])


if __name__ == "__main__":
    main()
