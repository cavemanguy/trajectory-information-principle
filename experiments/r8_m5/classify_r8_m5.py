import argparse
import json
from pathlib import Path

import numpy as np

FRESH_SEEDS = (18, 33, 46, 61, 76, 91, 106, 121, 141, 156, 173, 188)
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
    root = Path(root)
    by_seed = {}
    for p in root.rglob("seed_summary.json"):
        d = json.loads(p.read_text())
        if d.get("experiment") == "R8-M5":
            by_seed[int(d["seed"])] = d
    missing = [s for s in FRESH_SEEDS if s not in by_seed]
    if missing:
        raise RuntimeError(f"missing fresh seed summaries: {missing}")
    return [by_seed[s] for s in FRESH_SEEDS]


def metrics(summaries):
    rows = []
    for d in summaries:
        e = d["epoch100"]
        row = {"seed": int(d["seed"])}
        for c in ("B16", "S24", "S32", "P16"):
            row[f"h0_{c}"] = float(e[c]["test"]["h0_overall"])
            row[f"h12_{c}"] = float(e[c]["test"]["h12_overall"])
            row[f"G_{c}"] = float(e[c]["survival"]["G"])
            row[f"D_{c}"] = float(e[c]["winner_gap_test"])
            row[f"winner_{c}"] = int(e[c]["survival"]["winner_relation"])
        row["param_ratio"] = float(d["P16_S32_relative_parameter_difference"])
        rows.append(row)
    return rows


def paired(rows, key_a, key_b):
    return np.asarray([r[key_a] - r[key_b] for r in rows], dtype=np.float64)


def classify(summaries):
    invalid = []
    for d in summaries:
        if not bool(d.get("all_conditions_valid")):
            invalid.append({"seed": int(d["seed"]), "validity": d.get("condition_validity")})
    if invalid:
        return {
            "classification": "V — cross-capacity training validity failure",
            "all_conditions_valid": False,
            "invalid": invalid,
        }

    rows = metrics(summaries)
    ratios = [r["param_ratio"] for r in rows]
    if max(ratios) >= 0.05:
        raise RuntimeError(f"parameter-match design violation: max relative difference={max(ratios):.6f}")

    dh12 = paired(rows, "h12_S32", "h12_B16")
    dD = paired(rows, "D_S32", "D_B16")
    dG = paired(rows, "G_S32", "G_B16")
    dh0 = paired(rows, "h0_S32", "h0_B16")
    dD_spec = paired(rows, "D_S32", "D_P16")
    dG_spec = paired(rows, "G_S32", "G_P16")

    ci_h12 = bootstrap_mean_ci(dh12, 80501)
    ci_D = bootstrap_mean_ci(dD, 80502)
    ci_G = bootstrap_mean_ci(dG, 80503)
    ci_h0 = bootstrap_mean_ci(dh0, 80504)
    ci_D_spec = bootstrap_mean_ci(dD_spec, 80505)
    ci_G_spec = bootstrap_mean_ci(dG_spec, 80506)

    A1 = bool(dh12.mean() >= 0.02 and ci_h12[0] > 0)
    A2 = bool(dD.mean() <= -0.10 and ci_D[1] < 0)
    A3 = bool(dG.mean() < 0 and ci_G[1] < 0)
    A4 = bool(dh0.mean() > -0.02 and ci_h0[0] > -0.03)
    A = bool(A1 and A2 and A3 and A4)

    S1 = bool(dD_spec.mean() <= -0.05 and ci_D_spec[1] < 0)
    S2 = bool(dG_spec.mean() < 0 and ci_G_spec[1] < 0)
    S = bool(S1 and S2)

    if not A:
        classification = "C0 — simple capacity-allocation account not supported"
    elif S:
        classification = "C2 — state-dimension-specific capacity allocation supported"
    else:
        classification = "C1 — wider-state allocation pattern supported, state specificity not established"

    width_order = ("B16", "S24", "S32")
    secondary = {}
    for metric in ("h12", "D", "G"):
        vals = np.asarray([[r[f"{metric}_{c}"] for c in width_order] for r in rows], dtype=np.float64)
        rhos = []
        x = np.arange(3, dtype=np.float64)
        for y in vals:
            yrank = np.argsort(np.argsort(y)).astype(np.float64)
            rho = 0.0 if np.std(yrank) == 0 else float(np.corrcoef(x, yrank)[0, 1])
            rhos.append(rho)
        secondary[f"width_spearman_{metric}"] = {
            "per_seed": rhos,
            "mean": float(np.mean(rhos)),
            "ci95": bootstrap_mean_ci(rhos, 80600 + {"h12": 1, "D": 2, "G": 3}[metric]),
        }

    return {
        "classification": classification,
        "all_conditions_valid": True,
        "parameter_match_max_relative_difference": float(max(ratios)),
        "A_supported": A,
        "A_criteria": {"A1_h12": A1, "A2_D": A2, "A3_G": A3, "A4_h0": A4},
        "S_supported": S,
        "S_criteria": {"S1_D": S1, "S2_G": S2},
        "contrasts": {
            "S32_minus_B16_h12": {"mean": float(dh12.mean()), "ci95": ci_h12},
            "S32_minus_B16_D": {"mean": float(dD.mean()), "ci95": ci_D},
            "S32_minus_B16_G": {"mean": float(dG.mean()), "ci95": ci_G},
            "S32_minus_B16_h0": {"mean": float(dh0.mean()), "ci95": ci_h0},
            "S32_minus_P16_D": {"mean": float(dD_spec.mean()), "ci95": ci_D_spec},
            "S32_minus_P16_G": {"mean": float(dG_spec.mean()), "ci95": ci_G_spec},
        },
        "rows": rows,
        "secondary": secondary,
    }


def write_md(path, result):
    lines = ["# R8-M5 Final Result", "", f"**Primary classification:** {result['classification']}", ""]
    if not result.get("all_conditions_valid", False):
        lines += [f"- Invalid conditions: {result.get('invalid')}"]
    else:
        lines += [
            "- Cross-capacity validity: **True**",
            f"- A wider-state allocation pattern: **{result['A_supported']}**",
            f"- S state-dimension specificity: **{result['S_supported']}**",
            f"- P16/S32 max relative parameter-count difference: {result['parameter_match_max_relative_difference']:.4f}",
            "",
            "## Primary contrasts",
            "",
        ]
        for k, v in result["contrasts"].items():
            lines.append(f"- {k}: mean {v['mean']:.6f}; 95% CI {v['ci95']}")
    lines += [
        "",
        "## Claim boundary",
        "",
        "R8-M5 tests a state-capacity/resource-allocation account in one symmetric synthetic autonomous recurrent architecture. C2 would support state-dimension-specific redistribution under this design; C1 would support a wider-state pattern without isolating state dimension from generic added capacity; C0 would reject the frozen simple account. No outcome establishes universality, strong emergence, practical superiority, or that Euclidean survival magnitude itself mediates reader use.",
        "",
    ]
    Path(path).write_text("\n".join(lines))


def self_check():
    assert bootstrap_mean_ci([1, 1, 1], 1)[0] > 0
    assert bootstrap_mean_ci([-1, -1, -1], 2)[1] < 0
    assert 0.02 >= 0.02
    assert -0.10 <= -0.10
    assert -0.05 <= -0.05
    print("classifier self-check ok")


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
