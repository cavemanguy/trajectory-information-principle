import argparse
import json
from pathlib import Path

import numpy as np

SEEDS = (631, 648, 664, 681, 699, 716, 733, 751, 768, 784, 802, 821)
N_BOOT = 5000
HOLD_CHECKS = (30, 60, 90, 120)


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
        if d.get("experiment") == "R8-M9":
            by_seed[int(d["seed"])] = d
    missing = [s for s in SEEDS if s not in by_seed]
    if missing:
        raise RuntimeError(f"missing M9 seed summaries: {missing}")
    return [by_seed[s] for s in SEEDS]


def stat(x, seed):
    x = np.asarray(x, dtype=np.float64)
    return {"mean": float(x.mean()), "ci95": bootstrap_mean_ci(x, seed)}


def hold_q(d, mode, side, hold_epoch):
    recs = d["holds"][mode][side]["records"]
    for r in recs:
        if int(r["hold_epoch"]) == int(hold_epoch):
            return float(r["Q"])
    raise KeyError((mode, side, hold_epoch))


def classify(summaries):
    maturity_bad = []
    execution_bad = []
    for d in summaries:
        v = d.get("validity", {})
        if not bool(v.get("maturity")):
            maturity_bad.append(int(d["seed"]))
        elif not all(bool(v.get(k)) for k in ("fork_identity", "transplants", "holds", "finite", "complete")):
            execution_bad.append({"seed": int(d["seed"]), "validity": v})
    if maturity_bad:
        return {"classification": "V0 — maturity validity failure", "all_valid": False, "maturity_failures": maturity_bad}
    if execution_bad:
        return {"classification": "V1 — post-maturity localization execution failure", "all_valid": False, "execution_failures": execution_bad}

    rows = []
    for d in summaries:
        e = d["effects"]
        q = {k: float(d["transplants"][k]["record"]["Q"]) for k in ("AA", "AB", "BA", "BB")}
        row = {
            "seed": int(d["seed"]),
            "M": int(d["maturity_epoch"]),
            "A": int(d["A_baseline_winner"]),
            "B": int(d["B_baseline_loser"]),
            "Q_AA": q["AA"], "Q_AB": q["AB"], "Q_BA": q["BA"], "Q_BB": q["BB"],
            "H_parent": float(e["H_parent"]),
            "E_effect": float(e["E_effect"]),
            "F_effect": float(e["F_effect"]),
            "I_EF": float(e["I_EF"]),
            "latent_h0_distance": float(d["midpoint_latent_distance"]["mean_h0_distance"]),
            "latent_h12_distance": float(d["midpoint_latent_distance"]["mean_h12_distance"]),
        }
        hp = row["H_parent"]
        row["E_fraction"] = None if abs(hp) < 1e-12 else row["E_effect"] / hp
        row["F_fraction"] = None if abs(hp) < 1e-12 else row["F_effect"] / hp
        for mode in ("INHERITED", "RESET", "CROSSED"):
            for h in HOLD_CHECKS:
                row[f"H_{mode}_{h}"] = hold_q(d, mode, "B", h) - hold_q(d, mode, "A", h)
        rows.append(row)

    stats = {
        "H_parent": stat([r["H_parent"] for r in rows], 82901),
        "E_effect": stat([r["E_effect"] for r in rows], 82902),
        "F_effect": stat([r["F_effect"] for r in rows], 82903),
        "I_EF": stat([r["I_EF"] for r in rows], 82904),
        "H_reset120": stat([r["H_RESET_120"] for r in rows], 82905),
        "H_inherited120": stat([r["H_INHERITED_120"] for r in rows], 82906),
        "H_crossed120": stat([r["H_CROSSED_120"] for r in rows], 82907),
        "latent_h0_distance": stat([r["latent_h0_distance"] for r in rows], 82908),
        "latent_h12_distance": stat([r["latent_h12_distance"] for r in rows], 82909),
    }

    R = bool(stats["H_parent"]["mean"] >= 0.50 and stats["H_parent"]["ci95"][0] > 0)
    if not R:
        classification = "R0 — persistent-history midpoint phenomenon not replicated strongly enough for localization"
        return {
            "classification": classification, "all_valid": True, "R_supported": False,
            "stats": stats, "rows": rows,
            "claim_boundary": "No component-localization claim is promoted because the fresh parent history effect missed the frozen replication gate.",
        }

    E = bool(stats["E_effect"]["mean"] >= 0.25 and stats["E_effect"]["ci95"][0] > 0)
    F = bool(stats["F_effect"]["mean"] >= 0.25 and stats["F_effect"]["ci95"][0] > 0)
    if E and F:
        classification = "C3 — distributed encoder + recurrent contribution supported"
    elif E:
        classification = "C1 — encoder-carried contribution supported"
    elif F:
        classification = "C2 — recurrent-map-carried contribution supported"
    else:
        classification = "C0 — component localization unresolved"

    O = bool(stats["H_reset120"]["mean"] >= 0.25 and stats["H_reset120"]["ci95"][0] > 0)
    ii = stats["I_EF"]
    interaction = bool(abs(ii["mean"]) >= 0.25 and (ii["ci95"][0] > 0 or ii["ci95"][1] < 0))

    hold_curves = {}
    for mode_i, mode in enumerate(("INHERITED", "RESET", "CROSSED")):
        hold_curves[mode] = {}
        for j, h in enumerate(HOLD_CHECKS):
            hold_curves[mode][str(h)] = stat([r[f"H_{mode}_{h}"] for r in rows], 82920 + 10 * mode_i + j)

    reader_keys = sorted(summaries[0]["reader_transfer_validation"].keys())
    reader_summary = {}
    for key_i, key in enumerate(reader_keys):
        reader_summary[key] = {}
        for metric_i, metric in enumerate(("h0_overall", "h12_overall", "h12_A", "h12_B", "h12_mid_weighted")):
            reader_summary[key][metric] = stat(
                [float(d["reader_transfer_validation"][key][metric]) for d in summaries],
                83000 + key_i * 10 + metric_i,
            )

    return {
        "classification": classification,
        "all_valid": True,
        "R_supported": True,
        "E_supported": E,
        "F_supported": F,
        "O_optimizer_reset_persistence_supported": O,
        "optimizer_interpretation": (
            "optimizer state not necessary for 120-epoch persistence under the reset intervention" if O
            else "optimizer-state necessity not ruled out"
        ),
        "EF_interaction_supported": interaction,
        "stats": stats,
        "hold_curves": hold_curves,
        "reader_transfer_validation": reader_summary,
        "rows": rows,
        "claim_boundary": "Component transplants localize causal contribution within this architecture; they do not establish formal bistability/hysteresis or generalization beyond the tested synthetic system.",
    }


def write_md(path, r):
    lines = ["# R8-M9 Final Result", "", f"**Primary classification:** {r['classification']}", ""]
    if not r.get("all_valid", False):
        lines.append(f"- Validity record: {r}")
    elif not r.get("R_supported", False):
        s = r["stats"]["H_parent"]
        lines += ["- Fresh midpoint phenomenon replication: **False**", f"- H_parent: mean {s['mean']:.6f}; 95% CI {s['ci95']}"]
    else:
        lines += [
            "- Fresh midpoint phenomenon replication R: **True**",
            f"- Encoder contribution E: **{r['E_supported']}**",
            f"- Recurrent contribution F: **{r['F_supported']}**",
            f"- Optimizer-reset persistence O: **{r['O_optimizer_reset_persistence_supported']}**",
            f"- EF interaction descriptor supported: **{r['EF_interaction_supported']}**",
            "", "## Frozen primary statistics", "",
        ]
        for k in ("H_parent", "E_effect", "F_effect", "I_EF", "H_reset120", "H_inherited120", "H_crossed120"):
            v = r["stats"][k]
            lines.append(f"- {k}: mean {v['mean']:.6f}; 95% CI {v['ci95']}")
        lines += ["", "## Optimizer hold curves", ""]
        for mode, dd in r["hold_curves"].items():
            vals = ", ".join(f"+{h}: {v['mean']:.6f} {v['ci95']}" for h, v in dd.items())
            lines.append(f"- {mode}: {vals}")
    lines += ["", "## Claim boundary", "", r.get("claim_boundary", ""), ""]
    Path(path).write_text("\n".join(lines))


def self_check():
    assert bootstrap_mean_ci([1.0, 1.0, 1.0], 1)[0] > 0
    assert bootstrap_mean_ci([-1.0, -1.0, -1.0], 2)[1] < 0
    qa, qab, qba, qb = -1.0, -0.5, 0.0, 0.5
    e = 0.5 * ((qba - qa) + (qb - qab))
    f = 0.5 * ((qab - qa) + (qb - qba))
    assert abs((e + f) - (qb - qa)) < 1e-12
    print("R8-M9 classifier self-check ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root")
    ap.add_argument("--outdir")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        self_check(); return
    if not args.root or not args.outdir:
        raise SystemExit("--root and --outdir required")
    summaries = load_summaries(args.root)
    result = classify(summaries)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    save_json(out / "FINAL_RESULT.json", result)
    write_md(out / "FINAL_RESULT.md", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
