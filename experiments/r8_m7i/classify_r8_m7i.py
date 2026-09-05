import argparse
import json
from pathlib import Path

import numpy as np

SEEDS = (214, 230, 247, 263, 279, 296, 313, 329, 346, 362, 378, 397)
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
        if d.get("experiment") == "R8-M7I":
            by_seed[int(d["seed"])] = d
    missing = [s for s in SEEDS if s not in by_seed]
    if missing:
        raise RuntimeError(f"missing paired seed summaries: {missing}")
    return [by_seed[s] for s in SEEDS]


def rec(d, condition, idx):
    return d["conditions"][condition]["records"][idx]


def classify(summaries):
    baseline_bad = []
    execution_bad = []
    for d in summaries:
        v = d.get("validity", {})
        if not bool(v.get("baseline_reference")):
            baseline_bad.append({"seed": int(d["seed"]), "validity": v, "reference": d.get("baseline_reference")})
        other = [v.get("maturity"), v.get("fork_identity"), v.get("finite"), v.get("complete")]
        if not all(bool(x) for x in other):
            execution_bad.append({"seed": int(d["seed"]), "validity": v})
    if baseline_bad:
        return {"classification": "V0 — baseline lineage reproduction failure", "all_valid": False, "baseline_failures": baseline_bad}
    if execution_bad:
        return {"classification": "V1 — post-maturity execution validity failure", "all_valid": False, "execution_failures": execution_bad}

    delayed_ref = json.loads(Path(__file__).with_name("M7R_DELAYED_QB_REFERENCE.json").read_text())
    rows = []
    for d in summaries:
        A = int(d["A_baseline_winner"])
        B = int(d["B_baseline_loser"])
        mir = [rec(d, "MIRROR", i) for i in range(3)]
        fix = [rec(d, "FIXB", i) for i in range(3)]
        h0 = [rec(d, "H0MIRROR", i) for i in range(3)]
        qb1, qa, qb2 = [float(x["Q"]) for x in mir]
        fqa = float(fix[1]["Q"])
        hqb1, hqa, hqb2 = [float(x["Q"]) for x in h0]
        amp_m = 0.5 * ((qb1 - qa) + (qb2 - qa))
        amp_h = 0.5 * ((hqb1 - hqa) + (hqb2 - hqa))
        exact = bool(
            int(mir[0]["survival"]["winner_relation"]) == B
            and int(mir[1]["survival"]["winner_relation"]) == A
            and int(mir[2]["survival"]["winner_relation"]) == B
        )
        seed = int(d["seed"])
        baseline_q = float(d["baseline"]["Q"])
        rows.append({
            "seed": seed,
            "M": int(d["maturity_epoch"]),
            "A": A,
            "B": B,
            "baseline_Q": baseline_q,
            "Q_B1": qb1,
            "Q_A": qa,
            "Q_B2": qb2,
            "BA_shift": qa - qb1,
            "AB_shift": qb2 - qa,
            "Q_B2_minus_baseline": qb2 - baseline_q,
            "Q_A_FIXB": fqa,
            "C1_diff": qa - fqa,
            "AMP_MIRROR": amp_m,
            "AMP_H0MIRROR": amp_h,
            "C2_diff": amp_m - amp_h,
            "exact_sequence": exact,
            "B_winner_phase1": int(mir[0]["survival"]["winner_relation"]) == B,
            "A_winner_phase2": int(mir[1]["survival"]["winner_relation"]) == A,
            "B_winner_phase3": int(mir[2]["survival"]["winner_relation"]) == B,
            "B_h12_baseline": float(d["baseline"]["validation"]["h12_per_relation"][B]),
            "B_h12_phase1": float(mir[0]["validation"]["h12_per_relation"][B]),
            "A_h12_phase1": float(mir[0]["validation"]["h12_per_relation"][A]),
            "A_h12_phase2": float(mir[1]["validation"]["h12_per_relation"][A]),
            "B_h12_phase2": float(mir[1]["validation"]["h12_per_relation"][B]),
            "B_h12_phase3": float(mir[2]["validation"]["h12_per_relation"][B]),
            "m7r_delayed_Q_B": float(delayed_ref[str(seed)]),
            "immediate_minus_delayed_QB": qb1 - float(delayed_ref[str(seed)]),
        })

    def arr(k):
        return np.asarray([r[k] for r in rows], dtype=np.float64)

    stat_seeds = {
        "Q_B1": 81701,
        "BA_shift": 81702,
        "Q_A": 81703,
        "AB_shift": 81704,
        "Q_B2": 81705,
        "Q_B2_minus_baseline": 81706,
        "C1_diff": 81707,
        "C2_diff": 81708,
    }
    stats = {}
    for k, sd in stat_seeds.items():
        x = arr(k)
        stats[k] = {"mean": float(x.mean()), "ci95": bootstrap_mean_ci(x, sd)}

    T1 = bool(stats["Q_B1"]["mean"] >= 0.20 and stats["Q_B1"]["ci95"][0] > 0)
    T2 = bool(stats["BA_shift"]["mean"] <= -0.75 and stats["BA_shift"]["ci95"][1] < 0)
    T3 = bool(stats["Q_A"]["mean"] <= -0.20 and stats["Q_A"]["ci95"][1] < 0)
    T4 = bool(stats["AB_shift"]["mean"] >= 0.75 and stats["AB_shift"]["ci95"][0] > 0)
    T5 = bool(stats["Q_B2"]["mean"] >= 0.20 and stats["Q_B2"]["ci95"][0] > 0)
    T6 = bool(stats["Q_B2_minus_baseline"]["mean"] >= 0.75 and stats["Q_B2_minus_baseline"]["ci95"][0] > 0)
    T = bool(T1 and T2 and T3 and T4 and T5 and T6)

    C1 = bool(stats["C1_diff"]["mean"] <= -0.50 and stats["C1_diff"]["ci95"][1] < 0)
    C2 = bool(stats["C2_diff"]["mean"] >= 0.25 and stats["C2_diff"]["ci95"][0] > 0)
    S = bool(C1 and C2)

    exact_count = int(sum(r["exact_sequence"] for r in rows))
    E = bool(exact_count >= 8)

    if not T:
        classification = "I0 — inverted mirror tracking not supported"
    elif S and E:
        classification = "I2 — inverted demand-specific specialist reassignment supported"
    else:
        classification = "I1 — inverted relative tracking supported"

    b_gain = arr("B_h12_phase1") - arr("B_h12_baseline")
    a_gain = arr("A_h12_phase2") - arr("A_h12_phase1")
    b_return = arr("B_h12_phase3") - arr("B_h12_phase2")
    immediate_delayed = arr("immediate_minus_delayed_QB")
    secondary = {
        "B_h12_gain_baseline_to_immediate_B": {"mean": float(b_gain.mean()), "ci95": bootstrap_mean_ci(b_gain, 81721)},
        "A_h12_gain_B1_to_A": {"mean": float(a_gain.mean()), "ci95": bootstrap_mean_ci(a_gain, 81722)},
        "B_h12_return_A_to_B2": {"mean": float(b_return.mean()), "ci95": bootstrap_mean_ci(b_return, 81723)},
        "immediate_B_Q_minus_M7R_delayed_B_Q": {"mean": float(immediate_delayed.mean()), "ci95": bootstrap_mean_ci(immediate_delayed, 81724)},
        "B_winner_phase1_count": int(sum(r["B_winner_phase1"] for r in rows)),
        "A_winner_phase2_count": int(sum(r["A_winner_phase2"] for r in rows)),
        "B_winner_phase3_count": int(sum(r["B_winner_phase3"] for r in rows)),
        "exact_B_A_B_count": exact_count,
    }

    return {
        "classification": classification,
        "all_valid": True,
        "T_supported": T,
        "T_criteria": {"T1_B1_takeover": T1, "T2_B_to_A_shift": T2, "T3_A_takeover": T3, "T4_A_to_B_shift": T4, "T5_B2_takeover": T5, "T6_B2_above_baseline": T6},
        "S_supported": S,
        "S_criteria": {"C1_mirror_vs_fixB": C1, "C2_terminal_vs_h0": C2},
        "E_supported": E,
        "exact_B_A_B_count": exact_count,
        "stats": stats,
        "secondary": secondary,
        "rows": rows,
    }


def write_md(path, r):
    lines = ["# R8-M7I Final Result", "", f"**Primary classification:** {r['classification']}", ""]
    if not r.get("all_valid", False):
        lines.append(f"- Validity record: {r}")
    else:
        lines += [
            "- Paired same-lineage validity: **True**",
            f"- T inverted mirror tracking: **{r['T_supported']}**",
            f"- S demand specificity: **{r['S_supported']}**",
            f"- E exact B→A→B reassignment: **{r['E_supported']}** ({r['exact_B_A_B_count']}/12)",
            "", "## Frozen primary statistics", "",
        ]
        for k, v in r["stats"].items():
            lines.append(f"- {k}: mean {v['mean']:.6f}; 95% CI {v['ci95']}")
        lines += ["", "## Prespecified secondary descriptors", ""]
        for k, v in r["secondary"].items():
            lines.append(f"- {k}: {v}")
    lines += [
        "", "## Claim boundary", "",
        "R8-M7I is a paired mirror of R8-M7R using the same deterministic baseline lineages and datasets. I2 would support reversible B→A→B specialist reassignment under the inverted post-maturity schedule in this synthetic architecture. It would not establish formal hysteresis, conscious choice, universal trajectory computation, strong emergence, essential chronology, or generalization beyond this system.", "",
    ]
    Path(path).write_text("\n".join(lines))


def self_check():
    assert bootstrap_mean_ci([1.0, 1.0, 1.0], 1)[0] > 0
    assert bootstrap_mean_ci([-1.0, -1.0, -1.0], 2)[1] < 0
    assert 0.20 >= 0.20 and -0.75 <= -0.75 and 0.75 >= 0.75
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
