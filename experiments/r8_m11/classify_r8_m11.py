import argparse
import json
from pathlib import Path

import numpy as np

SEEDS = (839, 856, 872, 891, 907, 926, 944, 963, 981, 1003, 1021, 1042)
N_BOOT = 5000
BOOT_SEED = 20260911

R_MIN = 0.50
F_MIN = 0.25
LAYER_MIN = 0.15
INTERACTION_MIN = 0.15


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
        if d.get("experiment") == "R8-M11":
            by_seed[int(d["seed"])] = d
    missing = [s for s in SEEDS if s not in by_seed]
    if missing:
        raise RuntimeError(f"missing R8-M11 seed summaries: {missing}")
    return [by_seed[s] for s in SEEDS]


def stat_block(x, seed):
    x = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "min": float(x.min()),
        "max": float(x.max()),
        "n_positive": int((x > 0).sum()),
        "n": int(x.size),
        "ci95": bootstrap_mean_ci(x, seed),
        "per_seed": [float(v) for v in x],
    }


def classify_vectors(h_parent, f_total, f1, f2, i12):
    stats = {
        "H_parent": stat_block(h_parent, BOOT_SEED),
        "F_total": stat_block(f_total, BOOT_SEED + 1),
        "F1_effect": stat_block(f1, BOOT_SEED + 2),
        "F2_effect": stat_block(f2, BOOT_SEED + 3),
        "I12": stat_block(i12, BOOT_SEED + 4),
    }

    r_pass = bool(
        stats["H_parent"]["mean"] >= R_MIN
        and stats["H_parent"]["ci95"][0] > 0
    )
    f_pass = bool(
        stats["F_total"]["mean"] >= F_MIN
        and stats["F_total"]["ci95"][0] > 0
    )
    f1_pass = bool(
        stats["F1_effect"]["mean"] >= LAYER_MIN
        and stats["F1_effect"]["ci95"][0] > 0
    )
    f2_pass = bool(
        stats["F2_effect"]["mean"] >= LAYER_MIN
        and stats["F2_effect"]["ci95"][0] > 0
    )

    i_ci = stats["I12"]["ci95"]
    interaction = bool(
        abs(stats["I12"]["mean"]) >= INTERACTION_MIN
        and (i_ci[0] > 0 or i_ci[1] < 0)
    )

    if not r_pass:
        classification = "R0 — matched-midpoint history effect not replicated strongly enough for substructure localization"
    elif not f_pass:
        classification = "F0 — parent history effect replicated, but recurrent-map carrier did not replicate strongly enough for intra-F localization"
    elif f1_pass and f2_pass:
        classification = "L3 — distributed two-stage contribution supported"
    elif f1_pass:
        classification = "L1 — input-stage contribution supported"
    elif f2_pass:
        classification = "L2 — output-stage contribution supported"
    else:
        classification = "L0 — intra-F localization unresolved"

    return {
        "classification": classification,
        "R_pass": r_pass,
        "F_pass": f_pass,
        "F1_pass": f1_pass,
        "F2_pass": f2_pass,
        "I12_supported": interaction,
        "stats": stats,
    }


def classify(summaries):
    maturity_bad = []
    execution_bad = []
    for d in summaries:
        v = d.get("validity", {})
        if not bool(v.get("maturity")):
            maturity_bad.append({"seed": int(d["seed"]), "validity": v})
            continue
        for key in ("fork_identity", "hybrid_hashes", "finite", "complete"):
            if not bool(v.get(key)):
                execution_bad.append({"seed": int(d["seed"]), "validity": v})
                break

    if maturity_bad:
        return {
            "classification": "V0 — maturity validity failure",
            "all_valid": False,
            "maturity_failures": maturity_bad,
        }
    if execution_bad:
        return {
            "classification": "V1 — post-maturity substructure execution failure",
            "all_valid": False,
            "execution_failures": execution_bad,
        }

    h = [float(d["effects"]["H_parent"]) for d in summaries]
    ft = [float(d["effects"]["F_total"]) for d in summaries]
    f1 = [float(d["effects"]["F1_effect"]) for d in summaries]
    f2 = [float(d["effects"]["F2_effect"]) for d in summaries]
    i12 = [float(d["effects"]["I12"]) for d in summaries]

    out = classify_vectors(h, ft, f1, f2, i12)
    out["all_valid"] = True
    out["rows"] = [
        {
            "seed": int(d["seed"]),
            "M": int(d["maturity_epoch"]),
            "A": int(d["A_baseline_winner"]),
            "B": int(d["B_baseline_loser"]),
            "H_parent": float(d["effects"]["H_parent"]),
            "F_total": float(d["effects"]["F_total"]),
            "F1_effect": float(d["effects"]["F1_effect"]),
            "F2_effect": float(d["effects"]["F2_effect"]),
            "I12": float(d["effects"]["I12"]),
            "F1_fraction": d["effects"].get("F1_fraction"),
            "F2_fraction": d["effects"].get("F2_fraction"),
        }
        for d in summaries
    ]
    return out


def render_md(result):
    lines = ["# R8-M11 Final Result", ""]
    lines += [f"**Primary classification:** {result['classification']}", ""]
    if not result.get("all_valid"):
        lines += ["Execution validity failed; no mechanistic localization is promoted.", ""]
        return "\n".join(lines)

    lines += [
        f"- Parent replication R: **{result['R_pass']}**",
        f"- Recurrent carrier replication F: **{result['F_pass']}**",
        f"- Input-stage F1 criterion: **{result['F1_pass']}**",
        f"- Output-stage F2 criterion: **{result['F2_pass']}**",
        f"- F1×F2 interaction descriptor supported: **{result['I12_supported']}**",
        "",
        "## Frozen primary statistics",
        "",
    ]
    for key in ("H_parent", "F_total", "F1_effect", "F2_effect", "I12"):
        s = result["stats"][key]
        lines.append(
            f"- {key}: mean {s['mean']:.6f}; median {s['median']:.6f}; "
            f"95% CI [{s['ci95'][0]:.6f}, {s['ci95'][1]:.6f}]; "
            f"positive {s['n_positive']}/{s['n']}"
        )

    lines += ["", "## Per-family values", ""]
    for r in result["rows"]:
        lines.append(
            f"- seed {r['seed']}: M={r['M']}, A={r['A']}, B={r['B']}, "
            f"H={r['H_parent']:+.6f}, F={r['F_total']:+.6f}, "
            f"F1={r['F1_effect']:+.6f}, F2={r['F2_effect']:+.6f}, I12={r['I12']:+.6f}"
        )

    lines += [
        "",
        "## Claim boundary",
        "",
        "R8-M11 localizes causal contribution only to the frozen two-stage parameter partition inside the tested recurrent map. "
        "It does not establish unique storage, individual-neuron or low-rank causality, formal bistability/hysteresis, "
        "information beyond the complete state, or generalization beyond this synthetic architecture.",
        "",
    ]
    return "\n".join(lines)


def self_check():
    n = len(SEEDS)
    c = lambda x: [x] * n

    assert classify_vectors(c(0.1), c(0.5), c(0.25), c(0.25), c(0.0))["classification"].startswith("R0")
    assert classify_vectors(c(0.8), c(0.1), c(0.05), c(0.05), c(0.0))["classification"].startswith("F0")
    assert classify_vectors(c(0.8), c(0.5), c(0.4), c(0.1), c(0.0))["classification"].startswith("L1")
    assert classify_vectors(c(0.8), c(0.5), c(0.1), c(0.4), c(0.0))["classification"].startswith("L2")
    assert classify_vectors(c(0.8), c(0.5), c(0.25), c(0.25), c(0.2))["classification"].startswith("L3")
    assert classify_vectors(c(0.8), c(0.5), c(0.1), c(0.1), c(0.0))["classification"].startswith("L0")
    print("R8-M11 classifier self-check ok")


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
        raise SystemExit("--root and --outdir are required unless --self-check is used")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    result = classify(load_summaries(args.root))
    save_json(outdir / "FINAL_RESULT.json", result)
    (outdir / "FINAL_RESULT.md").write_text(render_md(result))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2, default=str))


if __name__ == "__main__":
    main()
