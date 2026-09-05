import argparse
import json
from pathlib import Path

import numpy as np

FRESH_SEEDS = (22, 36, 49, 67, 82, 97, 113, 129, 144, 159, 177, 193)
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
        if d.get("experiment") == "R8-M6":
            by_seed[int(d["seed"])] = d
    missing = [s for s in FRESH_SEEDS if s not in by_seed]
    if missing:
        raise RuntimeError(f"missing fresh seed summaries: {missing}")
    return [by_seed[s] for s in FRESH_SEEDS]


def equiv(mean, ci):
    return bool(abs(mean) <= 0.02 and ci[0] >= -0.03 and ci[1] <= 0.03)


def extract(d, name):
    e = d["epoch100"][name]
    return {
        "h0": float(e["test"]["h0_overall"]),
        "h12": float(e["test"]["h12_overall"]),
        "D": float(e["D_test"]),
        "G": float(e["test_survival"]["G"]),
        "winner": int(e["test_survival"]["winner_relation"]),
        "params": int(e["parameter_count"]),
    }


def classify(summaries):
    fork_fail = [int(d["seed"]) for d in summaries if not bool(d["fork_equivalence"]["passed"])]
    if fork_fail:
        return {
            "classification": "V0 — fork-equivalence/design failure",
            "fork_equivalence_all": False,
            "failed_seeds": fork_fail,
        }

    invalid = []
    for d in summaries:
        for name, ok in d["training_validity"].items():
            if not bool(ok):
                invalid.append({"seed": int(d["seed"]), "condition": name})
    if invalid:
        return {
            "classification": "V1 — post-fork training validity failure",
            "fork_equivalence_all": True,
            "training_valid_all": False,
            "invalid": invalid,
        }

    rows = []
    xb = {k: [] for k in ("h0", "h12", "D", "G")}
    xp = {k: [] for k in ("h0", "h12", "D", "G")}
    param_match = []

    for d in summaries:
        b = extract(d, "B16")
        x = extract(d, "X32")
        p = extract(d, "P16")
        for k in xb:
            xb[k].append(x[k] - b[k])
            xp[k].append(x[k] - p[k])
        param_match.append(abs(x["params"] - p["params"]) / max(x["params"], p["params"]))
        rows.append({
            "seed": int(d["seed"]),
            "B16": b,
            "X32": x,
            "P16": p,
            "Delta_XB": {k: x[k] - b[k] for k in xb},
            "Delta_XP": {k: x[k] - p[k] for k in xp},
        })

    seeds = {"h0": 80601, "h12": 80602, "D": 80603, "G": 80604}
    xb_stats = {}
    xp_stats = {}
    for k in xb:
        xb_stats[k] = {"mean": float(np.mean(xb[k])), "ci95": bootstrap_mean_ci(xb[k], seeds[k])}
        xp_stats[k] = {"mean": float(np.mean(xp[k])), "ci95": bootstrap_mean_ci(xp[k], seeds[k] + 100)}

    w = bool(
        xb_stats["h12"]["mean"] >= 0.02 and xb_stats["h12"]["ci95"][0] > 0 and
        xb_stats["D"]["mean"] >= 0.10 and xb_stats["D"]["ci95"][0] > 0 and
        xb_stats["G"]["mean"] > 0 and xb_stats["G"]["ci95"][0] > 0 and
        equiv(xb_stats["h0"]["mean"], xb_stats["h0"]["ci95"])
    )
    s = bool(
        xp_stats["h12"]["mean"] >= 0.015 and xp_stats["h12"]["ci95"][0] > 0 and
        xp_stats["D"]["mean"] >= 0.05 and xp_stats["D"]["ci95"][0] > 0 and
        xp_stats["G"]["mean"] > 0 and xp_stats["G"]["ci95"][0] > 0 and
        equiv(xp_stats["h0"]["mean"], xp_stats["h0"]["ci95"]) and
        max(param_match) < 0.05
    )

    if w and s:
        cls = "W2 — recurrent-state-dimension-specific workspace effect supported"
    elif w:
        cls = "W1 — workspace-expanded continuation pattern supported, state specificity not established"
    else:
        cls = "W0 — isolated recurrent-workspace account not supported"

    return {
        "classification": cls,
        "fork_equivalence_all": True,
        "training_valid_all": True,
        "W_supported": w,
        "S_supported": s,
        "X32_minus_B16": xb_stats,
        "X32_minus_P16": xp_stats,
        "h0_equivalence_XB": equiv(xb_stats["h0"]["mean"], xb_stats["h0"]["ci95"]),
        "h0_equivalence_XP": equiv(xp_stats["h0"]["mean"], xp_stats["h0"]["ci95"]),
        "max_parameter_relative_difference_X32_P16": float(max(param_match)),
        "rows": rows,
    }


def fake_summary(seed, fork=True, valid=True, xb=(0.0, 0.03, 0.15, 0.10), xp=(0.0, 0.02, 0.08, 0.05)):
    b = {"h0": 0.60, "h12": 0.20, "D": 0.30, "G": 0.70}
    x = {k: b[k] + v for k, v in zip(("h0", "h12", "D", "G"), xb)}
    p = {k: x[k] - v for k, v in zip(("h0", "h12", "D", "G"), xp)}

    def pack(v, params):
        return {
            "test": {"h0_overall": v["h0"], "h12_overall": v["h12"]},
            "D_test": v["D"],
            "test_survival": {"G": v["G"], "winner_relation": 0},
            "parameter_count": params,
        }

    return {
        "experiment": "R8-M6",
        "seed": seed,
        "fork_equivalence": {"passed": fork},
        "training_validity": {"B16": valid, "X32": valid, "P16": valid},
        "epoch100": {"B16": pack(b, 10944), "X32": pack(x, 14032), "P16": pack(p, 14046)},
    }


def self_check():
    good = [fake_summary(s) for s in FRESH_SEEDS]
    assert classify(good)["classification"].startswith("W2")
    weak = [fake_summary(s, xb=(0.0, 0.005, 0.01, 0.0), xp=(0.0, 0.0, 0.0, 0.0)) for s in FRESH_SEEDS]
    assert classify(weak)["classification"].startswith("W0")
    bad = good.copy()
    bad[0] = fake_summary(FRESH_SEEDS[0], fork=False)
    assert classify(bad)["classification"].startswith("V0")
    print("R8-M6 classifier self-check: OK")


def write_md(path, result):
    lines = ["# R8-M6 Final Result", "", f"**Primary classification:** {result['classification']}", ""]
    if result["classification"].startswith("V0"):
        lines += [f"- Fork-equivalence failed seeds: {result.get('failed_seeds')}"]
    elif result["classification"].startswith("V1"):
        lines += [f"- Invalid condition records: {result.get('invalid')}"]
    else:
        lines += [
            f"- Fork equivalence: **{result['fork_equivalence_all']}**",
            f"- Cross-condition training validity: **{result['training_valid_all']}**",
            f"- W workspace pattern: **{result['W_supported']}**",
            f"- S state specificity: **{result['S_supported']}**",
            f"- Max X32/P16 parameter-count relative difference: {result['max_parameter_relative_difference_X32_P16']:.6f}",
            "",
            "## X32 minus B16",
        ]
        for k in ("h12", "D", "G", "h0"):
            s = result["X32_minus_B16"][k]
            lines.append(f"- {k}: mean {s['mean']:.6f}; 95% CI {s['ci95']}")
        lines += ["", "## X32 minus P16"]
        for k in ("h12", "D", "G", "h0"):
            s = result["X32_minus_P16"][k]
            lines.append(f"- {k}: mean {s['mean']:.6f}; 95% CI {s['ci95']}")
        lines += [
            "",
            f"- h0 equivalence X32/B16: **{result['h0_equivalence_XB']}**",
            f"- h0 equivalence X32/P16: **{result['h0_equivalence_XP']}**",
        ]
    lines += [
        "",
        "## Claim boundary",
        "",
        "R8-M6 is a local causal architecture test of post-encoding recurrent workspace dimension in one symmetric synthetic autonomous recurrent system. W2 would not establish a universal trajectory-information principle, strong emergence, essential chronology, language-model generalization, or that Euclidean survival magnitude itself mediates reader usefulness.",
        "",
    ]
    Path(path).write_text("\n".join(lines))


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
