import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

EXPECTED = (1782, 1801, 1819, 1838, 1856, 1875, 1893, 1912, 1930, 1949, 1967, 1986)
INTERNAL_TIMES = tuple(range(1, 12))
REPS = ("STATE", "VELOCITY", "STATE_VELOCITY", "STATE_SHUFFLED_VELOCITY", "STATE_QUADRATIC")
MIN_EFFECT = 0.005
MIN_POSITIVE = 10
N_BOOT = 10000
BOOT_SEED = 83021


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def bootstrap_mean_ci(x):
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(BOOT_SEED)
    ix = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    means = x[ix].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def gate(x):
    x = np.asarray(x, dtype=np.float64)
    ci = bootstrap_mean_ci(x)
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "min": float(x.min()),
        "max": float(x.max()),
        "bootstrap_95_ci": ci,
        "positive_lineages": int((x > 0).sum()),
        "min_effect_required": MIN_EFFECT,
        "min_positive_required": MIN_POSITIVE,
        "pass": bool(x.mean() >= MIN_EFFECT and ci[0] > 0 and int((x > 0).sum()) >= MIN_POSITIVE),
    }


def find_summaries(root):
    paths = sorted(Path(root).rglob("seed_summary.json"))
    rows = {}
    for p in paths:
        d = json.loads(p.read_text())
        if d.get("experiment") != "R8-AD3":
            continue
        seed = int(d["seed"])
        if seed in rows:
            raise RuntimeError(f"duplicate seed summary {seed}")
        rows[seed] = d
    return rows


def classify(rows):
    valid = []
    for seed in EXPECTED:
        d = rows.get(seed)
        if d is None:
            continue
        q = d.get("diagnostic", {})
        complete = all(str(t) in q.get("by_time", {}) for t in INTERNAL_TIMES + (12,))
        complete = complete and all(
            rep in q["by_time"][str(t)]
            for t in INTERNAL_TIMES + (12,)
            for rep in REPS
        ) if complete else False
        if d.get("all_valid") is True and q.get("finite") is True and complete:
            valid.append(d)

    if len(valid) < 12:
        return {
            "experiment": "R8-AD3",
            "classification": "H4 — invalid/incomplete",
            "n_valid": len(valid),
            "expected_seeds": list(EXPECTED),
        }

    valid = sorted(valid, key=lambda d: int(d["seed"]))
    ds = np.asarray([d["diagnostic"]["delta_state"] for d in valid], dtype=np.float64)
    dh = np.asarray([d["diagnostic"]["delta_shuffle"] for d in valid], dtype=np.float64)
    dq = np.asarray([d["diagnostic"]["delta_quadratic"] for d in valid], dtype=np.float64)

    g_state = gate(ds)
    g_shuffle = gate(dh)
    if g_state["pass"] and g_shuffle["pass"]:
        classification = "H1 — one-step path-history accessibility supported"
    elif g_state["pass"] or g_shuffle["pass"]:
        classification = "H2 — partial/ambiguous path-history accessibility"
    else:
        classification = "H3 — no preregistered path-history accessibility"

    family_rows = []
    for d in valid:
        q = d["diagnostic"]
        family_rows.append({
            "seed": int(d["seed"]),
            "M": int(d["maturity_epoch"]),
            "ad2_regime": q.get("ad2_regime_secondary"),
            "state": float(q["primary"]["STATE"]["mean_accuracy_internal"]),
            "velocity": float(q["primary"]["VELOCITY"]["mean_accuracy_internal"]),
            "joint": float(q["primary"]["STATE_VELOCITY"]["mean_accuracy_internal"]),
            "shuffle": float(q["primary"]["STATE_SHUFFLED_VELOCITY"]["mean_accuracy_internal"]),
            "quadratic": float(q["primary"]["STATE_QUADRATIC"]["mean_accuracy_internal"]),
            "delta_state": float(q["delta_state"]),
            "delta_shuffle": float(q["delta_shuffle"]),
            "delta_quadratic": float(q["delta_quadratic"]),
        })

    per_time = {}
    for t in INTERNAL_TIMES + (12,):
        rec = {}
        for rep in REPS:
            vals = np.asarray([
                d["diagnostic"]["by_time"][str(t)][rep]["test_mean_accuracy"]
                for d in valid
            ], dtype=np.float64)
            rec[rep] = {
                "mean": float(vals.mean()),
                "median": float(np.median(vals)),
            }
        delta = np.asarray([
            d["diagnostic"]["by_time"][str(t)]["STATE_VELOCITY"]["test_mean_accuracy"]
            - d["diagnostic"]["by_time"][str(t)]["STATE"]["test_mean_accuracy"]
            for d in valid
        ], dtype=np.float64)
        rec["DELTA_STATE"] = {
            "mean": float(delta.mean()),
            "median": float(np.median(delta)),
        }
        per_time[str(t)] = rec

    per_relation = {}
    for r in range(8):
        delta = np.asarray([
            d["diagnostic"]["primary"]["STATE_VELOCITY"]["per_relation_accuracy_internal"][r]
            - d["diagnostic"]["primary"]["STATE"]["per_relation_accuracy_internal"][r]
            for d in valid
        ], dtype=np.float64)
        per_relation[str(r)] = {
            "mean_delta_joint_minus_state": float(delta.mean()),
            "median_delta_joint_minus_state": float(np.median(delta)),
            "positive_lineages": int((delta > 0).sum()),
        }

    regime_groups = defaultdict(list)
    for row in family_rows:
        regime_groups[str(row["ad2_regime"])].append(row["delta_state"])
    regime_secondary = {
        k: {
            "n": len(v),
            "mean_delta_state": float(np.mean(v)),
            "median_delta_state": float(np.median(v)),
        }
        for k, v in sorted(regime_groups.items())
    }

    endpoint_delta = np.asarray([
        d["diagnostic"]["by_time"]["12"]["STATE_VELOCITY"]["test_mean_accuracy"]
        - d["diagnostic"]["by_time"]["12"]["STATE"]["test_mean_accuracy"]
        for d in valid
    ], dtype=np.float64)

    claim = (
        "R8-AD3 tests whether correctly paired one-step native trajectory history improves simple linear "
        "readout accessibility relative to the instantaneous state and a dimension-matched shuffled-history "
        "control. It does not establish information beyond the complete Markov state-plus-map, causal necessity, "
        "essential chronology, or a universal trajectory code."
    )

    return {
        "experiment": "R8-AD3",
        "classification": classification,
        "n_valid": 12,
        "expected_seeds": list(EXPECTED),
        "gates": {"G_STATE": g_state, "G_SHUFFLE": g_shuffle},
        "delta_quadratic_secondary": {
            "mean": float(dq.mean()),
            "median": float(np.median(dq)),
            "bootstrap_95_ci": bootstrap_mean_ci(dq),
            "positive_lineages": int((dq > 0).sum()),
        },
        "endpoint_t12_delta_joint_minus_state_secondary": {
            "mean": float(endpoint_delta.mean()),
            "median": float(np.median(endpoint_delta)),
            "bootstrap_95_ci": bootstrap_mean_ci(endpoint_delta),
        },
        "per_time_secondary": per_time,
        "per_relation_secondary": per_relation,
        "ad2_regime_groups_secondary": regime_secondary,
        "rows": family_rows,
        "claim_boundary": claim,
    }


def write_markdown(result, path):
    lines = [
        "# R8-AD3 Final Result — One-Step Path-History Accessibility",
        "",
        f"**Primary classification:** {result['classification']}",
        "",
    ]
    if result["classification"].startswith("H4"):
        lines += [f"Valid lineages: **{result['n_valid']}/12**", ""]
    else:
        gs = result["gates"]["G_STATE"]
        gh = result["gates"]["G_SHUFFLE"]
        dq = result["delta_quadratic_secondary"]
        ep = result["endpoint_t12_delta_joint_minus_state_secondary"]
        lines += [
            "## Frozen primary gates",
            "",
            f"- G_STATE: **{gs['pass']}** — mean joint−state {gs['mean']:+.6f}, 95% CI [{gs['bootstrap_95_ci'][0]:+.6f}, {gs['bootstrap_95_ci'][1]:+.6f}], positive {gs['positive_lineages']}/12",
            f"- G_SHUFFLE: **{gh['pass']}** — mean joint−shuffled {gh['mean']:+.6f}, 95% CI [{gh['bootstrap_95_ci'][0]:+.6f}, {gh['bootstrap_95_ci'][1]:+.6f}], positive {gh['positive_lineages']}/12",
            "",
            "## Secondary controls",
            "",
            f"- joint−quadratic-static mean {dq['mean']:+.6f}, 95% CI [{dq['bootstrap_95_ci'][0]:+.6f}, {dq['bootstrap_95_ci'][1]:+.6f}]",
            f"- t=12 joint−state mean {ep['mean']:+.6f}, 95% CI [{ep['bootstrap_95_ci'][0]:+.6f}, {ep['bootstrap_95_ci'][1]:+.6f}]",
            "",
            "## Per-lineage primary internal-window values",
            "",
            "| seed | M | AD2 regime | state | velocity | joint | shuffled | quadratic | Δstate | Δshuffle | Δquadratic |",
            "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in result["rows"]:
            lines.append(
                f"| {r['seed']} | {r['M']} | {r['ad2_regime']} | {r['state']:.4f} | {r['velocity']:.4f} | "
                f"{r['joint']:.4f} | {r['shuffle']:.4f} | {r['quadratic']:.4f} | {r['delta_state']:+.4f} | "
                f"{r['delta_shuffle']:+.4f} | {r['delta_quadratic']:+.4f} |"
            )
        lines += ["", "## Claim boundary", "", result["claim_boundary"], ""]
    Path(path).write_text("\n".join(lines))


def self_check():
    x = np.linspace(0.01, 0.02, 12)
    assert gate(x)["pass"] is True
    y = np.linspace(-0.002, 0.002, 12)
    assert gate(y)["pass"] is False
    print("R8-AD3 classifier self-check ok")


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
    rows = find_summaries(args.root)
    result = classify(rows)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "FINAL_RESULT.json", result)
    write_markdown(result, out / "FINAL_RESULT.md")
    print(result["classification"])


if __name__ == "__main__":
    main()
