import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

EXPECTED_SEEDS = (1782, 1801, 1819, 1838, 1856, 1875, 1893, 1912, 1930, 1949, 1967, 1986)
N_BOOT = 10000
BOOT_SEED = 8044
MIN_GAIN = 0.005
MIN_POSITIVE = 10
MIN_RET_MEAN = 0.75
MIN_RET_CI = 0.50


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def bootstrap_ci(x, seed_offset=0):
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(BOOT_SEED + int(seed_offset))
    vals = np.empty(N_BOOT, dtype=np.float64)
    n = len(x)
    for i in range(N_BOOT):
        vals[i] = x[rng.integers(0, n, size=n)].mean()
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def summarize(x, seed_offset=0):
    x = np.asarray(x, dtype=np.float64)
    ci = bootstrap_ci(x, seed_offset)
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "min": float(x.min()),
        "max": float(x.max()),
        "bootstrap_95_ci": ci,
        "positive_lineages": int((x > 0).sum()),
    }


def standard_gain_gate(s):
    return bool(s["mean"] >= MIN_GAIN and s["bootstrap_95_ci"][0] > 0 and s["positive_lineages"] >= MIN_POSITIVE)


def retention_gate(gain_summary, ret_summary):
    return bool(
        standard_gain_gate(gain_summary)
        and ret_summary["mean"] >= MIN_RET_MEAN
        and ret_summary["bootstrap_95_ci"][0] > MIN_RET_CI
    )


def load_rows(root):
    rows = []
    for p in sorted(Path(root).rglob("seed_summary.json")):
        d = json.loads(p.read_text())
        if d.get("experiment") == "R8-AD4":
            rows.append(d)
    return rows


def classify(rows):
    if len(rows) != len(EXPECTED_SEEDS):
        return {"classification": "D0 — invalid/incomplete", "reason": f"expected 12 rows, found {len(rows)}"}
    by_seed = {int(r["seed"]): r for r in rows}
    if set(by_seed) != set(EXPECTED_SEEDS):
        return {"classification": "D0 — invalid/incomplete", "reason": "seed set mismatch"}
    ordered = [by_seed[s] for s in EXPECTED_SEEDS]
    if not all(r.get("all_valid") for r in ordered):
        return {"classification": "D0 — invalid/incomplete", "reason": "one or more invalid lineages"}

    ds = np.asarray([r["diagnostic"]["delta_full"] for r in ordered], dtype=np.float64)
    dsh = np.asarray([r["diagnostic"]["delta_shuffle"] for r in ordered], dtype=np.float64)
    dd = np.asarray([r["diagnostic"]["delta_direction"] for r in ordered], dtype=np.float64)
    dm = np.asarray([r["diagnostic"]["delta_magnitude"] for r in ordered], dtype=np.float64)
    dc = np.asarray([r["diagnostic"]["delta_curvature"] for r in ordered], dtype=np.float64)
    dt = np.asarray([r["diagnostic"]["delta_turn"] for r in ordered], dtype=np.float64)

    ret_d = np.asarray([r["diagnostic"]["ret_direction"] for r in ordered if r["diagnostic"]["ret_direction"] is not None], dtype=np.float64)
    ret_m = np.asarray([r["diagnostic"]["ret_magnitude"] for r in ordered if r["diagnostic"]["ret_magnitude"] is not None], dtype=np.float64)

    sums = {
        "DELTA_FULL": summarize(ds, 1),
        "DELTA_SHUFFLE": summarize(dsh, 2),
        "DELTA_DIRECTION": summarize(dd, 3),
        "DELTA_MAGNITUDE": summarize(dm, 4),
        "RET_DIRECTION": summarize(ret_d, 5),
        "RET_MAGNITUDE": summarize(ret_m, 6),
        "DELTA_CURVATURE": summarize(dc, 7),
        "DELTA_TURN": summarize(dt, 8),
    }

    g_full = bool(standard_gain_gate(sums["DELTA_FULL"]) and standard_gain_gate(sums["DELTA_SHUFFLE"]))
    g_dir = retention_gate(sums["DELTA_DIRECTION"], sums["RET_DIRECTION"])
    g_mag = retention_gate(sums["DELTA_MAGNITUDE"], sums["RET_MAGNITUDE"])
    g_curv = standard_gain_gate(sums["DELTA_CURVATURE"])

    if not g_full:
        classification = "D5 — AD3 one-step accessibility did not replicate on fresh AD4 probes"
    elif g_dir and not g_mag:
        classification = "D1 — direction-preserving decomposition"
    elif g_mag and not g_dir:
        classification = "D2 — magnitude-preserving decomposition"
    elif g_dir and g_mag:
        classification = "D3 — direction and magnitude both preserve substantial accessibility"
    else:
        classification = "D4 — full displacement effect not preserved by direction or magnitude alone"

    coord_counts = Counter()
    per_lineage = []
    abs_gains, sign_gains, single_gains, top4_gains = [], [], [], []
    for r in ordered:
        q = r["diagnostic"]
        state = q["primary"]["STATE"]["mean_accuracy"]
        abs_gains.append(q["secondary"]["ABS_PATTERN"]["mean_accuracy"] - state)
        sign_gains.append(q["secondary"]["SIGN_PATTERN"]["mean_accuracy"] - state)
        single_gains.append(q["secondary"]["BEST_SINGLE_COORD"]["mean_accuracy"] - state)
        top4_gains.append(q["secondary"]["TOP4_COORD"]["mean_accuracy"] - state)
        for t in q["internal_times"]:
            coord_counts[int(q["coordinate_by_time"][str(t)]["best_coordinate"])] += 1
        per_lineage.append({
            "seed": int(r["seed"]),
            "M": int(r["maturity_epoch"]),
            "state": float(state),
            "full": float(q["primary"]["FULL"]["mean_accuracy"]),
            "direction": float(q["primary"]["DIRECTION"]["mean_accuracy"]),
            "magnitude": float(q["primary"]["MAGNITUDE"]["mean_accuracy"]),
            "delta_full": float(q["delta_full"]),
            "delta_direction": float(q["delta_direction"]),
            "delta_magnitude": float(q["delta_magnitude"]),
            "ret_direction": q["ret_direction"],
            "ret_magnitude": q["ret_magnitude"],
            "delta_curvature": float(q["delta_curvature"]),
            "delta_turn": float(q["delta_turn"]),
        })

    per_time = {}
    for t in range(1, 12):
        vals = {}
        for name in ("FULL", "DIRECTION", "MAGNITUDE", "ABS_PATTERN", "SIGN_PATTERN"):
            a = np.asarray([r["diagnostic"]["by_time"][str(t)][name]["test_mean_accuracy"] for r in ordered])
            vals[name] = {"mean": float(a.mean()), "median": float(np.median(a))}
        st = np.asarray([r["diagnostic"]["by_time"][str(t)]["STATE"]["test_mean_accuracy"] for r in ordered])
        vals["STATE"] = {"mean": float(st.mean()), "median": float(np.median(st))}
        vals["DELTA_FULL"] = {"mean": float(np.mean([r["diagnostic"]["by_time"][str(t)]["FULL"]["test_mean_accuracy"] - r["diagnostic"]["by_time"][str(t)]["STATE"]["test_mean_accuracy"] for r in ordered]))}
        vals["DELTA_DIRECTION"] = {"mean": float(np.mean([r["diagnostic"]["by_time"][str(t)]["DIRECTION"]["test_mean_accuracy"] - r["diagnostic"]["by_time"][str(t)]["STATE"]["test_mean_accuracy"] for r in ordered]))}
        vals["DELTA_MAGNITUDE"] = {"mean": float(np.mean([r["diagnostic"]["by_time"][str(t)]["MAGNITUDE"]["test_mean_accuracy"] - r["diagnostic"]["by_time"][str(t)]["STATE"]["test_mean_accuracy"] for r in ordered]))}
        per_time[str(t)] = vals

    return {
        "experiment": "R8-AD4",
        "classification": classification,
        "n_valid": 12,
        "expected_seeds": list(EXPECTED_SEEDS),
        "gates": {
            "G_FULL_REPLICATION": g_full,
            "G_DIRECTION_PRESERVES": g_dir,
            "G_MAGNITUDE_PRESERVES": g_mag,
            "G_CURVATURE_ADDS": g_curv,
        },
        "summaries": sums,
        "secondary": {
            "ABS_PATTERN_GAIN": summarize(abs_gains, 20),
            "SIGN_PATTERN_GAIN": summarize(sign_gains, 21),
            "BEST_SINGLE_COORD_GAIN": summarize(single_gains, 22),
            "TOP4_COORD_GAIN": summarize(top4_gains, 23),
            "best_coordinate_selection_counts": {str(k): int(v) for k, v in sorted(coord_counts.items())},
        },
        "per_time_secondary": per_time,
        "rows": per_lineage,
        "thresholds": {
            "min_gain": MIN_GAIN,
            "min_positive_lineages": MIN_POSITIVE,
            "min_retained_fraction_mean": MIN_RET_MEAN,
            "min_retained_fraction_ci_lower": MIN_RET_CI,
            "bootstrap_resamples": N_BOOT,
        },
        "claim_boundary": "R8-AD4 decomposes the previously established one-step linear-accessibility gain into normalized displacement direction and scalar magnitude, with prespecified curvature and basis-dependent coordinate analyses. It does not establish information beyond the complete Markov state-plus-map, causal necessity, essential chronology, or a universal trajectory code.",
    }


def write_md(out, d):
    s = d.get("summaries", {})
    lines = [
        "# R8-AD4 Final Result — Motion-Component Decomposition",
        "",
        f"**Primary classification:** {d['classification']}",
        "",
        "## Frozen gates",
        "",
    ]
    for k, v in d.get("gates", {}).items():
        lines.append(f"- {k}: **{v}**")
    if s:
        lines += ["", "## Primary summaries", ""]
        for k in ("DELTA_FULL", "DELTA_SHUFFLE", "DELTA_DIRECTION", "DELTA_MAGNITUDE", "RET_DIRECTION", "RET_MAGNITUDE", "DELTA_CURVATURE", "DELTA_TURN"):
            q = s[k]
            lines.append(f"- {k}: mean {q['mean']:+.6f}; 95% CI [{q['bootstrap_95_ci'][0]:+.6f}, {q['bootstrap_95_ci'][1]:+.6f}]; positive {q['positive_lineages']}/{q['n']}")
    if d.get("rows"):
        lines += ["", "## Per-lineage values", "", "| seed | M | Δfull | Δdir | Δmag | ret dir | ret mag | Δcurv |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in d["rows"]:
            rd = "NA" if r["ret_direction"] is None else f"{r['ret_direction']:.3f}"
            rm = "NA" if r["ret_magnitude"] is None else f"{r['ret_magnitude']:.3f}"
            lines.append(f"| {r['seed']} | {r['M']} | {r['delta_full']:+.4f} | {r['delta_direction']:+.4f} | {r['delta_magnitude']:+.4f} | {rd} | {rm} | {r['delta_curvature']:+.4f} |")
    lines += ["", "## Claim boundary", "", d.get("claim_boundary", "")]
    (Path(out) / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n")


def self_check():
    fake = []
    for i, seed in enumerate(EXPECTED_SEEDS):
        q = {
            "delta_full": 0.06,
            "delta_shuffle": 0.06,
            "delta_direction": 0.055,
            "delta_magnitude": 0.001,
            "ret_direction": 0.916,
            "ret_magnitude": 0.017,
            "delta_curvature": 0.001,
            "delta_turn": 0.0,
            "internal_times": list(range(1, 12)),
            "primary": {"STATE": {"mean_accuracy": 0.16}, "FULL": {"mean_accuracy": 0.22}, "DIRECTION": {"mean_accuracy": 0.215}, "MAGNITUDE": {"mean_accuracy": 0.161}},
            "secondary": {"ABS_PATTERN": {"mean_accuracy": 0.18}, "SIGN_PATTERN": {"mean_accuracy": 0.19}, "BEST_SINGLE_COORD": {"mean_accuracy": 0.17}, "TOP4_COORD": {"mean_accuracy": 0.19}},
            "by_time": {},
            "coordinate_by_time": {},
        }
        for t in range(1, 12):
            q["by_time"][str(t)] = {name: {"test_mean_accuracy": val} for name, val in {"STATE":0.16,"FULL":0.22,"DIRECTION":0.215,"MAGNITUDE":0.161,"ABS_PATTERN":0.18,"SIGN_PATTERN":0.19}.items()}
            q["coordinate_by_time"][str(t)] = {"best_coordinate": (t + i) % 16}
        fake.append({"experiment":"R8-AD4","seed":seed,"maturity_epoch":100,"all_valid":True,"diagnostic":q})
    d = classify(fake)
    assert d["classification"] == "D1 — direction-preserving decomposition"
    print("R8-AD4 classifier self-check ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root")
    ap.add_argument("--outdir")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        self_check()
        return
    rows = load_rows(args.root)
    d = classify(rows)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "FINAL_RESULT.json", d)
    write_md(out, d)
    print(d["classification"])
    print(d.get("gates", {}))


if __name__ == "__main__":
    main()
