import argparse
import json
from pathlib import Path

import numpy as np

SEEDS = (1109, 1127, 1144, 1162, 1181, 1199, 1218, 1237, 1255, 1274, 1292, 1311)
CONDITIONS = ("FULL", "F1", "F2", "HEAD", "NOADAPT")
DRIFTS = ("A", "B")
N_BOOT = 5000
BOOT_SEED = 20260905

CAL_GAIN_MIN = 0.05
NEUTRAL_MEAN_MIN = -0.02
NEUTRAL_CI_MIN = -0.05
DFULL_MIN = 0.03
DF2_MIN = 0.02
EFF_MAX_RATIO = 0.50


def load_summaries(root):
    by_seed = {}
    for p in Path(root).rglob("seed_summary.json"):
        d = json.loads(p.read_text())
        if d.get("experiment") == "E1-Persistent-Calibration":
            by_seed[int(d["seed"])] = d
    missing = [s for s in SEEDS if s not in by_seed]
    if missing:
        raise RuntimeError(f"missing E1 seed summaries: {missing}")
    return [by_seed[s] for s in SEEDS]


def bootstrap_ci(x, seed):
    x = np.asarray(x, dtype=np.float64)
    g = np.random.default_rng(seed)
    draws = np.empty(N_BOOT, dtype=np.float64)
    for i in range(N_BOOT):
        ix = g.integers(0, len(x), len(x))
        draws[i] = x[ix].mean()
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def avg2(a, b):
    return 0.5 * (float(a) + float(b))


def hold_at(r, epoch=120):
    for h in r["hold"]:
        if int(h["hold_epoch"]) == int(epoch):
            return h
    raise KeyError(epoch)


def switch_at(r, epoch):
    for x in r["switch"]:
        if int(x["switch_epoch"]) == int(epoch):
            return x
    raise KeyError(epoch)


def family_row(d):
    row = {"seed": int(d["seed"]), "base_neutral_acc": float(d["base"]["neutral_test_acc"]), "conditions": {}}
    for c in CONDITIONS:
        ra = d["results"]["A"][c]
        rb = d["results"]["B"][c]
        ha = hold_at(ra, 120)
        hb = hold_at(rb, 120)
        cal_gain = avg2(ra["cal_gain"], rb["cal_gain"])
        cal_acc = avg2(ra["post_calibration_acc"], rb["post_calibration_acc"])
        base_drift = avg2(ra["base_drift_acc"], rb["base_drift_acc"])
        retained_gain = avg2(ha["retained_gain"], hb["retained_gain"])
        return_acc = avg2(ha["return_acc"], hb["return_acc"])
        neutral_acc = avg2(ha["neutral_acc"], hb["neutral_acc"])
        ret_frac_vals = []
        for rr, hh in ((ra, ha), (rb, hb)):
            if float(rr["cal_gain"]) > 0:
                ret_frac_vals.append(float(hh["retained_gain"]) / float(rr["cal_gain"]))
        ret_frac = float(np.mean(ret_frac_vals)) if ret_frac_vals else None
        switch_curve = {}
        for ep in (1, 3, 5, 10):
            switch_curve[str(ep)] = avg2(switch_at(ra, ep)["opposite_acc"], switch_at(rb, ep)["opposite_acc"])
        row["conditions"][c] = {
            "trainable_params": int(ra["trainable_params"]),
            "base_drift_acc": base_drift,
            "post_calibration_acc": cal_acc,
            "cal_gain": cal_gain,
            "neutral_acc_120": neutral_acc,
            "return_acc_120": return_acc,
            "retained_gain_120": retained_gain,
            "retention_fraction_120": ret_frac,
            "switch_curve": switch_curve,
            "online_optimizer_steps": int(ra["online_optimizer_steps"]),
            "trainable_parameter_element_updates": int(ra["trainable_parameter_element_updates"]),
        }
    row["D_neutral"] = row["conditions"]["F1"]["neutral_acc_120"] - row["conditions"]["FULL"]["neutral_acc_120"]
    row["D_full"] = row["conditions"]["F1"]["retained_gain_120"] - row["conditions"]["FULL"]["retained_gain_120"]
    row["D_F2"] = row["conditions"]["F1"]["retained_gain_120"] - row["conditions"]["F2"]["retained_gain_120"]
    return row


def stat(x, seed):
    arr = np.asarray(x, dtype=np.float64)
    return {"mean": float(arr.mean()), "median": float(np.median(arr)), "min": float(arr.min()), "max": float(arr.max()), "ci95": bootstrap_ci(arr, seed), "positive": int((arr > 0).sum()), "n": int(arr.size), "per_seed": [float(v) for v in arr]}


def classify_rows(rows):
    cal = [r["conditions"]["F1"]["cal_gain"] for r in rows]
    dneutral = [r["D_neutral"] for r in rows]
    dfull = [r["D_full"] for r in rows]
    df2 = [r["D_F2"] for r in rows]
    retained = {c: [r["conditions"][c]["retained_gain_120"] for r in rows] for c in CONDITIONS}
    neutral = {c: [r["conditions"][c]["neutral_acc_120"] for r in rows] for c in CONDITIONS}
    cal_gain = {c: [r["conditions"][c]["cal_gain"] for r in rows] for c in CONDITIONS}
    post_cal_acc = {c: [r["conditions"][c]["post_calibration_acc"] for r in rows] for c in CONDITIONS}
    ret_frac = {c: [r["conditions"][c]["retention_fraction_120"] for r in rows if r["conditions"][c]["retention_fraction_120"] is not None] for c in CONDITIONS}
    switch_acc = {c: {ep: [r["conditions"][c]["switch_curve"][str(ep)] for r in rows] for ep in (1, 3, 5, 10)} for c in CONDITIONS}

    stats = {
        "F1_CAL_GAIN": stat(cal, BOOT_SEED),
        "D_neutral_F1_minus_FULL": stat(dneutral, BOOT_SEED + 1),
        "D_full_retained_gain": stat(dfull, BOOT_SEED + 2),
        "D_F2_retained_gain": stat(df2, BOOT_SEED + 3),
        "retained_gain": {}, "neutral_acc_120": {}, "cal_gain": {}, "post_calibration_acc": {}, "retention_fraction_120": {}, "switch_acc": {},
    }
    for i, c in enumerate(CONDITIONS):
        stats["retained_gain"][c] = stat(retained[c], BOOT_SEED + 10 + i)
        stats["neutral_acc_120"][c] = stat(neutral[c], BOOT_SEED + 20 + i)
        stats["cal_gain"][c] = stat(cal_gain[c], BOOT_SEED + 30 + i)
        stats["post_calibration_acc"][c] = stat(post_cal_acc[c], BOOT_SEED + 40 + i)
        stats["retention_fraction_120"][c] = stat(ret_frac[c], BOOT_SEED + 50 + i) if ret_frac[c] else None
        stats["switch_acc"][c] = {str(ep): stat(switch_acc[c][ep], BOOT_SEED + 100 + 10*i + ep) for ep in (1, 3, 5, 10)}

    full_params = rows[0]["conditions"]["FULL"]["trainable_params"]
    f1_params = rows[0]["conditions"]["F1"]["trainable_params"]
    f2_params = rows[0]["conditions"]["F2"]["trainable_params"]
    counts_consistent = all(r["conditions"]["FULL"]["trainable_params"] == full_params and r["conditions"]["F1"]["trainable_params"] == f1_params and r["conditions"]["F2"]["trainable_params"] == f2_params for r in rows)
    efficiency_ratio = float(f1_params / full_params)

    gate_a = stats["F1_CAL_GAIN"]["mean"] >= CAL_GAIN_MIN and stats["F1_CAL_GAIN"]["ci95"][0] > 0
    gate_n = stats["D_neutral_F1_minus_FULL"]["mean"] >= NEUTRAL_MEAN_MIN and stats["D_neutral_F1_minus_FULL"]["ci95"][0] > NEUTRAL_CI_MIN
    gate_rfull = stats["D_full_retained_gain"]["mean"] >= DFULL_MIN and stats["D_full_retained_gain"]["ci95"][0] > 0
    gate_rf2 = stats["D_F2_retained_gain"]["mean"] >= DF2_MIN and stats["D_F2_retained_gain"]["ci95"][0] > 0
    efficiency = bool(counts_consistent and efficiency_ratio <= EFF_MAX_RATIO)

    if not gate_a:
        classification = "P0 — F1 calibration adapter not viable"
    elif not gate_n:
        classification = "P1 — calibration works but neutral-operation tradeoff is too large"
    elif not gate_rfull:
        classification = "P2 — compact persistent adapter, no specific retention edge"
    elif not gate_rf2 or not efficiency:
        classification = "P3 — compact retention advantage over full adaptation"
    else:
        classification = "P4 — F1-specific persistent-adapter advantage"

    online_costs = {c: {"online_optimizer_steps": int(rows[0]["conditions"][c]["online_optimizer_steps"]), "trainable_parameter_element_updates": int(rows[0]["conditions"][c]["trainable_parameter_element_updates"])} for c in CONDITIONS}
    return {
        "classification": classification,
        "gate_A_calibration": bool(gate_a),
        "gate_N_neutral_noninferiority": bool(gate_n),
        "gate_Rfull": bool(gate_rfull),
        "gate_Rmatch_F2": bool(gate_rf2),
        "efficiency_gate": bool(efficiency),
        "trainable_params": {"FULL": int(full_params), "F1": int(f1_params), "F2": int(f2_params), "F1_over_FULL": efficiency_ratio},
        "online_costs": online_costs,
        "stats": stats,
        "rows": rows,
    }


def classify(summaries):
    invalid = []
    for d in summaries:
        v = d.get("validity", {})
        if not (bool(v.get("base_neutral_accuracy")) and bool(v.get("fork_identity")) and bool(v.get("finite")) and bool(v.get("complete"))):
            invalid.append({"seed": int(d["seed"]), "validity": v})
    if invalid:
        return {"classification": "V0 — E1 benchmark validity failure", "all_valid": False, "invalid_families": invalid}
    out = classify_rows([family_row(d) for d in summaries])
    out["all_valid"] = True
    return out


def fmt_stat(s):
    return f"mean {s['mean']:+.4f}; median {s['median']:+.4f}; 95% CI [{s['ci95'][0]:+.4f},{s['ci95'][1]:+.4f}]"


def render_md(result):
    lines = ["# E1 Persistent Calibration Final Result", "", f"**Primary classification:** {result['classification']}", ""]
    if not result.get("all_valid", False):
        lines.append("Benchmark validity failed; no engineering P-classification is interpreted.")
        lines.append("")
        for x in result.get("invalid_families", []):
            lines.append(f"- seed {x['seed']}: {x['validity']}")
        return "\n".join(lines) + "\n"

    lines.extend([
        f"- Gate A calibration: **{result['gate_A_calibration']}**",
        f"- Gate N neutral noninferiority: **{result['gate_N_neutral_noninferiority']}**",
        f"- Gate Rfull: **{result['gate_Rfull']}**",
        f"- Gate Rmatch(F2): **{result['gate_Rmatch_F2']}**",
        f"- Efficiency gate: **{result['efficiency_gate']}**",
        "", "## Primary statistics", "",
        f"- F1 calibration gain: {fmt_stat(result['stats']['F1_CAL_GAIN'])}",
        f"- Neutral F1-FULL at +120: {fmt_stat(result['stats']['D_neutral_F1_minus_FULL'])}",
        f"- Retained-gain F1-FULL: {fmt_stat(result['stats']['D_full_retained_gain'])}",
        f"- Retained-gain F1-F2: {fmt_stat(result['stats']['D_F2_retained_gain'])}",
        "", "## Trainable parameter counts", "",
        f"- FULL: {result['trainable_params']['FULL']}", f"- F1: {result['trainable_params']['F1']}", f"- F2: {result['trainable_params']['F2']}", f"- F1/FULL: {result['trainable_params']['F1_over_FULL']:.4f}",
        "", "## Online update cost", "",
    ])
    for c in CONDITIONS:
        lines.append(f"- {c}: optimizer steps {result['online_costs'][c]['online_optimizer_steps']}; trainable-parameter element updates {result['online_costs'][c]['trainable_parameter_element_updates']}")
    lines.extend(["", "## Condition summaries", ""])
    for c in CONDITIONS:
        rf = result["stats"]["retention_fraction_120"][c]
        rf_txt = f"{rf['mean']:.4f}" if rf is not None else "n/a"
        lines.append(f"- {c}: post-cal acc mean {result['stats']['post_calibration_acc'][c]['mean']:.4f}; cal gain {fmt_stat(result['stats']['cal_gain'][c])}; retained gain {fmt_stat(result['stats']['retained_gain'][c])}; retention fraction mean {rf_txt}; neutral acc +120 mean {result['stats']['neutral_acc_120'][c]['mean']:.4f}")
    lines.extend(["", "## Switch challenge mean accuracy", ""])
    for c in CONDITIONS:
        curve = result["stats"]["switch_acc"][c]
        lines.append(f"- {c}: +1 {curve['1']['mean']:.4f}; +3 {curve['3']['mean']:.4f}; +5 {curve['5']['mean']:.4f}; +10 {curve['10']['mean']:.4f}")
    lines.extend(["", "## Explicit checkpoint oracle", "", "Restoring the exact post-calibration snapshot yields the post-calibration accuracy reported above; this is an explicit-storage ceiling, not a no-mode baseline.", "", "## Per-family primary contrasts", ""])
    for r in result["rows"]:
        lines.append(f"- seed {r['seed']}: F1 cal={r['conditions']['F1']['cal_gain']:+.4f}, D_neutral={r['D_neutral']:+.4f}, D_full={r['D_full']:+.4f}, D_F2={r['D_F2']:+.4f}")
    lines.extend(["", "## Interpretation boundary", "", "E1 is an engineering benchmark. A positive result supports only a retention/efficiency tradeoff on this hidden synthetic sensor-calibration task. It does not establish real-sensor generalization, runtime hidden-state memory, or a stronger trajectory-information claim."])
    return "\n".join(lines) + "\n"


def self_test():
    rows = []
    for seed in SEEDS:
        row = {
            "seed": seed,
            "conditions": {
                "FULL": {"trainable_params": 1412, "cal_gain": 0.12, "post_calibration_acc": 0.90, "retained_gain_120": 0.01, "retention_fraction_120": 0.08, "neutral_acc_120": 0.92, "switch_curve": {"1":0.70,"3":0.80,"5":0.86,"10":0.90}, "online_optimizer_steps": 100, "trainable_parameter_element_updates": 141200},
                "F1": {"trainable_params": 544, "cal_gain": 0.10, "post_calibration_acc": 0.88, "retained_gain_120": 0.09, "retention_fraction_120": 0.90, "neutral_acc_120": 0.915, "switch_curve": {"1":0.68,"3":0.79,"5":0.85,"10":0.89}, "online_optimizer_steps": 100, "trainable_parameter_element_updates": 54400},
                "F2": {"trainable_params": 528, "cal_gain": 0.08, "post_calibration_acc": 0.86, "retained_gain_120": 0.04, "retention_fraction_120": 0.50, "neutral_acc_120": 0.91, "switch_curve": {"1":0.67,"3":0.78,"5":0.84,"10":0.88}, "online_optimizer_steps": 100, "trainable_parameter_element_updates": 52800},
                "HEAD": {"trainable_params": 68, "cal_gain": 0.07, "post_calibration_acc": 0.85, "retained_gain_120": 0.02, "retention_fraction_120": 0.29, "neutral_acc_120": 0.92, "switch_curve": {"1":0.65,"3":0.76,"5":0.82,"10":0.87}, "online_optimizer_steps": 100, "trainable_parameter_element_updates": 6800},
                "NOADAPT": {"trainable_params": 0, "cal_gain": 0.0, "post_calibration_acc": 0.78, "retained_gain_120": 0.0, "retention_fraction_120": None, "neutral_acc_120": 0.90, "switch_curve": {"1":0.78,"3":0.78,"5":0.78,"10":0.78}, "online_optimizer_steps": 0, "trainable_parameter_element_updates": 0},
            },
        }
        row["D_neutral"] = -0.005
        row["D_full"] = 0.08
        row["D_F2"] = 0.05
        rows.append(row)
    r = classify_rows(rows)
    assert r["classification"].startswith("P4"), r
    assert r["efficiency_gate"]
    print("self-test ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root")
    ap.add_argument("--outdir")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.root or not args.outdir:
        raise SystemExit("--root and --outdir required")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    result = classify(load_summaries(args.root))
    (outdir / "classification.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
    (outdir / "FINAL_RESULT.md").write_text(render_md(result))
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "stats")}, indent=2))


if __name__ == "__main__":
    main()
