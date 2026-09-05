import argparse
import json
from pathlib import Path

import numpy as np

FRESH_SEEDS = (16, 26, 39, 52, 64, 78, 93, 109, 122, 136, 148, 163)
N_BOOT = 5000


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def bootstrap_mean_ci(x, seed=80404):
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
        if d.get("experiment") == "R8-M4":
            by_seed[int(d["seed"])] = d
    missing = [s for s in FRESH_SEEDS if s not in by_seed]
    if missing:
        raise RuntimeError(f"missing fresh seed summaries: {missing}")
    return [by_seed[s] for s in FRESH_SEEDS]


def self_check():
    # Outcome-free logic tests.
    assert bootstrap_mean_ci([-1, -1, -1], 1)[1] < 0
    return True


def classify(summaries):
    baseline_valid = [bool(d["baseline_valid"]) for d in summaries]
    if not all(baseline_valid):
        return {
            "classification": "V — baseline lineage validity failure",
            "baseline_valid_all": False,
            "failed_seeds": [int(d["seed"]) for d in summaries if not d["baseline_valid"]],
        }

    rows = []
    delta_g = []
    rel_red = []
    delta_eb = []
    delta_em = []
    gm_minus_ge = []
    e_lower_count = 0

    for d in summaries:
        e100 = d["epoch100"]
        gb = float(e100["B"]["survival"]["G"])
        ge = float(e100["E"]["survival"]["G"])
        gm = float(e100["M"]["survival"]["G"])
        ab = float(e100["B"]["test"]["h12_overall"])
        ae = float(e100["E"]["test"]["h12_overall"])
        am = float(e100["M"]["test"]["h12_overall"])
        dg = ge - gb
        rr = (gb - ge) / (gb + 1e-8)
        deb = ae - ab
        dem = ae - am
        gdiff = gm - ge
        if ge < gb:
            e_lower_count += 1
        delta_g.append(dg)
        rel_red.append(rr)
        delta_eb.append(deb)
        delta_em.append(dem)
        gm_minus_ge.append(gdiff)
        rows.append({
            "seed": int(d["seed"]),
            "G_B": gb, "G_E": ge, "G_M": gm,
            "relative_G_reduction": rr,
            "h12_B": ab, "h12_E": ae, "h12_M": am,
            "Delta_EB": deb, "Delta_EM": dem,
        })

    ci_dg = bootstrap_mean_ci(delta_g, 80401)
    mean_rel_red = float(np.mean(rel_red))
    suppression = bool(e_lower_count >= 9 and ci_dg[1] < 0 and mean_rel_red >= 0.30)

    out = {
        "baseline_valid_all": True,
        "suppression_gate": suppression,
        "E_lower_G_count": e_lower_count,
        "mean_DeltaG_EB": float(np.mean(delta_g)),
        "DeltaG_EB_ci95": ci_dg,
        "mean_relative_G_reduction": mean_rel_red,
        "rows": rows,
    }

    if not suppression:
        out["classification"] = "F0 — manipulation failure"
        return out

    ci_eb = bootstrap_mean_ci(delta_eb, 80402)
    ci_em = bootstrap_mean_ci(delta_em, 80403)
    ci_gmge = bootstrap_mean_ci(gm_minus_ge, 80405)
    mean_eb = float(np.mean(delta_eb))
    mean_em = float(np.mean(delta_em))
    mean_gmge = float(np.mean(gm_minus_ge))

    f1 = bool(mean_eb > -0.02 and ((ci_eb[0] <= 0 <= ci_eb[1]) or ci_eb[0] > -0.02))
    f3 = bool(
        mean_eb <= -0.02 and ci_eb[1] < 0 and
        mean_em <= -0.015 and ci_em[1] < 0 and
        mean_gmge > 0 and ci_gmge[0] > 0
    )

    out.update({
        "mean_Delta_EB": mean_eb,
        "Delta_EB_ci95": ci_eb,
        "mean_Delta_EM": mean_em,
        "Delta_EM_ci95": ci_em,
        "mean_GM_minus_GE": mean_gmge,
        "GM_minus_GE_ci95": ci_gmge,
        "F1_criteria": f1,
        "F3_criteria": f3,
    })

    if f3:
        out["classification"] = "F3 — selective-specialization contribution supported"
    elif f1:
        out["classification"] = "F1 — specialization suppressed, task performance preserved"
    else:
        out["classification"] = "F2 — ambiguous functional effect"
    return out


def write_md(path, result):
    lines = ["# R8-M4 Final Result", "", f"**Primary classification:** {result['classification']}", ""]
    if result.get("baseline_valid_all") is False:
        lines += [f"- Failed baseline seeds: {result.get('failed_seeds')}"]
    else:
        lines += [
            f"- Baseline validity: **{result.get('baseline_valid_all')}**",
            f"- Suppression gate: **{result.get('suppression_gate')}**",
            f"- E lower G count: {result.get('E_lower_G_count')}/12",
            f"- Mean DeltaG(E-B): {result.get('mean_DeltaG_EB'):.6f}; 95% CI {result.get('DeltaG_EB_ci95')}",
            f"- Mean relative G reduction: {result.get('mean_relative_G_reduction'):.3f}",
        ]
        if result.get("suppression_gate"):
            lines += [
                f"- Mean h12 Delta(E-B): {result.get('mean_Delta_EB'):.6f}; 95% CI {result.get('Delta_EB_ci95')}",
                f"- Mean h12 Delta(E-M): {result.get('mean_Delta_EM'):.6f}; 95% CI {result.get('Delta_EM_ci95')}",
                f"- Mean G(M)-G(E): {result.get('mean_GM_minus_GE'):.6f}; 95% CI {result.get('GM_minus_GE_ci95')}",
            ]
    lines += ["", "## Claim boundary", "", "R8-M4 tests functional necessity under one preregistered training intervention in one symmetric synthetic recurrent architecture. F3 supports a functional contribution under this intervention, not a universal necessity theorem or proof that Euclidean survival magnitude is the causal mediator. F1 shows comparable task performance can be learned despite successful suppression under this intervention. F0/F2 do not support a necessity claim.", ""]
    Path(path).write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root")
    ap.add_argument("--outdir")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        self_check()
        print("classifier self-check ok")
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
