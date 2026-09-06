import argparse
import json
from pathlib import Path

import numpy as np

EXPECTED = (2111, 2131, 2153, 2179, 2203, 2221, 2243, 2267)
BOOT_SEED = 91051
BOOT_N = 20000


def load_rows(root):
    return [json.loads(p.read_text()) for p in sorted(Path(root).rglob("seed_summary.json"))]


def ci(vals):
    a = np.asarray(vals, dtype=np.float64)
    rng = np.random.default_rng(BOOT_SEED)
    ix = rng.integers(0, len(a), size=(BOOT_N, len(a)))
    m = a[ix].mean(1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def stats(vals):
    a = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "min": float(a.min()),
        "max": float(a.max()),
        "positive": int((a > 0).sum()),
        "ci95": ci(a),
        "n": int(len(a)),
    }


def classify(rows):
    seeds = [int(r.get("seed", -1)) for r in rows]
    exact = sorted(seeds) == list(EXPECTED) and len(set(seeds)) == len(EXPECTED)
    valid = bool(exact and all(r.get("all_valid") is True for r in rows))
    if not valid:
        return {
            "experiment": "R9-T1",
            "classification": "R9-X — invalid/protocol failure",
            "validity": {"exact_seed_set": exact, "all_valid": False},
            "observed_seeds": seeds,
        }

    rows = sorted(rows, key=lambda r: int(r["seed"]))
    grr_data = [r["architecture"]["GRR"]["test"]["DATA_ACC"] for r in rows]
    grr_ret = [r["architecture"]["GRR"]["test"]["RETURN_EARLY_ACC"] for r in rows]
    competence_seed = [
        (d >= 0.80 and q >= 0.70) for d, q in zip(grr_data, grr_ret)
    ]
    g_comp = bool(np.mean(grr_data) >= 0.85 and np.mean(grr_ret) >= 0.75 and sum(competence_seed) >= 6)

    h_gain = [r["history_transfer"]["h_gain"] for r in rows]
    h_shuf = [r["history_transfer"]["h_vs_shuffled"] for r in rows]
    sh = stats(h_gain)
    ss = stats(h_shuf)
    g_hist = bool(
        sh["mean"] >= 0.01 and sh["ci95"][0] > 0 and sh["positive"] >= 6
        and ss["mean"] > 0 and ss["ci95"][0] > 0
    )

    drec = [r["anisotropy_transfer"]["d_rec"] for r in rows]
    dfunc = [r["anisotropy_transfer"]["d_func"] for r in rows]
    sr = stats(drec)
    sf = stats(dfunc)
    g_aniso = bool(
        sr["mean"] >= 0.05 and sr["ci95"][0] > 0 and sr["positive"] >= 6
        and sf["mean"] > 0 and sf["positive"] >= 6
    )

    sret = [
        r["architecture"]["SANDWICH"]["test"]["RETURN_EARLY_ACC"]
        - r["architecture"]["TRANSFORMER"]["test"]["RETURN_EARLY_ACC"]
        for r in rows
    ]
    s_sand = stats(sret)
    if s_sand["mean"] >= 0.015 and s_sand["ci95"][0] > 0:
        sandwich_label = "S+ — sandwich return advantage"
    elif s_sand["mean"] <= -0.015 and s_sand["ci95"][1] < 0:
        sandwich_label = "S- — sandwich return disadvantage"
    else:
        sandwich_label = "S0 — no clear sandwich return advantage"

    if not g_comp:
        label = "R9-D — primary GRR fails temporal competence"
    elif g_hist and g_aniso:
        label = "R9-A — temporal competence plus both R8 signature transfers"
    elif g_hist or g_aniso:
        label = "R9-B — temporal competence plus one R8 signature transfer"
    else:
        label = "R9-C — temporal competence without R8 signature transfer"

    arch_names = ("GRR", "GRU", "TRANSFORMER", "SANDWICH")
    architecture = {}
    for name in arch_names:
        architecture[name] = {
            "params": [int(r["architecture"][name]["params"]) for r in rows],
            "DATA_ACC": stats([r["architecture"][name]["test"]["DATA_ACC"] for r in rows]),
            "NEW_EARLY_ACC": stats([r["architecture"][name]["test"]["NEW_EARLY_ACC"] for r in rows]),
            "RETURN_EARLY_ACC": stats([r["architecture"][name]["test"]["RETURN_EARLY_ACC"] for r in rows]),
            "STEADY_ACC": stats([r["architecture"][name]["test"]["STEADY_ACC"] for r in rows]),
            "LONG_GAP_ACC": stats([r["architecture"][name]["test"]["LONG_GAP_ACC"] for r in rows]),
        }

    grr_vs_gru = {
        key: stats([
            r["architecture"]["GRR"]["test"][key] - r["architecture"]["GRU"]["test"][key]
            for r in rows
        ])
        for key in ("DATA_ACC", "NEW_EARLY_ACC", "RETURN_EARLY_ACC", "STEADY_ACC", "LONG_GAP_ACC")
    }

    return {
        "experiment": "R9-T1",
        "classification": label,
        "sandwich_classification": sandwich_label,
        "validity": {"exact_seed_set": True, "all_valid": True},
        "gates": {
            "G_TEMPORAL_COMPETENCE": g_comp,
            "G_HISTORY_TRANSFER": g_hist,
            "G_ANISOTROPY_TRANSFER": g_aniso,
        },
        "summaries": {
            "GRR_DATA": stats(grr_data),
            "GRR_RETURN": stats(grr_ret),
            "H_GAIN": sh,
            "H_VS_SHUFFLED": ss,
            "D_REC": sr,
            "D_FUNC": sf,
            "S_RETURN": s_sand,
        },
        "architecture": architecture,
        "grr_vs_gru": grr_vs_gru,
        "rows": [
            {
                "seed": int(r["seed"]),
                "grr_data": r["architecture"]["GRR"]["test"]["DATA_ACC"],
                "grr_return": r["architecture"]["GRR"]["test"]["RETURN_EARLY_ACC"],
                "gru_return": r["architecture"]["GRU"]["test"]["RETURN_EARLY_ACC"],
                "transformer_return": r["architecture"]["TRANSFORMER"]["test"]["RETURN_EARLY_ACC"],
                "sandwich_return": r["architecture"]["SANDWICH"]["test"]["RETURN_EARLY_ACC"],
                "h_gain": r["history_transfer"]["h_gain"],
                "h_vs_shuffled": r["history_transfer"]["h_vs_shuffled"],
                "d_rec": r["anisotropy_transfer"]["d_rec"],
                "d_func": r["anisotropy_transfer"]["d_func"],
            }
            for r in rows
        ],
        "claim_boundary": (
            "R9-T1 is a synthetic continuously driven regime-memory benchmark. Positive transfer gates "
            "would extend specific R8 signatures to this richer recurrent substrate, not establish an "
            "independent trajectory information substance, a universal phase code, Transformer superiority, "
            "or novelty of recurrent-attention hybrids."
        ),
    }


def render(d):
    if d["classification"].startswith("R9-X"):
        return f"# R9-T1 Final Result\n\n**Primary classification:** {d['classification']}\n"
    s = d["summaries"]
    lines = [
        "# R9-T1 Final Result — Continuously Driven Temporal Substrate",
        "",
        f"**Primary classification:** {d['classification']}",
        f"**Sandwich secondary:** {d['sandwich_classification']}",
        "",
        "## Frozen gates",
        "",
        f"- G_TEMPORAL_COMPETENCE: **{d['gates']['G_TEMPORAL_COMPETENCE']}**",
        f"- G_HISTORY_TRANSFER: **{d['gates']['G_HISTORY_TRANSFER']}**",
        f"- G_ANISOTROPY_TRANSFER: **{d['gates']['G_ANISOTROPY_TRANSFER']}**",
        "",
        "## Primary summaries",
        "",
        f"- GRR DATA accuracy: {s['GRR_DATA']['mean']:.4f}",
        f"- GRR return-early accuracy: {s['GRR_RETURN']['mean']:.4f}",
        f"- H_GAIN: {s['H_GAIN']['mean']:+.4f}; CI [{s['H_GAIN']['ci95'][0]:+.4f}, {s['H_GAIN']['ci95'][1]:+.4f}]; positive {s['H_GAIN']['positive']}/8",
        f"- H_vs_shuffled: {s['H_VS_SHUFFLED']['mean']:+.4f}; CI [{s['H_VS_SHUFFLED']['ci95'][0]:+.4f}, {s['H_VS_SHUFFLED']['ci95'][1]:+.4f}]",
        f"- D_REC: {s['D_REC']['mean']:+.4f}; CI [{s['D_REC']['ci95'][0]:+.4f}, {s['D_REC']['ci95'][1]:+.4f}]; positive {s['D_REC']['positive']}/8",
        f"- D_FUNC: {s['D_FUNC']['mean']:+.4f}; positive {s['D_FUNC']['positive']}/8",
        f"- Sandwich minus Transformer return: {s['S_RETURN']['mean']:+.4f}; CI [{s['S_RETURN']['ci95'][0]:+.4f}, {s['S_RETURN']['ci95'][1]:+.4f}]",
        "",
        "## Architecture means",
        "",
        "| architecture | params | DATA | new-early | return-early | steady | long-gap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("GRR", "GRU", "TRANSFORMER", "SANDWICH"):
        a = d["architecture"][name]
        lines.append(
            f"| {name} | {a['params'][0]} | {a['DATA_ACC']['mean']:.4f} | "
            f"{a['NEW_EARLY_ACC']['mean']:.4f} | {a['RETURN_EARLY_ACC']['mean']:.4f} | "
            f"{a['STEADY_ACC']['mean']:.4f} | {a['LONG_GAP_ACC']['mean']:.4f} |"
        )
    lines += ["", "## Claim boundary", "", d["claim_boundary"], ""]
    return "\n".join(lines)


def self_check():
    def fake(seed, good=True, hist=0.03, rec=0.3, func=0.02, sand=0.03):
        base = 0.90 if good else 0.20
        ret = 0.85 if good else 0.15
        def am(r):
            return {
                "params": 1000,
                "test": {
                    "DATA_ACC": base,
                    "NEW_EARLY_ACC": base,
                    "RETURN_EARLY_ACC": r,
                    "STEADY_ACC": base,
                    "LONG_GAP_ACC": base,
                },
            }
        return {
            "seed": seed, "all_valid": True,
            "architecture": {
                "GRR": am(ret), "GRU": am(ret - 0.01),
                "TRANSFORMER": am(ret - sand), "SANDWICH": am(ret),
            },
            "history_transfer": {"h_gain": hist, "h_vs_shuffled": hist},
            "anisotropy_transfer": {"d_rec": rec, "d_func": func},
        }
    x = classify([fake(s) for s in EXPECTED])
    assert x["classification"].startswith("R9-A")
    assert x["sandwich_classification"].startswith("S+")
    x = classify([fake(s, good=False) for s in EXPECTED])
    assert x["classification"].startswith("R9-D")
    print("R9-T1 classifier self-check ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root")
    ap.add_argument("--outdir")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        self_check(); return
    rows = load_rows(args.root)
    d = classify(rows)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    (out / "FINAL_RESULT.json").write_text(json.dumps(d, indent=2, sort_keys=True))
    (out / "FINAL_RESULT.md").write_text(render(d))
    print(d["classification"])
    print(d.get("sandwich_classification", ""))


if __name__ == "__main__":
    main()
