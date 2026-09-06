import argparse
import json
from pathlib import Path

import numpy as np

EXPECTED = (1782, 1801, 1819, 1838, 1856, 1875, 1893, 1912, 1930, 1949, 1967, 1986)
BOOT_SEED = 85051
BOOT_N = 20000
MIN_EFFECT = 0.005
MIN_POSITIVE = 10


def load_rows(root):
    rows = []
    for p in sorted(Path(root).rglob("seed_summary.json")):
        rows.append(json.loads(p.read_text()))
    return rows


def bootstrap_ci(vals):
    vals = np.asarray(vals, dtype=np.float64)
    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.integers(0, len(vals), size=(BOOT_N, len(vals)))
    means = vals[draws].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def stats(vals):
    vals = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "positive_lineages": int((vals > 0).sum()),
        "bootstrap_95_ci": bootstrap_ci(vals),
        "n": int(len(vals)),
    }


def summarize_secondary(rows):
    by_time = {}
    by_rho = {}
    by_angle = {}
    by_regime = {}

    for t in (2, 4, 6, 8, 10):
        a = []
        b = []
        for row in rows:
            diag = row["diagnostic"]
            a.extend([x["k_dir"] for x in diag["phase_a"] if int(x["t"]) == t])
            b.extend([x["k_curv"] for x in diag["phase_b"] if int(x["t"]) == t])
        by_time[str(t)] = {
            "mean_k_dir": float(np.mean(a)),
            "mean_k_curv": float(np.mean(b)),
        }

    for rho in (0.05, 0.10, 0.20):
        vals = []
        for row in rows:
            vals.extend([x["k_dir"] for x in row["diagnostic"]["phase_a"]
                         if abs(float(x["rho"]) - rho) < 1e-12])
        by_rho[f"{rho:.2f}"] = {
            "mean_k_dir": float(np.mean(vals)),
            "median_k_dir": float(np.median(vals)),
        }

    for angle in (15.0, 30.0):
        vals = []
        for row in rows:
            vals.extend([x["k_curv"] for x in row["diagnostic"]["phase_b"]
                         if abs(float(x["angle_deg"]) - angle) < 1e-12])
        by_angle[f"{angle:.0f}"] = {
            "mean_k_curv": float(np.mean(vals)),
            "median_k_curv": float(np.median(vals)),
        }

    groups = {}
    for row in rows:
        regime = row["diagnostic"].get("ad2_regime_secondary") or "UNKNOWN"
        groups.setdefault(regime, {"k_dir": [], "k_curv": []})
        groups[regime]["k_dir"].append(row["diagnostic"]["primary"]["k_dir"])
        groups[regime]["k_curv"].append(row["diagnostic"]["primary"]["k_curv"])
    for regime, g in groups.items():
        by_regime[regime] = {
            "n": len(g["k_dir"]),
            "mean_k_dir": float(np.mean(g["k_dir"])),
            "mean_k_curv": float(np.mean(g["k_curv"])),
        }

    return {
        "by_time": by_time,
        "by_rho": by_rho,
        "by_angle": by_angle,
        "by_ad2_regime": by_regime,
    }


def classify(rows):
    seeds = [int(r.get("seed", -1)) for r in rows]
    exact = sorted(seeds) == list(EXPECTED) and len(seeds) == len(set(seeds)) == len(EXPECTED)
    valid = bool(exact and all(r.get("all_valid") is True for r in rows))

    if not valid:
        return {
            "experiment": "R8-AD5",
            "classification": "KX — invalid/protocol failure",
            "n_valid": int(sum(bool(r.get("all_valid")) for r in rows)),
            "expected_seeds": list(EXPECTED),
            "observed_seeds": seeds,
            "validity": {"exact_seed_set": exact, "all_valid": False},
            "gates": {
                "G_DIRECTION_CAUSAL": False,
                "G_CURVATURE_ORIENTATION": False,
            },
        }

    rows = sorted(rows, key=lambda r: int(r["seed"]))
    k_dir = [r["diagnostic"]["primary"]["k_dir"] for r in rows]
    k_dir_ce = [r["diagnostic"]["primary"]["k_dir_ce"] for r in rows]
    k_curv = [r["diagnostic"]["primary"]["k_curv"] for r in rows]
    k_curv_ce = [r["diagnostic"]["primary"]["k_curv_ce"] for r in rows]

    s_dir = stats(k_dir)
    s_dir_ce = stats(k_dir_ce)
    s_curv = stats(k_curv)
    s_curv_ce = stats(k_curv_ce)

    g_dir = bool(
        s_dir["mean"] >= MIN_EFFECT
        and s_dir["bootstrap_95_ci"][0] > 0
        and s_dir["positive_lineages"] >= MIN_POSITIVE
        and s_dir_ce["mean"] > 0
        and s_dir_ce["positive_lineages"] >= MIN_POSITIVE
    )
    g_curv = bool(
        s_curv["mean"] >= MIN_EFFECT
        and s_curv["bootstrap_95_ci"][0] > 0
        and s_curv["positive_lineages"] >= MIN_POSITIVE
        and s_curv_ce["mean"] > 0
        and s_curv_ce["positive_lineages"] >= MIN_POSITIVE
    )

    if g_dir and g_curv:
        label = "K3 — direction and turn-orientation causal sensitivity supported"
    elif g_dir:
        label = "K2 — direction-specific causal sensitivity supported"
    elif g_curv:
        label = "K1 — turn-orientation causal sensitivity supported"
    else:
        label = "K0 — neither preregistered causal contrast supported"

    out_rows = []
    for r in rows:
        d = r["diagnostic"]
        p = d["primary"]
        out_rows.append({
            "seed": int(r["seed"]),
            "M": int(r["maturity_epoch"]),
            "ad2_regime": d.get("ad2_regime_secondary"),
            "native_accuracy": d["native"]["accuracy"],
            "drop_dir": p["drop_dir"],
            "drop_mag": p["drop_mag"],
            "drop_rand": p["drop_rand"],
            "k_dir": p["k_dir"],
            "k_dir_ce": p["k_dir_ce"],
            "drop_az": p["drop_az"],
            "drop_pol": p["drop_pol"],
            "k_curv": p["k_curv"],
            "k_curv_ce": p["k_curv_ce"],
            "max_geometry_error": d["max_geometry_error"],
        })

    return {
        "experiment": "R8-AD5",
        "classification": label,
        "n_valid": len(rows),
        "expected_seeds": list(EXPECTED),
        "validity": {"exact_seed_set": True, "all_valid": True},
        "frozen_thresholds": {
            "min_effect_accuracy": MIN_EFFECT,
            "min_positive_lineages": MIN_POSITIVE,
            "bootstrap_seed": BOOT_SEED,
            "bootstrap_resamples": BOOT_N,
        },
        "gates": {
            "G_DIRECTION_CAUSAL": g_dir,
            "G_CURVATURE_ORIENTATION": g_curv,
        },
        "summaries": {
            "K_DIR": s_dir,
            "K_DIR_CE": s_dir_ce,
            "K_CURV": s_curv,
            "K_CURV_CE": s_curv_ce,
            "DROP_DIR": stats([r["drop_dir"] for r in out_rows]),
            "DROP_MAG": stats([r["drop_mag"] for r in out_rows]),
            "DROP_RAND": stats([r["drop_rand"] for r in out_rows]),
            "DROP_AZ": stats([r["drop_az"] for r in out_rows]),
            "DROP_POL": stats([r["drop_pol"] for r in out_rows]),
            "MAX_GEOMETRY_ERROR": {
                "max": float(max(r["max_geometry_error"] for r in out_rows)),
                "mean": float(np.mean([r["max_geometry_error"] for r in out_rows])),
            },
        },
        "secondary": summarize_secondary(rows),
        "rows": out_rows,
        "claim_boundary": (
            "R8-AD5 tests explicit transition splices in frozen mature recurrent systems. "
            "A positive contrast supports anisotropic downstream functional sensitivity aligned to native "
            "trajectory geometry under matched Euclidean splice size. It does not establish information "
            "beyond the complete Markov state-plus-map, state-independent motion information, essential "
            "chronology, a universal trajectory code, or novelty relative to all prior literature."
        ),
    }


def render_md(d):
    if d["classification"].startswith("KX"):
        return (
            "# R8-AD5 Final Result — Causal Motion Intervention\n\n"
            f"**Primary classification:** {d['classification']}\n\n"
            "Required validity checks failed. No scientific gate is promoted.\n"
        )

    s = d["summaries"]
    lines = [
        "# R8-AD5 Final Result — Causal Motion Intervention",
        "",
        f"**Primary classification:** {d['classification']}",
        "",
        "## Frozen gates",
        "",
        f"- G_DIRECTION_CAUSAL: **{d['gates']['G_DIRECTION_CAUSAL']}**",
        f"- G_CURVATURE_ORIENTATION: **{d['gates']['G_CURVATURE_ORIENTATION']}**",
        "",
        "## Primary summaries",
        "",
        f"- K_DIR: mean {s['K_DIR']['mean']:+.6f}; 95% CI [{s['K_DIR']['bootstrap_95_ci'][0]:+.6f}, {s['K_DIR']['bootstrap_95_ci'][1]:+.6f}]; positive {s['K_DIR']['positive_lineages']}/12",
        f"- K_DIR_CE: mean {s['K_DIR_CE']['mean']:+.6f}; positive {s['K_DIR_CE']['positive_lineages']}/12",
        f"- direction accuracy drop: mean {s['DROP_DIR']['mean']:+.6f}",
        f"- magnitude-only accuracy drop: mean {s['DROP_MAG']['mean']:+.6f}",
        f"- generic random accuracy drop: mean {s['DROP_RAND']['mean']:+.6f}",
        f"- K_CURV: mean {s['K_CURV']['mean']:+.6f}; 95% CI [{s['K_CURV']['bootstrap_95_ci'][0]:+.6f}, {s['K_CURV']['bootstrap_95_ci'][1]:+.6f}]; positive {s['K_CURV']['positive_lineages']}/12",
        f"- K_CURV_CE: mean {s['K_CURV_CE']['mean']:+.6f}; positive {s['K_CURV_CE']['positive_lineages']}/12",
        f"- azimuth/turn-orientation accuracy drop: mean {s['DROP_AZ']['mean']:+.6f}",
        f"- polar/scalar-turn accuracy drop: mean {s['DROP_POL']['mean']:+.6f}",
        f"- maximum frozen geometry mismatch: {s['MAX_GEOMETRY_ERROR']['max']:.3e}",
        "",
        "## Per-lineage values",
        "",
        "| seed | M | AD2 regime | drop dir | drop mag | K_DIR | drop az | drop pol | K_CURV |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in d["rows"]:
        lines.append(
            f"| {r['seed']} | {r['M']} | {r['ad2_regime']} | "
            f"{r['drop_dir']:+.4f} | {r['drop_mag']:+.4f} | {r['k_dir']:+.4f} | "
            f"{r['drop_az']:+.4f} | {r['drop_pol']:+.4f} | {r['k_curv']:+.4f} |"
        )
    lines += ["", "## Claim boundary", "", d["claim_boundary"], ""]
    return "\n".join(lines)


def self_check():
    def fake(seed, kd, kc):
        return {
            "seed": seed,
            "maturity_epoch": 100,
            "all_valid": True,
            "diagnostic": {
                "native": {"accuracy": 0.5},
                "ad2_regime_secondary": "TEST",
                "max_geometry_error": 1e-7,
                "primary": {
                    "k_dir": kd, "k_dir_ce": 0.02 if kd > 0 else -0.02,
                    "drop_dir": 0.03, "drop_mag": 0.01, "drop_rand": 0.02,
                    "k_curv": kc, "k_curv_ce": 0.02 if kc > 0 else -0.02,
                    "drop_az": 0.03, "drop_pol": 0.01,
                },
                "phase_a": [{"t": t, "rho": rho, "k_dir": kd}
                            for t in (2,4,6,8,10) for rho in (0.05,0.10,0.20)],
                "phase_b": [{"t": t, "angle_deg": a, "k_curv": kc}
                            for t in (2,4,6,8,10) for a in (15.0,30.0)],
            },
        }
    assert classify([fake(s, 0.02, 0.02) for s in EXPECTED])["classification"].startswith("K3")
    assert classify([fake(s, 0.02, -0.01) for s in EXPECTED])["classification"].startswith("K2")
    assert classify([fake(s, -0.01, 0.02) for s in EXPECTED])["classification"].startswith("K1")
    assert classify([fake(s, -0.01, -0.01) for s in EXPECTED])["classification"].startswith("K0")
    print("R8-AD5 classifier self-check ok")


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
    rows = load_rows(args.root)
    result = classify(rows)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "FINAL_RESULT.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    (out / "FINAL_RESULT.md").write_text(render_md(result))
    print(result["classification"])


if __name__ == "__main__":
    main()
