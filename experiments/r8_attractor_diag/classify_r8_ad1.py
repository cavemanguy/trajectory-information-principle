import argparse
import json
from pathlib import Path

import numpy as np

EXPECTED_SEEDS = (1561, 1579, 1597, 1616, 1634, 1652, 1671, 1689, 1708, 1726, 1745, 1763)


def load_rows(root):
    rows = []
    for p in sorted(Path(root).glob("**/seed_summary.json")):
        d = json.loads(p.read_text())
        if d.get("experiment") == "R8-AD1":
            rows.append(d)
    return rows


def stat(vals):
    a = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(len(a)),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def classify(rows):
    valid = [r for r in rows if r.get("all_valid")]
    seeds = [int(r["seed"]) for r in valid]
    complete = (len(valid) == 12 and set(seeds) == set(EXPECTED_SEEDS))
    counts = {"FIXED": 0, "SHORT_CYCLE": 0, "UNRESOLVED": 0}
    for r in valid:
        counts[r["diagnostic"]["family_class"]] += 1

    if not complete:
        label = "A4 — invalid/incomplete"
    elif counts["FIXED"] >= 10 and counts["SHORT_CYCLE"] == 0:
        label = "A1 — stable fixed-point attractor behavior supported"
    elif counts["FIXED"] <= 3:
        label = "A3 — fixed-point attractor behavior not supported"
    else:
        label = "A2 — mixed long-run regimes"

    family_rows = []
    for r in sorted(valid, key=lambda x: int(x["seed"])):
        d = r["diagnostic"]
        family_rows.append({
            "seed": int(r["seed"]),
            "M": int(r["maturity_epoch"]),
            "class": d["family_class"],
            "period": d["family_period"],
            "fixed_fraction": float(d["fixed_fraction"]),
            "d12_end_median": float(d["d12_end"]["median"]),
            "end_residual_median": float(d["extra_fixed_residual"]["median"]),
            "final_spread_median": float(d["final_state_spread_from_centroid"]["median"]),
            "residual_at_steps": d["residual_at_steps"],
        })

    result = {
        "experiment": "R8-AD1",
        "classification": label,
        "all_valid": bool(complete),
        "counts": counts,
        "n_valid": len(valid),
        "expected_seeds": list(EXPECTED_SEEDS),
        "fixed_fraction": stat([x["fixed_fraction"] for x in family_rows]) if family_rows else None,
        "d12_end_median_across_families": stat([x["d12_end_median"] for x in family_rows]) if family_rows else None,
        "end_residual_median_across_families": stat([x["end_residual_median"] for x in family_rows]) if family_rows else None,
        "final_spread_median_across_families": stat([x["final_spread_median"] for x in family_rows]) if family_rows else None,
        "rows": family_rows,
        "claim_boundary": (
            "This diagnostic addresses fixed-point/short-cycle behavior for native encoded states of the synthetic learned R8 recurrent map under a 512-step horizon and 1e-5 tolerance. "
            "It does not establish a global attractor over all latent states, formal basin topology, hysteresis, or generalization beyond this system."
        ),
    }
    return result


def render_md(r):
    lines = [
        "# R8-AD1 Final Result — Learned Long-Run Attractor Diagnostic",
        "",
        f"**Primary classification:** {r['classification']}",
        "",
        "## Family classifications",
        "",
        f"- FIXED: **{r['counts']['FIXED']}/12**",
        f"- SHORT_CYCLE: **{r['counts']['SHORT_CYCLE']}/12**",
        f"- UNRESOLVED: **{r['counts']['UNRESOLVED']}/12**",
        "",
    ]
    if r["all_valid"]:
        fs = r["fixed_fraction"]
        d12 = r["d12_end_median_across_families"]
        er = r["end_residual_median_across_families"]
        sp = r["final_spread_median_across_families"]
        lines += [
            "## Cross-family diagnostics",
            "",
            f"- fixed fraction: mean {fs['mean']:.6f}; median {fs['median']:.6f}; range [{fs['min']:.6f}, {fs['max']:.6f}]",
            f"- family median `||h12-h512||`: mean {d12['mean']:.6g}; median {d12['median']:.6g}; range [{d12['min']:.6g}, {d12['max']:.6g}]",
            f"- family median `||F(h512)-h512||`: mean {er['mean']:.6g}; median {er['median']:.6g}; range [{er['min']:.6g}, {er['max']:.6g}]",
            f"- family median final-state spread from centroid: mean {sp['mean']:.6g}; median {sp['median']:.6g}; range [{sp['min']:.6g}, {sp['max']:.6g}]",
            "",
        ]
    lines += [
        "## Per-family values",
        "",
        "| seed | M | class | fixed fraction | median d12→512 | median end residual | median final spread |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for x in r["rows"]:
        lines.append(
            f"| {x['seed']} | {x['M']} | {x['class']} | {x['fixed_fraction']:.4f} | "
            f"{x['d12_end_median']:.6g} | {x['end_residual_median']:.6g} | {x['final_spread_median']:.6g} |"
        )
    lines += ["", "## Claim boundary", "", r["claim_boundary"], ""]
    return "\n".join(lines)


def self_check():
    fake = []
    for i, s in enumerate(EXPECTED_SEEDS):
        fake.append({
            "experiment": "R8-AD1", "seed": s, "all_valid": True, "maturity_epoch": 100,
            "diagnostic": {
                "family_class": "FIXED", "family_period": None, "fixed_fraction": 1.0,
                "d12_end": {"median": 1.0}, "extra_fixed_residual": {"median": 0.0},
                "final_state_spread_from_centroid": {"median": 0.0}, "residual_at_steps": {},
            },
        })
    assert classify(fake)["classification"].startswith("A1")
    for x in fake[:9]:
        x["diagnostic"]["family_class"] = "UNRESOLVED"
    assert classify(fake)["classification"].startswith("A3")
    print("self-check ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root")
    ap.add_argument("--outdir")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        self_check(); return
    rows = load_rows(args.root)
    result = classify(rows)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    (out / "FINAL_RESULT.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    (out / "FINAL_RESULT.md").write_text(render_md(result))
    print(result["classification"])


if __name__ == "__main__":
    main()
