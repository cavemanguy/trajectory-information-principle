import argparse
import json
from pathlib import Path

SEEDS = (11, 37, 71)


def load_seed(root, seed):
    candidates = list(Path(root).glob(f"**/seed_{seed}/seed_summary.json")) + list(Path(root).glob(f"**/*{seed}*/seed_summary.json"))
    seen = []
    for p in candidates:
        if p not in seen:
            seen.append(p)
    if not seen:
        raise FileNotFoundError(f"seed_summary.json not found for seed {seed} under {root}")
    return json.loads(seen[0].read_text()), str(seen[0])


def supported(c):
    return bool(c["observed"] > 0.0 and c["ci95_percentile"][0] > 0.0)


def main(root, outdir):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    sources = {}
    for seed in SEEDS:
        d, src = load_seed(root, seed)
        sources[str(seed)] = src
        rows.append(d)

    valid = all(r["baseline_validity"] for r in rows)
    h1_seed = [supported(r["primary_contrasts"]["H1_HT_minus_H0"]) for r in rows]
    h2_seed = [supported(r["primary_contrasts"]["H2_J_minus_FF"]) for r in rows]
    h3_seed = [supported(r["primary_contrasts"]["H3_J_minus_EF"]) for r in rows]
    H1 = all(h1_seed)
    H2 = all(h2_seed)
    H3 = all(h3_seed)
    H4 = H2 and H3

    if not valid:
        outcome = "V — baseline validity failure"
    elif H4:
        outcome = "M3 — encoder–recurrence coadaptation supported"
    elif H1 and not H2 and not H3:
        outcome = "M1 — terminal supervision effect only"
    elif H2 or H3:
        outcome = "M2 — component-plasticity effect without full coadaptation"
    else:
        outcome = "M0 — no primary decomposition supported"

    seed_table = []
    for r in rows:
        f = r["final_conditions"]
        c = r["primary_contrasts"]
        seed_table.append({
            "seed": r["seed"],
            "baseline_validity": r["baseline_validity"],
            "G_J": f["J"]["G"],
            "G_H0": f["H0"]["G"],
            "G_HT": f["HT"]["G"],
            "G_FF": f["FF"]["G"],
            "G_EF": f["EF"]["G"],
            "H1": c["H1_HT_minus_H0"],
            "H2": c["H2_J_minus_FF"],
            "H3": c["H3_J_minus_EF"],
            "winner_J": f["J"]["winner_relation"],
            "winner_H0": f["H0"]["winner_relation"],
            "winner_HT": f["HT"]["winner_relation"],
            "winner_FF": f["FF"]["winner_relation"],
            "winner_EF": f["EF"]["winner_relation"],
        })

    agg = {
        "experiment": "R8-M1",
        "primary_seeds": list(SEEDS),
        "baseline_validity_all_seeds": valid,
        "H1_terminal_supervision_supported": H1 if valid else False,
        "H2_recurrent_plasticity_supported": H2 if valid else False,
        "H3_encoder_plasticity_supported": H3 if valid else False,
        "H4_coadaptation_supported": H4 if valid else False,
        "outcome": outcome,
        "seed_results": seed_table,
        "source_files": sources,
        "claim_boundary": "R8-M1 decomposes training-induced native relation-selective survival in one symmetric synthetic recurrent architecture. It does not establish strong emergence, new information creation, chronology, practical advantage, or generality.",
    }
    (out / "aggregate_result.json").write_text(json.dumps(agg, indent=2, sort_keys=True))

    lines = [
        "# R8-M1 Final Result",
        "",
        f"**Primary classification:** {outcome}",
        "",
        f"- Baseline validity: **{valid}**",
        f"- H1 terminal supervision contributes: **{H1 if valid else False}**",
        f"- H2 recurrent-map plasticity contributes: **{H2 if valid else False}**",
        f"- H3 encoder plasticity contributes: **{H3 if valid else False}**",
        f"- H4 full encoder–recurrence coadaptation: **{H4 if valid else False}**",
        "",
        "| Seed | valid | G J | G H0 | G HT | G F-frozen | G encoder-frozen | H1 diff [CI] | H2 diff [CI] | H3 diff [CI] |",
        "|---:|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for r in seed_table:
        def fmt(x):
            return f"{x['observed']:+.4f} [{x['ci95_percentile'][0]:+.4f}, {x['ci95_percentile'][1]:+.4f}]"
        lines.append(
            f"| {r['seed']} | {r['baseline_validity']} | {r['G_J']:.4f} | {r['G_H0']:.4f} | {r['G_HT']:.4f} | {r['G_FF']:.4f} | {r['G_EF']:.4f} | {fmt(r['H1'])} | {fmt(r['H2'])} | {fmt(r['H3'])} |"
        )
    lines += [
        "",
        "## Claim boundary",
        "",
        "This result concerns ordinary training-induced native survival specialization in the tested synthetic recurrent architecture. No perturbation is part of the primary study. Even M3 would establish coadaptation under this training regime, not a new universal trajectory-information principle or strong theoretical emergence.",
        "",
    ]
    (out / "FINAL_RESULT.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    main(a.root, a.outdir)
