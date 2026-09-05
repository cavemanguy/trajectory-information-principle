import argparse
import csv
import json
from pathlib import Path

import numpy as np

PRIMARY_SEEDS = (13, 29, 53)


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True))


def classify(root, outdir):
    root = Path(root)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    files = list(root.rglob("primary_result.json"))
    if len(files) != 3:
        raise RuntimeError(f"expected exactly 3 primary_result.json files, found {len(files)}")

    rows = {}
    for f in files:
        r = json.loads(f.read_text())
        seed = int(r["seed"])
        if seed in rows:
            raise RuntimeError(f"duplicate seed {seed}")
        rows[seed] = r

    if set(rows) != set(PRIMARY_SEEDS):
        raise RuntimeError(f"wrong seed set: {sorted(rows)}")

    ordered = [rows[s] for s in PRIMARY_SEEDS]
    competence = all(bool(r["competence_gate_pass"]) for r in ordered)
    selective = all(bool(r["selective_preservation_seed_criterion_pass"]) for r in ordered)

    if not competence:
        outcome = "A — training reproduction failure"
    elif selective:
        outcome = "C — reproducible training-emergent selective preservation"
    else:
        outcome = "B — competent training without reproducible selective-preservation emergence"

    aggregate = {
        "experiment": "ND-R1",
        "primary_seeds": list(PRIMARY_SEEDS),
        "outcome": outcome,
        "all_seed_competence_gate_pass": competence,
        "all_seed_selective_preservation_criterion_pass": selective,
        "seed_results": ordered,
        "cross_seed_summary": {
            "epoch100_h12_validation_accuracy_mean": float(np.mean([r["epoch100_h12_validation_accuracy"] for r in ordered])),
            "C0_mean": float(np.mean([r["C0_initial_terminal_mean_log_survival"] for r in ordered])),
            "G0_mean": float(np.mean([r["G0_initial_relation_selectivity"] for r in ordered])),
            "G100_mean": float(np.mean([r["G100_final_relation_selectivity"] for r in ordered])),
            "delta_G_mean": float(np.mean([r["delta_G"] for r in ordered])),
            "early_establishment_spearman_mean": float(np.nanmean([r["early_establishment_spearman_t2_vs_t12"] for r in ordered])),
        },
        "frozen_decision_rule": {
            "A": "at least one fresh seed has epoch-100 native h12 validation accuracy < 0.50",
            "C": "all seeds competent and, in every seed, C0<0, delta_G>0, and bootstrap 95% CI lower bound for delta_G>0",
            "B": "all seeds competent but C criterion fails",
        },
        "claim_boundary": "Outcome C supports a training-induced change in how natural task distinctions survive native recurrent evolution in this architecture. It does not establish new information creation, chronology, reader causation, perturbation necessity, attractors/chaos, universality, or practical utility.",
    }

    save_json(out / "aggregate_result.json", aggregate)

    with (out / "primary_seed_table.csv").open("w", newline="") as f:
        fields = [
            "seed",
            "competence_gate_pass",
            "epoch100_h12_validation_accuracy",
            "C0_initial_terminal_mean_log_survival",
            "G0_initial_relation_selectivity",
            "G100_final_relation_selectivity",
            "delta_G",
            "delta_G_ci_low",
            "delta_G_ci_high",
            "selective_preservation_seed_criterion_pass",
            "early_establishment_spearman_t2_vs_t12",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in ordered:
            ci = r["delta_G_bootstrap"]["ci95_percentile"]
            w.writerow(
                {
                    "seed": r["seed"],
                    "competence_gate_pass": r["competence_gate_pass"],
                    "epoch100_h12_validation_accuracy": r["epoch100_h12_validation_accuracy"],
                    "C0_initial_terminal_mean_log_survival": r["C0_initial_terminal_mean_log_survival"],
                    "G0_initial_relation_selectivity": r["G0_initial_relation_selectivity"],
                    "G100_final_relation_selectivity": r["G100_final_relation_selectivity"],
                    "delta_G": r["delta_G"],
                    "delta_G_ci_low": ci[0],
                    "delta_G_ci_high": ci[1],
                    "selective_preservation_seed_criterion_pass": r["selective_preservation_seed_criterion_pass"],
                    "early_establishment_spearman_t2_vs_t12": r["early_establishment_spearman_t2_vs_t12"],
                }
            )

    md = [
        "# ND-R1 Final Result",
        "",
        f"**Primary classification:** {outcome}",
        "",
        "## Fresh primary seeds",
        "",
        "| Seed | h12 val @100 | C0 | G0 | G100 | Delta G | Delta G 95% CI | seed criterion |",
        "|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in ordered:
        ci = r["delta_G_bootstrap"]["ci95_percentile"]
        md.append(
            f"| {r['seed']} | {r['epoch100_h12_validation_accuracy']:.4f} | "
            f"{r['C0_initial_terminal_mean_log_survival']:.4f} | "
            f"{r['G0_initial_relation_selectivity']:.4f} | "
            f"{r['G100_final_relation_selectivity']:.4f} | "
            f"{r['delta_G']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | "
            f"{'PASS' if r['selective_preservation_seed_criterion_pass'] else 'FAIL'} |"
        )

    md += [
        "",
        "## Scope",
        "",
        "This classification uses only the frozen ND-R1 primary criteria. R2-style transient geometry and training-time co-organization measurements are secondary and cannot rescue a failed primary result.",
        "",
        "ND-R1 uses no injected perturbations and no active controller in the primary study.",
        "",
    ]
    (out / "FINAL_RESULT.md").write_text("\n".join(md))
    print(json.dumps(aggregate, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    classify(args.root, args.outdir)


if __name__ == "__main__":
    main()
