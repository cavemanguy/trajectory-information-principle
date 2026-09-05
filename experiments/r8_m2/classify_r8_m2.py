import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

FAMILY_SEEDS = (3, 9, 21, 28, 44, 62, 86, 101)
PI = np.array([3, 4, 5, 6, 7, 0, 1, 2], dtype=np.int64)
BOOT_N = 5000
EARLY_EPOCHS = (0, 1, 2, 5, 10, 20, 40, 60)


def derive_seed(name):
    h = hashlib.sha256(f"R8-M2|classifier|{name}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def ranks(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    out = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        avg = (i + j - 1) / 2.0
        out[order[i:j]] = avg
        i = j
    return out


def spearman(a, b):
    ra = ranks(a)
    rb = ranks(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def map_to_identity(v):
    v = np.asarray(v, dtype=np.float64)
    out = np.empty_like(v)
    for r in range(len(v)):
        out[PI[r]] = v[r]
    return out


def bootstrap_mean(values, name, nboot=BOOT_N):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(derive_seed(name))
    boots = np.empty(nboot, dtype=np.float64)
    for i in range(nboot):
        ix = rng.integers(0, len(values), size=len(values))
        boots[i] = float(np.mean(values[ix]))
    return {
        "mean": float(np.mean(values)),
        "ci95_percentile": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
        "n_bootstrap": nboot,
    }


def binomial_tail(n, k, p):
    return float(sum(math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k, n + 1)))


def load_summaries(root):
    root = Path(root)
    rows = []
    for seed in FAMILY_SEEDS:
        candidates = list(root.glob(f"**/r8-m2-seed-{seed}/seed_summary.json"))
        if not candidates:
            candidates = list(root.glob(f"**/seed_{seed}/seed_summary.json"))
        if not candidates:
            candidates = list(root.glob(f"**/*{seed}*/seed_summary.json"))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one seed_summary for seed {seed}, found {candidates}")
        rows.append(json.loads(candidates[0].read_text()))
    return rows


def row_at_epoch(summary, condition, epoch):
    for row in summary["checkpoint_analysis"][condition]:
        if int(row["epoch"]) == int(epoch):
            return row
    raise KeyError((condition, epoch))


def analyze_seed(s):
    final = s["final_conditions"]
    b = np.asarray(final["B"]["relation_log_median_survival_terminal"], dtype=np.float64)
    p = np.asarray(final["P"]["relation_log_median_survival_terminal"], dtype=np.float64)
    d = np.asarray(final["D"]["relation_log_median_survival_terminal"], dtype=np.float64)
    o = np.asarray(final["O"]["relation_log_median_survival_terminal"], dtype=np.float64)

    wb = int(final["B"]["winner_relation"])
    wp = int(final["P"]["winner_relation"])
    wd = int(final["D"]["winner_relation"])
    wo = int(final["O"]["winner_relation"])

    p_mapped = map_to_identity(p)
    d_mapped = map_to_identity(d)

    init_raw = spearman(b, p)
    init_aligned = spearman(b, p_mapped)
    data_raw = spearman(b, d)
    data_aligned = spearman(b, d_mapped)
    order_rho = spearman(b, o)

    grad = np.asarray(s["initial_shared_gradient_predictor"]["shared_gradient_norm_by_relation"], dtype=np.float64)
    grad_rho = spearman(grad, b)

    early = {}
    for ep in EARLY_EPOCHS:
        r = row_at_epoch(s, "B", ep)
        v = np.asarray(r["relation_log_median_survival_terminal"], dtype=np.float64)
        early[str(ep)] = {
            "rho_to_epoch100": spearman(v, b),
            "winner_relation": int(r["winner_relation"]),
            "winner_matches_epoch100": bool(int(r["winner_relation"]) == wb),
        }

    return {
        "seed": int(s["family_seed"]),
        "valid": bool(s["all_conditions_valid"]),
        "winner_B": wb,
        "winner_P": wp,
        "winner_P_bundle_identity": int(PI[wp]),
        "init_bundle_winner_match": bool(int(PI[wp]) == wb),
        "rho_init_raw": init_raw,
        "rho_init_bundle_aligned": init_aligned,
        "delta_rho_init": init_aligned - init_raw,
        "winner_D": wd,
        "winner_D_source_identity": int(PI[wd]),
        "data_source_winner_match": bool(int(PI[wd]) == wb),
        "rho_data_raw": data_raw,
        "rho_data_source_aligned": data_aligned,
        "delta_rho_data": data_aligned - data_raw,
        "winner_O": wo,
        "order_winner_match": bool(wo == wb),
        "rho_order_B_vs_O": order_rho,
        "initial_shared_gradient_rho_to_final": grad_rho,
        "early_commitment": early,
        "G_B": float(final["B"]["G"]),
        "G_P": float(final["P"]["G"]),
        "G_D": float(final["D"]["G"]),
        "G_O": float(final["O"]["G"]),
        "validation": {c: final[c]["validation"] for c in ("B", "P", "D", "O")},
    }


def classify(rows):
    analyzed = [analyze_seed(r) for r in rows]
    all_valid = all(r["valid"] for r in analyzed)

    init_matches = sum(int(r["init_bundle_winner_match"]) for r in analyzed)
    data_matches = sum(int(r["data_source_winner_match"]) for r in analyzed)
    order_matches = sum(int(r["order_winner_match"]) for r in analyzed)

    init_delta = bootstrap_mean([r["delta_rho_init"] for r in analyzed], "init_delta")
    data_delta = bootstrap_mean([r["delta_rho_data"] for r in analyzed], "data_delta")
    order_rho = bootstrap_mean([r["rho_order_B_vs_O"] for r in analyzed], "order_rho")
    grad_rho = bootstrap_mean([r["initial_shared_gradient_rho_to_final"] for r in analyzed], "gradient_rho")

    H1 = bool(init_matches >= 4 and init_delta["ci95_percentile"][0] > 0.0)
    H2 = bool(data_matches >= 4 and data_delta["ci95_percentile"][0] > 0.0)

    if not all_valid:
        outcome = "V — training validity failure"
    elif H1 and H2:
        outcome = "S3 — initialization and data-column tracking"
    elif H1:
        outcome = "S1 — relation-specific initialization tracking"
    elif H2:
        outcome = "S2 — finite-sample data-column tracking"
    else:
        outcome = "S0 — neither simple source tracks winner"

    early = {}
    onset = None
    for ep in EARLY_EPOCHS:
        vals = [r["early_commitment"][str(ep)]["rho_to_epoch100"] for r in analyzed]
        boot = bootstrap_mean(vals, f"early_{ep}")
        positive_count = sum(int(v > 0) for v in vals)
        winner_matches = sum(int(r["early_commitment"][str(ep)]["winner_matches_epoch100"]) for r in analyzed)
        qualifies = bool(positive_count >= 6 and boot["ci95_percentile"][0] > 0.0 and winner_matches >= 4)
        early[str(ep)] = {
            "positive_rho_count": positive_count,
            "winner_match_count": winner_matches,
            "rho": boot,
            "commitment_criterion_met": qualifies,
        }
        if onset is None and qualifies:
            onset = ep

    return {
        "experiment": "R8-M2",
        "family_seeds": list(FAMILY_SEEDS),
        "pi": PI.tolist(),
        "all_conditions_all_families_valid": all_valid,
        "H1_initialization_bundle_tracking_supported": H1 if all_valid else False,
        "H2_data_column_tracking_supported": H2 if all_valid else False,
        "outcome": outcome,
        "primary": {
            "initialization": {
                "mapped_winner_matches": init_matches,
                "n_families": len(analyzed),
                "exact_binomial_tail_p_at_observed_or_more": binomial_tail(len(analyzed), init_matches, 1.0 / N_REL) if init_matches > 0 else 1.0,
                "mean_delta_rho_aligned_minus_raw": init_delta,
            },
            "data_column": {
                "mapped_winner_matches": data_matches,
                "n_families": len(analyzed),
                "exact_binomial_tail_p_at_observed_or_more": binomial_tail(len(analyzed), data_matches, 1.0 / N_REL) if data_matches > 0 else 1.0,
                "mean_delta_rho_aligned_minus_raw": data_delta,
            },
        },
        "secondary": {
            "minibatch_order": {
                "winner_matches": order_matches,
                "mean_spearman_B_vs_O": order_rho,
            },
            "early_commitment": early,
            "commitment_onset_epoch": onset,
            "initial_shared_gradient_predictor": grad_rho,
        },
        "seed_results": analyzed,
        "claim_boundary": "R8-M2 tests ordinary sources of seed-dependent native survival specialization in one symmetric synthetic recurrent architecture. S0 does not prove strong emergence; S1/S2/S3 do not imply universality or practical advantage.",
    }


def write_markdown(result, outdir):
    p = Path(outdir) / "FINAL_RESULT.md"
    lines = []
    lines.append("# R8-M2 Final Result")
    lines.append("")
    lines.append(f"**Primary classification:** {result['outcome']}")
    lines.append("")
    lines.append(f"- Training validity: **{result['all_conditions_all_families_valid']}**")
    lines.append(f"- H1 initialization-bundle tracking: **{result['H1_initialization_bundle_tracking_supported']}**")
    lines.append(f"- H2 data-column tracking: **{result['H2_data_column_tracking_supported']}**")
    lines.append(f"- Commitment onset epoch: **{result['secondary']['commitment_onset_epoch']}**")
    lines.append("")
    lines.append("| Seed | valid | winner B | P→bundle | init match | D→source | data match | winner O | order match | rho B/O | grad rho |")
    lines.append("|---:|---|---:|---:|---|---:|---|---:|---|---:|---:|")
    for r in result["seed_results"]:
        lines.append(
            f"| {r['seed']} | {r['valid']} | {r['winner_B']} | {r['winner_P_bundle_identity']} | "
            f"{r['init_bundle_winner_match']} | {r['winner_D_source_identity']} | {r['data_source_winner_match']} | "
            f"{r['winner_O']} | {r['order_winner_match']} | {r['rho_order_B_vs_O']:.3f} | "
            f"{r['initial_shared_gradient_rho_to_final']:.3f} |"
        )
    lines.append("")
    init = result["primary"]["initialization"]
    data = result["primary"]["data_column"]
    lines.append(
        f"Initialization tracking: {init['mapped_winner_matches']}/8 mapped winner matches; "
        f"mean aligned-minus-raw rho {init['mean_delta_rho_aligned_minus_raw']['mean']:.3f}, "
        f"95% CI {init['mean_delta_rho_aligned_minus_raw']['ci95_percentile']}."
    )
    lines.append(
        f"Data-column tracking: {data['mapped_winner_matches']}/8 mapped winner matches; "
        f"mean aligned-minus-raw rho {data['mean_delta_rho_aligned_minus_raw']['mean']:.3f}, "
        f"95% CI {data['mean_delta_rho_aligned_minus_raw']['ci95_percentile']}."
    )
    lines.append("")
    lines.append("## Claim boundary")
    lines.append("")
    lines.append(result["claim_boundary"])
    p.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_summaries(args.root)
    result = classify(rows)
    save_json(out / "aggregate_result.json", result)
    write_markdown(result, out)
    print((out / "FINAL_RESULT.md").read_text())


if __name__ == "__main__":
    main()
