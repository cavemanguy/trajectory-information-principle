import argparse
import json
import math
from pathlib import Path

import numpy as np

FAMILY_SEEDS = (14, 24, 34, 47, 58, 73, 89, 107, 116, 127, 139, 151)
CHECKPOINT_EPOCHS = (0, 1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 80, 100)
PRIMARY_EPOCH = 20
N_BOOT = 5000
WINNER_MATCH_THRESHOLD = 5


def load_json(path):
    return json.loads(Path(path).read_text())


def rankdata(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def spearman(a, b):
    ra = rankdata(a)
    rb = rankdata(b)
    if float(np.std(ra)) == 0.0 or float(np.std(rb)) == 0.0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def bootstrap_mean_ci(values, seed, n=N_BOOT):
    v = np.asarray(values, dtype=float)
    g = np.random.default_rng(seed)
    means = np.empty(n, dtype=float)
    for i in range(n):
        ix = g.integers(0, len(v), size=len(v))
        means[i] = float(v[ix].mean())
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def binomial_tail(n, k, p=1/8):
    return float(sum(math.comb(n, i) * p**i * (1-p)**(n-i) for i in range(k, n+1)))


def row_at(summary, cond, epoch):
    for row in summary["checkpoint_analysis"][cond]:
        if int(row["epoch"]) == int(epoch):
            return row
    raise KeyError(f"missing {cond} epoch {epoch}")


def final_log_survival(summary, cond="B"):
    return np.asarray(row_at(summary, cond, 100)["matched"]["relation_log_median_survival_terminal"], dtype=float)


def candidate_vectors(summary, cond, epoch):
    row = row_at(summary, cond, epoch)
    synergy = np.asarray(row["coadaptation_synergy"], dtype=float)
    align = np.asarray(row["shared_gradient"]["alignment_to_total"], dtype=float)
    return synergy, align


def predictor_stats(summaries, cond, epoch, which):
    matches = []
    rhos = []
    per_seed = []
    for s in summaries:
        synergy, align = candidate_vectors(s, cond, epoch)
        vec = synergy if which == "synergy" else align
        final = final_log_survival(s, cond)
        pred = int(np.argmax(vec))
        winner = int(np.argmax(final))
        rho = spearman(vec, final)
        matches.append(int(pred == winner))
        rhos.append(rho)
        per_seed.append({"seed": s["family_seed"], "predicted_winner": pred, "final_winner": winner, "rho": rho})
    ci = bootstrap_mean_ci(rhos, seed=9051 + epoch + (0 if which == "synergy" else 1000) + (0 if cond == "B" else 2000))
    return {"winner_matches": int(sum(matches)), "winner_match_threshold": WINNER_MATCH_THRESHOLD, "exact_binomial_tail_at_observed": binomial_tail(len(summaries), int(sum(matches))), "mean_rho": float(np.mean(rhos)), "rho_ci95": ci, "supported": bool(sum(matches) >= WINNER_MATCH_THRESHOLD and ci[0] > 0.0), "per_seed": per_seed}


def commitment_stats(summaries, cond="B"):
    by_epoch = {}
    onset = None
    for ep in CHECKPOINT_EPOCHS[:-1]:
        rhos = []
        matches = []
        for s in summaries:
            early = np.asarray(row_at(s, cond, ep)["matched"]["relation_log_median_survival_terminal"], dtype=float)
            final = final_log_survival(s, cond)
            rhos.append(spearman(early, final))
            matches.append(int(np.argmax(early) == np.argmax(final)))
        ci = bootstrap_mean_ci(rhos, seed=13000 + ep + (0 if cond == "B" else 2000))
        positive = int(sum(r > 0 for r in rhos))
        agree = int(sum(matches))
        qualifies = positive >= 9 and ci[0] > 0.0 and agree >= 6
        by_epoch[str(ep)] = {"positive_rhos": positive, "winner_agreement": agree, "mean_rho": float(np.mean(rhos)), "rho_ci95": ci, "qualifies": bool(qualifies)}
        if onset is None and qualifies:
            onset = ep
    return {"onset_epoch": onset, "by_epoch": by_epoch}


def collect(root):
    files = sorted(Path(root).rglob("seed_summary.json"))
    if len(files) != len(FAMILY_SEEDS):
        raise RuntimeError(f"expected {len(FAMILY_SEEDS)} seed summaries, found {len(files)}")
    summaries = sorted([load_json(p) for p in files], key=lambda x: int(x["family_seed"]))
    seeds = tuple(int(s["family_seed"]) for s in summaries)
    if seeds != FAMILY_SEEDS:
        raise RuntimeError(f"expected seeds {FAMILY_SEEDS}, got {seeds}")
    return summaries


def classify(summaries):
    validity = bool(all(bool(s["all_conditions_valid"]) for s in summaries))
    p1 = p2 = None
    if not validity:
        classification = "V — training validity failure"
    else:
        p1 = predictor_stats(summaries, "B", PRIMARY_EPOCH, "synergy")
        p2 = predictor_stats(summaries, "B", PRIMARY_EPOCH, "gradient")
        if p1["supported"] and p2["supported"]:
            classification = "T3 — both precommitment predictors supported"
        elif p1["supported"]:
            classification = "T1 — coadaptation-synergy predictor supported"
        elif p2["supported"]:
            classification = "T2 — shared-gradient alignment predictor supported"
        else:
            classification = "T0 — neither precommitment predictor supported"

    commitment_B = commitment_stats(summaries, "B") if validity else None
    commitment_O = commitment_stats(summaries, "O") if validity else None
    lead_lag = {}
    if validity:
        for cond in ("B", "O"):
            lead_lag[cond] = {"synergy": {}, "gradient": {}}
            for ep in CHECKPOINT_EPOCHS[:-1]:
                lead_lag[cond]["synergy"][str(ep)] = predictor_stats(summaries, cond, ep, "synergy")
                lead_lag[cond]["gradient"][str(ep)] = predictor_stats(summaries, cond, ep, "gradient")
    order_rows = []
    if validity:
        for s in summaries:
            b = final_log_survival(s, "B")
            o = final_log_survival(s, "O")
            order_rows.append({"seed": s["family_seed"], "winner_B": int(np.argmax(b)), "winner_O": int(np.argmax(o)), "winner_match": bool(np.argmax(b) == np.argmax(o)), "rho_B_O": spearman(b, o)})
    return {"experiment": "R8-M3", "classification": classification, "training_validity": validity, "primary_epoch": PRIMARY_EPOCH, "P1_coadaptation_synergy": p1, "P2_shared_gradient_alignment": p2, "commitment_B": commitment_B, "commitment_O": commitment_O, "predictor_lead_lag": lead_lag, "order_path": {"winner_agreement_count": int(sum(r["winner_match"] for r in order_rows)) if order_rows else None, "mean_rho_B_O": float(np.mean([r["rho_B_O"] for r in order_rows])) if order_rows else None, "per_seed": order_rows}}


def render_md(result):
    lines = ["# R8-M3 Final Result", "", f"**Primary classification:** {result['classification']}", "", f"- Training validity: **{result['training_validity']}**", f"- Primary precommitment epoch: **{result['primary_epoch']}**"]
    if result["training_validity"]:
        p1 = result["P1_coadaptation_synergy"]
        p2 = result["P2_shared_gradient_alignment"]
        lines += [f"- P1 coadaptation synergy: **{p1['supported']}** — winner matches {p1['winner_matches']}/12; mean rho {p1['mean_rho']:.3f}; 95% CI [{p1['rho_ci95'][0]:.3f}, {p1['rho_ci95'][1]:.3f}]", f"- P2 shared-gradient alignment: **{p2['supported']}** — winner matches {p2['winner_matches']}/12; mean rho {p2['mean_rho']:.3f}; 95% CI [{p2['rho_ci95'][0]:.3f}, {p2['rho_ci95'][1]:.3f}]", f"- B commitment onset: **{result['commitment_B']['onset_epoch']}**", f"- O commitment onset: **{result['commitment_O']['onset_epoch']}**", f"- B/O final winner agreement: **{result['order_path']['winner_agreement_count']}/12**", f"- Mean B/O final rank rho: **{result['order_path']['mean_rho_B_O']:.3f}**"]
    lines += ["", "## Claim boundary", "", "R8-M3 tests predictive precommitment markers of native dynamical specialization in one symmetric synthetic recurrent architecture. A supported predictor is a candidate mechanism marker, not proof of causation. Failure of both predictors does not prove strong emergence.", ""]
    return "\n".join(lines)


def self_check():
    assert abs(spearman([1, 2, 3], [1, 2, 3]) - 1.0) < 1e-9
    assert abs(spearman([1, 2, 3], [3, 2, 1]) + 1.0) < 1e-9
    assert binomial_tail(12, 5) < 0.05
    print("R8-M3 classifier self-check: OK")


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
        raise SystemExit("--root and --outdir required unless --self-check")
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    result = classify(collect(args.root))
    (out / "aggregate.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    (out / "FINAL_RESULT.md").write_text(render_md(result))
    print(render_md(result))


if __name__ == "__main__":
    main()
