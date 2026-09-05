import argparse
import json
from pathlib import Path
import numpy as np

FRESH_SEEDS = (20, 41, 56, 69, 84, 99, 118, 133, 146, 161, 181, 199)
N_BOOT = 5000


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def bootstrap_mean_ci(x, seed):
    x = np.asarray(x, dtype=np.float64)
    g = np.random.default_rng(seed)
    draws = np.empty(N_BOOT)
    for i in range(N_BOOT):
        ix = g.integers(0, len(x), len(x))
        draws[i] = x[ix].mean()
    return [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]


def spearman(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def load_summaries(root):
    by_seed = {}
    for p in Path(root).rglob("seed_summary.json"):
        d = json.loads(p.read_text())
        if d.get("experiment") == "R8-M5R":
            by_seed[int(d["seed"])] = d
    missing = [s for s in FRESH_SEEDS if s not in by_seed]
    if missing:
        raise RuntimeError(f"missing fresh seed summaries: {missing}")
    return [by_seed[s] for s in FRESH_SEEDS]


def rows_from(summaries):
    rows=[]
    for d in summaries:
        e=d["epoch100"]; r={"seed":int(d["seed"]), "param_ratio":float(d["P16_S32_relative_parameter_difference"])}
        for c in ("B16","S24","S32","P16"):
            r[f"h0_{c}"]=float(e[c]["test"]["h0_overall"])
            r[f"h12_{c}"]=float(e[c]["test"]["h12_overall"])
            r[f"G_{c}"]=float(e[c]["survival"]["G"])
            r[f"D_{c}"]=float(e[c]["winner_gap_test"])
            r[f"winner_{c}"]=int(e[c]["survival"]["winner_relation"])
            acc=np.asarray(e[c]["test"]["h12_per_relation"],dtype=float)
            r[f"accwinner_{c}"]=int(np.argmax(acc))
        rows.append(r)
    return rows


def paired(rows, a, b):
    return np.asarray([r[a]-r[b] for r in rows],dtype=float)


def direction(mean, ci):
    if ci[0] > 0: return "increase"
    if ci[1] < 0: return "decrease"
    return "indeterminate/mixed"


def classify(summaries):
    invalid=[]
    for d in summaries:
        if not bool(d.get("all_conditions_valid")):
            invalid.append({"seed":int(d["seed"]),"validity":d.get("condition_validity")})
    if invalid:
        return {"classification":"V — cross-capacity training validity failure","all_conditions_valid":False,"invalid":invalid}

    rows=rows_from(summaries)
    if max(r["param_ratio"] for r in rows) >= .05:
        raise RuntimeError("P16/S32 parameter-match design violation")

    db=paired(rows,"h12_S32","h12_B16")
    dp=paired(rows,"h12_S32","h12_P16")
    ci_db=bootstrap_mean_ci(db,80521); ci_dp=bootstrap_mean_ci(dp,80522)
    P=bool(db.mean() >= .02 and ci_db[0] > 0)
    S=bool(dp.mean() >= .015 and ci_dp[0] > 0)
    classification = ("R0 — wider-state terminal benefit not supported" if not P else
                      "R2 — state-dimension-specific terminal benefit supported" if S else
                      "R1 — wider-state terminal benefit supported, state-dimension specificity not established")

    contrasts={}
    for name,a,b,seed in [
        ("S32_minus_B16_h12","h12_S32","h12_B16",80521),
        ("S32_minus_P16_h12","h12_S32","h12_P16",80522),
        ("S32_minus_B16_G","G_S32","G_B16",80523),
        ("S32_minus_B16_D","D_S32","D_B16",80524),
        ("S32_minus_B16_h0","h0_S32","h0_B16",80525),
        ("S32_minus_P16_G","G_S32","G_P16",80526),
        ("S32_minus_P16_D","D_S32","D_P16",80527),
        ("S32_minus_P16_h0","h0_S32","h0_P16",80528)]:
        x=paired(rows,a,b); ci=bootstrap_mean_ci(x,seed)
        contrasts[name]={"mean":float(x.mean()),"ci95":ci,"direction":direction(float(x.mean()),ci)}

    dG=paired(rows,"G_S32","G_B16"); dD=paired(rows,"D_S32","D_B16")
    secondary={
        "spearman_DeltaG_vs_Deltah12":spearman(dG,db),
        "spearman_DeltaD_vs_Deltah12":spearman(dD,db),
        "per_seed_S32_minus_B16": [{"seed":r["seed"],"Delta_h12":float(db[i]),"Delta_G":float(dG[i]),"Delta_D":float(dD[i])} for i,r in enumerate(rows)],
        "condition_G_vs_h12_spearman":{},
        "survival_winner_accuracy_winner_agreement":{},
        "width_trends":{}
    }
    for c in ("B16","S24","S32","P16"):
        secondary["condition_G_vs_h12_spearman"][c]=spearman([r[f"G_{c}"] for r in rows],[r[f"h12_{c}"] for r in rows])
        secondary["survival_winner_accuracy_winner_agreement"][c]=sum(r[f"winner_{c}"]==r[f"accwinner_{c}"] for r in rows)
    for metric in ("h12","G","D"):
        per=[]
        for r in rows:
            per.append(spearman([0,1,2],[r[f"{metric}_B16"],r[f"{metric}_S24"],r[f"{metric}_S32"]]))
        secondary["width_trends"][metric]={"per_seed":per,"mean":float(np.mean(per)),"ci95":bootstrap_mean_ci(per,80620+{"h12":1,"G":2,"D":3}[metric])}

    return {"classification":classification,"all_conditions_valid":True,"P_supported":P,"S_supported":S,
            "parameter_match_max_relative_difference":float(max(r["param_ratio"] for r in rows)),
            "contrasts":contrasts,"rows":rows,"secondary":secondary}


def write_md(path,r):
    lines=["# R8-M5R Final Result","",f"**Primary classification:** {r['classification']}",""]
    if not r.get("all_conditions_valid",False):
        lines.append(f"- Invalid conditions: {r.get('invalid')}")
    else:
        lines += ["- Cross-capacity validity: **True**",f"- Wider-state terminal benefit P: **{r['P_supported']}**",f"- State-dimension specificity S: **{r['S_supported']}**",f"- P16/S32 max parameter-count difference: {r['parameter_match_max_relative_difference']:.4f}","","## Frozen contrasts",""]
        for k,v in r["contrasts"].items():
            lines.append(f"- {k}: mean {v['mean']:.6f}; 95% CI {v['ci95']}; {v['direction']}")
        lines += ["","## Secondary adaptive-specialization descriptors","",f"- Spearman DeltaG vs Deltah12: {r['secondary']['spearman_DeltaG_vs_Deltah12']:.3f}",f"- Spearman DeltaD vs Deltah12: {r['secondary']['spearman_DeltaD_vs_Deltah12']:.3f}"]
    lines += ["","## Claim boundary","","R8-M5R is a fresh-seed replication of the exact R8-M5 capacity manipulation with no directional success requirement on specialization. R0/R1/R2 concern terminal performance and state-dimension specificity only. G and D are reported separately and may increase, decrease, or vary across runs. No outcome establishes that the system explicitly chooses specialization, universal necessity, strong emergence, or generalization beyond this synthetic architecture.",""]
    Path(path).write_text("\n".join(lines))


def self_check():
    assert bootstrap_mean_ci([1,1,1],1)[0] > 0
    assert direction(1,[.1,.2]) == "increase"
    assert direction(-1,[-.2,-.1]) == "decrease"
    assert direction(0,[-.1,.1]) == "indeterminate/mixed"
    print("classifier self-check ok")


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root"); ap.add_argument("--outdir"); ap.add_argument("--self-check",action="store_true"); a=ap.parse_args()
    if a.self_check: self_check(); return
    if not a.root or not a.outdir: raise SystemExit("--root and --outdir required")
    r=classify(load_summaries(a.root)); out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True); save_json(out/"FINAL_RESULT.json",r); write_md(out/"FINAL_RESULT.md",r); print(json.dumps(r,indent=2))

if __name__ == "__main__": main()
