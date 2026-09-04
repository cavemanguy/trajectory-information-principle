import argparse, csv, json
from pathlib import Path

import numpy as np

PRIMARY_SEEDS=(7,19,43)
EPS_FRACS=(0.001,0.003,0.01,0.03,0.1,0.3)
STEPS=12


def save_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True))


def rankcorr(x,y):
    x=np.asarray(x); y=np.asarray(y)
    rx=np.empty_like(np.argsort(x),dtype=float)
    ry=np.empty_like(np.argsort(y),dtype=float)
    rx[np.argsort(x,kind='mergesort')]=np.arange(len(x))
    ry[np.argsort(y,kind='mergesort')]=np.arange(len(y))
    return float(np.corrcoef(rx,ry)[0,1])


def load_seed_cells(root):
    files=list(Path(root).rglob('cells.json'))
    if len(files) != 3:
        raise RuntimeError(f'expected exactly 3 seed cell files, found {len(files)}')
    byseed={}
    expected={(t,e) for e in EPS_FRACS for t in range(STEPS+1)}
    for f in files:
        cells=json.loads(f.read_text())
        seed=int(cells[0]['seed'])
        if seed in byseed: raise RuntimeError(f'duplicate seed {seed}')
        got={(int(c['t']),float(c['epsilon_fraction'])) for c in cells}
        if len(cells) != 78 or got != expected:
            raise RuntimeError(f'incomplete grid for seed {seed}: {len(cells)} cells')
        required=('pair_R_cross_minus_same','pair_cos_same_minus_cross','pair_R_advantage_bootstrap_ci95','numerically_stable_10pct')
        if any(any(k not in c for k in required) for c in cells):
            raise RuntimeError(f'seed {seed} does not contain Pre-outcome Amendment 1 metrics')
        byseed[seed]=cells
    if set(byseed) != set(PRIMARY_SEEDS):
        raise RuntimeError(f'wrong primary seeds: {sorted(byseed)}')
    return byseed


def classify(root,outdir):
    byseed=load_seed_cells(root)
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    rows=[]; consensus=[]
    keys_to_aggregate=(
        'meanop_D_temporal','meanop_R_temporal','meanop_cos_temporal',
        'meanop_R_cross_time_radius','meanop_cos_cross_time_radius',
        'meanop_R_same_time_radius','meanop_cos_same_time_radius',
        'pair_R_cross_minus_same','pair_cos_same_minus_cross',
        'native_response_norm_pair_mean','convergence_rel_half_mean','convergence_rel_double_mean',
    )
    for ei,frac in enumerate(EPS_FRACS):
        for t in range(STEPS+1):
            cs=[next(c for c in byseed[s] if int(c['t'])==t and float(c['epsilon_fraction'])==frac) for s in PRIMARY_SEEDS]
            row={'t':t,'epsilon_fraction':frac}
            for key in keys_to_aggregate:
                vals=[float(c[key]) for c in cs]
                row[key+'_seedmean']=float(np.mean(vals))
                row[key+'_seedmin']=float(np.min(vals))
                row[key+'_seedmax']=float(np.max(vals))
            stable=all(bool(c['numerically_stable_10pct']) for c in cs)
            geom=all(
                float(c['pair_R_cross_minus_same']) >= .05
                and float(c['pair_R_advantage_bootstrap_ci95'][0]) > 0
                and float(c['pair_cos_same_minus_cross']) > 0
                for c in cs
            )
            row['stable_all_seeds']=stable
            row['geometric_cell_all_seeds']=bool(stable and geom)
            rows.append(row)
            if row['geometric_cell_all_seeds']:
                consensus.append((t,ei,row['pair_R_cross_minus_same_seedmean']))

    cells_set={(t,e) for t,e,_ in consensus}
    visited=set(); comps=[]
    for node in cells_set:
        if node in visited: continue
        stack=[node]; comp=[]; visited.add(node)
        while stack:
            u=stack.pop(); comp.append(u)
            for v in ((u[0]-1,u[1]),(u[0]+1,u[1]),(u[0],u[1]-1),(u[0],u[1]+1)):
                if v in cells_set and v not in visited:
                    visited.add(v); stack.append(v)
        comps.append(comp)
    biggest=max(comps,key=len) if comps else []

    qualifies_C=(
        len(consensus)>=4
        and len({t for t,_,_ in consensus})>=2
        and len({e for _,e,_ in consensus})>=2
    )

    peak=None; peak_component_size=0; d_boundary_cells_stable=False; qualifies_D=False
    if qualifies_C:
        peak=max(consensus,key=lambda z:z[2]); pt,pe,pv=peak
        peak_component=next(comp for comp in comps if (pt,pe) in comp)
        peak_component_size=len(peak_component)
        def row_at(t,e):
            return next(r for r in rows if r['t']==t and r['epsilon_fraction']==EPS_FRACS[e])
        boundary_rows=[row_at(pt,0),row_at(pt,len(EPS_FRACS)-1),row_at(0,pe),row_at(STEPS,pe)]
        d_boundary_cells_stable=all(r['stable_all_seeds'] for r in boundary_rows)
        eps_boundary=max(boundary_rows[0]['pair_R_cross_minus_same_seedmean'],boundary_rows[1]['pair_R_cross_minus_same_seedmean'])
        time_boundary=max(boundary_rows[2]['pair_R_cross_minus_same_seedmean'],boundary_rows[3]['pair_R_cross_minus_same_seedmean'])
        qualifies_D=(
            0<pt<STEPS
            and 0<pe<len(EPS_FRACS)-1
            and peak_component_size>=4
            and d_boundary_cells_stable
            and pv>=1.5*eps_boundary
            and pv>=1.5*time_boundary
        )

    sensitivity_rankcorr=None; stable_rows=[r for r in rows if r['stable_all_seeds']]
    if qualifies_D:
        outcome='D — bounded trajectory–perturbation regime'
    elif qualifies_C:
        outcome='C — time-specific geometric structure beyond radius-matched within-time variation'
    elif not stable_rows:
        outcome='A — no valid trajectory-dependent response structure'
    else:
        sensitivity_rankcorr=rankcorr(
            [r['meanop_D_temporal_seedmean'] for r in stable_rows],
            [r['native_response_norm_pair_mean_seedmean'] for r in stable_rows],
        )
        maxadv=max(r['pair_R_cross_minus_same_seedmean'] for r in stable_rows)
        if abs(sensitivity_rankcorr)>=.80 or maxadv < .05:
            outcome='B — sensitivity/contraction or generic state-geometry explanation'
        else:
            outcome='A — no preregistered time-specific geometric trajectory structure'

    result={
        'experiment':'JTP-1',
        'primary_seeds':list(PRIMARY_SEEDS),
        'preoutcome_amendments':[1,2],
        'outcome':outcome,
        'consensus_geometric_cells':len(consensus),
        'largest_connected_consensus_component':len(biggest),
        'peak_connected_component_size':peak_component_size,
        'd_boundary_cells_stable':d_boundary_cells_stable,
        'stable_consensus_cells':len(stable_rows),
        'sensitivity_rankcorr_D_vs_response_norm':sensitivity_rankcorr,
        'peak_consensus_cell':None if peak is None else {
            't':peak[0],
            'epsilon_fraction':EPS_FRACS[peak[1]],
            'pair_R_cross_minus_same_seedmean':peak[2],
        },
        'decision_rule':{
            'numerical_stability':'all seeds: mean relative operator change <=0.10 for both eps/2 and 2eps',
            'geometric_cell':'all seeds: cross-time radius-matched R minus same-time radius-matched R >=0.05; paired bootstrap CI lower bound >0; cosine(same)-cosine(cross)>0; stable',
            'C':'at least 4 geometric cells spanning >=2 times and >=2 epsilon levels',
            'D':'C; peak interior; peak-containing connected component >=4; all four comparison boundary cells stable; peak >=1.5x both epsilon- and time-boundary effects',
            'B':'if C fails: stable-grid |rank corr(absolute temporal D,native response magnitude)|>=0.80 or maximum corrected normalized advantage <0.05',
        },
    }
    save_json(out/'aggregate_result.json',result)
    save_json(out/'aggregate_cells.json',rows)
    with (out/'aggregate_cells.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    md=[
        '# JTP-1 Final Result','',
        f'**Primary classification:** {outcome}','',
        f'Consensus geometric cells: {len(consensus)} / {len(rows)}',
        f'Largest connected consensus component: {len(biggest)}',
        f'Peak connected component size: {peak_component_size}',
        f'D boundary cells stable: {d_boundary_cells_stable}',
    ]
    if peak:
        md.append(f'Peak consensus cell: t={peak[0]}, epsilon fraction={EPS_FRACS[peak[1]]}, seed-mean corrected normalized advantage={peak[2]:.6f}')
    if sensitivity_rankcorr is not None:
        md.append(f'Stable-grid rank correlation of absolute temporal separation with native response magnitude: {sensitivity_rankcorr:.6f}')
    md += [
        '',
        'This is the authoritative mechanical classification under the original JTP-1 preregistration plus PREOUTCOME_AMENDMENT_1 and PREOUTCOME_AMENDMENT_2.',
        '',
        'The amendments were committed before authoritative result inspection and preserve the original claim boundary.',
    ]
    (out/'FINAL_RESULT.md').write_text('\n'.join(md)+'\n')
    print(json.dumps(result,indent=2))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--aggregate',required=True)
    ap.add_argument('--outdir',required=True)
    args=ap.parse_args()
    classify(args.aggregate,args.outdir)

if __name__=='__main__': main()
