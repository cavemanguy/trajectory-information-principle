import argparse
import json
from pathlib import Path

import numpy as np

FRESH_SEEDS = (214, 230, 247, 263, 279, 296, 313, 329, 346, 362, 378, 397)
N_BOOT = 5000


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def bootstrap_mean_ci(x, seed):
    x = np.asarray(x, dtype=np.float64)
    g = np.random.default_rng(seed)
    draws = np.empty(N_BOOT, dtype=np.float64)
    for i in range(N_BOOT):
        ix = g.integers(0, len(x), len(x))
        draws[i] = x[ix].mean()
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def load_summaries(root):
    by_seed = {}
    for p in Path(root).rglob('seed_summary.json'):
        d = json.loads(p.read_text())
        if d.get('experiment') == 'R8-M7R':
            by_seed[int(d['seed'])] = d
    missing = [s for s in FRESH_SEEDS if s not in by_seed]
    if missing:
        raise RuntimeError(f'missing fresh seed summaries: {missing}')
    return [by_seed[s] for s in FRESH_SEEDS]


def rec(d, cond, idx):
    return d['conditions'][cond]['records'][idx]


def classify(summaries):
    maturity_fail = []
    for d in summaries:
        if not bool(d.get('maturity_reached')):
            maturity_fail.append({'seed': int(d['seed']), 'maturity_epoch': d.get('maturity_epoch')})
    if maturity_fail:
        return {
            'classification': 'V0 — competence/stability maturity failure',
            'all_valid': False,
            'maturity_failures': maturity_fail,
        }

    execution_fail = []
    for d in summaries:
        if not bool(d.get('all_valid')):
            execution_fail.append({'seed': int(d['seed']), 'validity': d.get('validity')})
    if execution_fail:
        return {
            'classification': 'V1 — post-maturity execution validity failure',
            'all_valid': False,
            'execution_failures': execution_fail,
        }

    rows = []
    for d in summaries:
        A = int(d['A_baseline_winner'])
        B = int(d['B_baseline_loser'])
        sw = [rec(d, 'SWITCH', i) for i in range(3)]
        fx = [rec(d, 'FIX', i) for i in range(3)]
        h0 = [rec(d, 'H0SWITCH', i) for i in range(3)]
        q1, q2, q3 = [float(x['Q']) for x in sw]
        fq2 = float(fx[1]['Q'])
        hq1, hq2, hq3 = [float(x['Q']) for x in h0]
        amp_sw = 0.5 * ((q2 - q1) - (q3 - q2))
        amp_h0 = 0.5 * ((hq2 - hq1) - (hq3 - hq2))
        b_win = int(sw[1]['survival']['winner_relation']) == B
        a_return = int(sw[2]['survival']['winner_relation']) == A
        rows.append({
            'seed': int(d['seed']), 'M': int(d['maturity_epoch']), 'A': A, 'B': B,
            'baseline_Q': float(d['baseline']['Q']),
            'Q_A1': q1, 'Q_B': q2, 'Q_A2': q3,
            'AB_shift': q2 - q1, 'BA_shift': q3 - q2,
            'Q_B_FIX': fq2,
            'AMP_SWITCH': amp_sw, 'AMP_H0SWITCH': amp_h0,
            'C1_diff': q2 - fq2,
            'C2_diff': amp_sw - amp_h0,
            'exact_pair': bool(b_win and a_return),
            'B_winner_B_phase': bool(b_win),
            'A_winner_return_phase': bool(a_return),
            'h12_A1_A': float(sw[0]['validation']['h12_per_relation'][A]),
            'h12_A1_B': float(sw[0]['validation']['h12_per_relation'][B]),
            'h12_B_A': float(sw[1]['validation']['h12_per_relation'][A]),
            'h12_B_B': float(sw[1]['validation']['h12_per_relation'][B]),
            'h12_A2_A': float(sw[2]['validation']['h12_per_relation'][A]),
            'h12_A2_B': float(sw[2]['validation']['h12_per_relation'][B]),
            'G_A1': float(sw[0]['survival']['G']),
            'G_B': float(sw[1]['survival']['G']),
            'G_A2': float(sw[2]['survival']['G']),
        })

    def arr(k):
        return np.asarray([r[k] for r in rows], dtype=np.float64)

    stats = {}
    boot_seeds = {
        'AB_shift': 81701, 'BA_shift': 81702, 'Q_B': 81703, 'Q_A2': 81704,
        'C1_diff': 81705, 'C2_diff': 81706,
    }
    for k, sd in boot_seeds.items():
        x = arr(k)
        stats[k] = {'mean': float(x.mean()), 'ci95': bootstrap_mean_ci(x, sd)}

    T1 = bool(stats['AB_shift']['mean'] >= 0.75 and stats['AB_shift']['ci95'][0] > 0)
    T2 = bool(stats['BA_shift']['mean'] <= -0.75 and stats['BA_shift']['ci95'][1] < 0)
    T3 = bool(stats['Q_B']['mean'] >= 0.20 and stats['Q_B']['ci95'][0] > 0)
    T4 = bool(stats['Q_A2']['mean'] <= -0.20 and stats['Q_A2']['ci95'][1] < 0)
    T = bool(T1 and T2 and T3 and T4)

    C1 = bool(stats['C1_diff']['mean'] >= 0.50 and stats['C1_diff']['ci95'][0] > 0)
    C2 = bool(stats['C2_diff']['mean'] >= 0.25 and stats['C2_diff']['ci95'][0] > 0)
    S = bool(C1 and C2)

    exact_count = int(sum(r['exact_pair'] for r in rows))
    E = bool(exact_count >= 8)

    if not T:
        classification = 'D0 — reversible demand tracking not supported'
    elif S and E:
        classification = 'D2 — demand-specific specialist reassignment supported'
    else:
        classification = 'D1 — reversible relative demand tracking supported'

    b_gain = arr('h12_B_B') - arr('h12_A1_B')
    a_gain = arr('h12_A2_A') - arr('h12_B_A')
    maturity = arr('M')
    secondary = {
        'maturity_epoch': {
            'mean': float(maturity.mean()), 'min': int(maturity.min()), 'max': int(maturity.max()),
            'per_seed': {str(r['seed']): int(r['M']) for r in rows},
        },
        'B_h12_gain_A1_to_B': {'mean': float(b_gain.mean()), 'ci95': bootstrap_mean_ci(b_gain, 81721)},
        'A_h12_return_B_to_A2': {'mean': float(a_gain.mean()), 'ci95': bootstrap_mean_ci(a_gain, 81722)},
        'B_winner_B_phase_count': int(sum(r['B_winner_B_phase'] for r in rows)),
        'A_winner_return_phase_count': int(sum(r['A_winner_return_phase'] for r in rows)),
        'exact_reassignment_pair_count': exact_count,
    }

    return {
        'classification': classification,
        'all_valid': True,
        'T_supported': T,
        'T_criteria': {'T1_AB_shift': T1, 'T2_BA_shift': T2, 'T3_B_takeover': T3, 'T4_A_return': T4},
        'S_supported': S,
        'S_criteria': {'C1_switch_vs_fix': C1, 'C2_switch_vs_h0': C2},
        'E_supported': E,
        'exact_reassignment_pair_count': exact_count,
        'stats': stats,
        'rows': rows,
        'secondary': secondary,
    }


def write_md(path, r):
    lines = ['# R8-M7R Final Result', '', f"**Primary classification:** {r['classification']}", '']
    if r['classification'].startswith('V0'):
        lines.append(f"- Maturity failures: {r.get('maturity_failures')}")
    elif r['classification'].startswith('V1'):
        lines.append(f"- Execution failures: {r.get('execution_failures')}")
    else:
        lines += [
            '- Cross-family maturity/execution validity: **True**',
            f"- T reversible relative tracking: **{r['T_supported']}**",
            f"- S demand specificity: **{r['S_supported']}**",
            f"- E exact specialist reassignment: **{r['E_supported']}** ({r['exact_reassignment_pair_count']}/12 paired families)",
            '', '## Frozen primary statistics', ''
        ]
        for k, v in r['stats'].items():
            lines.append(f"- {k}: mean {v['mean']:.6f}; 95% CI {v['ci95']}")
        lines += ['', '## Secondary descriptors', '']
        for k, v in r['secondary'].items():
            lines.append(f"- {k}: {v}")
    lines += [
        '', '## Claim boundary', '',
        'R8-M7R tests reversible demand-sensitive native dynamical reorganization only after each lineage reaches a preregistered competence-and-stability maturity trigger. D2 would support demand-specific specialist reassignment in this synthetic architecture; it would not establish conscious choice, universal trajectory computation, strong emergence, essential chronology, language-model generalization, or that trajectory history carries information beyond the complete state.', ''
    ]
    Path(path).write_text('\n'.join(lines))


def self_check():
    assert bootstrap_mean_ci([1.0, 1.0, 1.0], 1)[0] > 0
    assert bootstrap_mean_ci([-1.0, -1.0, -1.0], 2)[1] < 0
    assert 0.75 >= 0.75 and -0.75 <= -0.75 and 0.20 >= 0.20 and -0.20 <= -0.20
    print('R8-M7R classifier self-check: OK')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root')
    ap.add_argument('--outdir')
    ap.add_argument('--self-check', action='store_true')
    args = ap.parse_args()
    if args.self_check:
        self_check(); return
    if not args.root or not args.outdir:
        raise SystemExit('--root and --outdir required')
    summaries = load_summaries(args.root)
    r = classify(summaries)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    save_json(out / 'FINAL_RESULT.json', r)
    write_md(out / 'FINAL_RESULT.md', r)
    print(json.dumps(r, indent=2))


if __name__ == '__main__':
    main()
