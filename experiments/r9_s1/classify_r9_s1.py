"""Frozen full-family aggregate, never a best-seed selector."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

SEEDS=(3109,3137,3163,3191,3221,3251,3271,3301)
ARMS=('DISCONNECTED','RING','RANDOM','FULL','LEARNED','RING_HET','FULL_HET','MONOLITHIC')
VERSION='R9-S1-v1'
PREREG='ca4da167c3be7f7b4ff98e11bf969ff41041eb94'
TASK_BLOB='b9ba7615f5cf70354b6c9b7da6cb13fabea567d6'
BOOT_N=20000;BOOT_SEED=91093


def signed(values,accuracy=False):
    x=np.asarray(values,np.float64)
    if x.shape!=(8,) or not np.isfinite(x).all():raise ValueError('invalid paired seed vector')
    rng=np.random.default_rng(BOOT_SEED)
    means=x[rng.integers(0,8,size=(BOOT_N,8))].mean(1)
    lo,hi=np.quantile(means,[.025,.975]);mean=float(x.mean());pos=int((x>0).sum());neg=int((x<0).sum());limit=.01 if accuracy else 0.
    cls='A+' if mean>=limit and pos>=6 and lo>0 else 'A-' if mean<=-limit and neg>=6 and hi<0 else 'A0'
    return {'class':cls,'mean':mean,'sd':float(x.std(ddof=1)),'ci95':[float(lo),float(hi)],'positive':pos,'negative':neg,'values':x.tolist(),'unit':'accuracy fraction' if accuracy else 'defined metric'}


def qualify(c,valid):
    c=dict(c)
    c['qualified_class']='LOW_COMPETENCE_'+c['class'] if c['class'] in ('A+','A-') and not valid else c['class']
    return c


def stat(values):
    x=np.asarray(values,np.float64)
    if not np.isfinite(x).all():raise ValueError('nonfinite metric')
    return {'mean':float(x.mean()),'sd':float(x.std(ddof=1)),'min':float(x.min()),'max':float(x.max()),'values':x.tolist()}


def load(root,expected_commit):
    files=sorted(Path(root).glob('**/family_result.json'))
    rows=[json.loads(p.read_text()) for p in files]
    if len(rows)!=8 or sorted(r['seed'] for r in rows)!=list(SEEDS):raise RuntimeError('missing, duplicate, or unexpected seed')
    by={r['seed']:r for r in rows};rows=[by[s] for s in SEEDS]
    first=rows[0]['source']
    if not expected_commit:raise RuntimeError('explicit frozen implementation commit required')
    for r in rows:
        if r.get('experiment')!='R9-S1' or r.get('protocol_version')!=VERSION or r.get('frozen_seed') is not True or r.get('all_valid') is not True:raise RuntimeError('protocol or family validity mismatch')
        if set(r['architecture'])!=set(ARMS) or set(r['native'])!=set(ARMS) or set(r['causal'])!=set(ARMS):raise RuntimeError('arm schema mismatch')
        s=r['source']
        if s!=first or s['implementation_commit']!=expected_commit or s['task_git_blob']!=TASK_BLOB or s['preregistration_commit']!=PREREG:raise RuntimeError('source/protocol mismatch')
        if r['configuration']!=rows[0]['configuration']:raise RuntimeError('configuration mismatch')
        if not r['causal']['RING']['hysteresis']['paired_semantics_verified']:raise RuntimeError('paired-history semantics invalid')
        for name in ARMS:
            n=r['native'][name];c=r['causal'][name]
            if 'JOINT' not in n['history'] or 'order_controls' not in n or 'organization' not in n:raise RuntimeError('missing native measurement')
            if not all(k in c for k in ('communication','node_ablations','influence','hysteresis')):raise RuntimeError('missing causal measurement')
        def finite(obj):
            if isinstance(obj,dict):return all(finite(v) for v in obj.values())
            if isinstance(obj,list):return all(finite(v) for v in obj)
            if isinstance(obj,(int,float)) and not isinstance(obj,bool):return math.isfinite(obj)
            return True
        if not finite(r):raise RuntimeError('nonfinite result')
    return rows


def values(rows,fn):return [float(fn(r)) for r in rows]


def aggregate(rows):
    result={'experiment':'R9-S1','protocol_version':VERSION,'seeds':list(SEEDS),'source':rows[0]['source'],'configuration':rows[0]['configuration'],'screening_only':True}
    validity={};arch={}
    for name in ARMS:
        tests=rows[0]['architecture'][name]['test']
        metrics={k:stat(values(rows,lambda r:r['architecture'][name]['test'][k])) for k in tests if not k.endswith('_N')}
        data=metrics['DATA_ACC']['mean'];ret=metrics['RETURN_EARLY_ACC']['mean']
        validity[name]='V+' if data>=.20 and ret>=.15 else 'V-'
        arch[name]={'params':stat(values(rows,lambda r:r['architecture'][name]['params'])),'test':metrics,'validity':validity[name]}
    result['validity']=validity;result['architecture']=arch
    result['history']={};result['complementarity']={};result['communication']={};result['native_organization']={};result['node_ablations']={};result['influence']={};result['hysteresis']={}
    for name in ARMS:
        valid=validity[name]=='V+'
        hist={}
        for stream in rows[0]['native'][name]['history']:
            hist[stream]={}
            for target in ('key','answer'):
                metrics={}
                for field in ('H1','HS','H2','H_DUP'):
                    metrics[field]=qualify(signed(values(rows,lambda r:r['native'][name]['history'][stream][target][field]),True),valid)
                good=metrics['H1']['class']=='A+' and metrics['HS']['class']=='A+'
                metrics['candidate']='A+' if good and valid else 'LOW_COMPETENCE_A+' if good else 'A0'
                hist[stream][target]=metrics
        result['history'][name]=hist
        comps=[r['native'][name]['complementarity'] for r in rows]
        if comps[0] is None:result['complementarity'][name]=None
        else:
            cs={}
            for field in ('joint_minus_best','joint_minus_shuffle','joint_minus_pair'):
                cs[field]=qualify(signed([c[field] for c in comps],True),valid)
            cs['joint_minus_duplicated']=qualify(signed([c['joint']['acc']-c['duplicated_best']['acc'] for c in comps],True),valid)
            good=all(cs[k]['class']=='A+' for k in ('joint_minus_best','joint_minus_shuffle','joint_minus_duplicated'))
            cs['candidate']='A+' if good and valid else 'LOW_COMPETENCE_A+' if good else 'A0'
            cs['best_node_by_seed']=[c['best_node'] for c in comps];cs['best_pair_by_seed']=[c['best_pair'] for c in comps]
            cs['node_accessibility']=[stat([c['nodes'][i]['acc'] for c in comps]) for i in range(8)]
            result['complementarity'][name]=cs
        comm=[r['causal'][name]['communication'] for r in rows]
        if comm[0]['controls'] is None:result['communication'][name]=None
        else:
            cm={}
            for mode in ('zero','shuffle','rewire'):
                cm[mode]={field:qualify(signed([c['contrasts'][mode][field] for c in comm],field!='CROSS_ENTROPY'),valid) for field in ('DATA_ACC','RETURN_EARLY_ACC','CROSS_ENTROPY')}
            good=all(cm[k]['RETURN_EARLY_ACC']['class']=='A+' for k in ('zero','shuffle')) and all(cm[k]['DATA_ACC']['class']=='A+' for k in ('zero','shuffle'))
            cm['candidate']='A+' if good and valid else 'LOW_COMPETENCE_A+' if good else 'A0'
            cm['rewire_informative']=comm[0]['rewire_informative']
            result['communication'][name]=cm
        org=[r['native'][name]['organization'] for r in rows]
        result['native_organization'][name]={'reactivation':{s:{'rank':[stat([o['reactivation'][s]['rank'][j]['acc'] for o in org]) for j in range(4)],'last_minus_first':qualify(signed([o['reactivation'][s]['last_minus_first'] for o in org],True),valid)} for s in org[0]['reactivation']}}
        if org[0]['node_statistics'] is not None:
            result['native_organization'][name]['node_statistics']=[{k:stat([o['node_statistics'][i][k] for o in org]) for k in org[0]['node_statistics'][i]} for i in range(8)]
            result['native_organization'][name]['cka_mean']=np.mean([o['cka'] for o in org],axis=0).tolist()
            result['native_organization'][name]['cka_sd']=np.std([o['cka'] for o in org],axis=0,ddof=1).tolist()
            result['native_organization'][name]['lag_correlations']={k:{'mean':np.mean([o['lag_correlations'][k] for o in org],axis=0).tolist(),'sd':np.std([o['lag_correlations'][k] for o in org],axis=0,ddof=1).tolist()} for k in ('native','shuffled','reversed')}
        ab=[r['causal'][name]['node_ablations'] for r in rows]
        if ab[0] is None:result['node_ablations'][name]=None
        else:
            result['node_ablations'][name]=[{k:{field:qualify(signed([a['nodes'][i][k][field] for a in ab],field!='CROSS_ENTROPY'),valid) for field in a['nodes'][i][k]} if isinstance(a['nodes'][i][k],dict) else qualify(signed([a['nodes'][i][k] for a in ab],True),valid) for k in ('outgoing_effect','readout_effect','outgoing_key_effect','readout_key_effect')} for i in range(8) for a in [ab[0]]]
        inf=[r['causal'][name]['influence'] for r in rows]
        if inf[0] is None:result['influence'][name]=None
        else:
            result['influence'][name]={}
            for kind in ('different_key','same_key'):
                result['influence'][name][kind]={'counts':[z[kind]['n'] for z in inf],'nodes':[]}
                for i in range(8):
                    entries=[z[kind]['nodes'][i] for z in inf if z[kind]['nodes'] is not None and z[kind]['nodes'][i]['n']>0]
                    result['influence'][name][kind]['nodes'].append({'estimable_seeds':len(entries),'response_mean':np.mean([e['response'] for e in entries],axis=0).tolist() if entries else None,'response_sd':np.std([e['response'] for e in entries],axis=0,ddof=1).tolist() if len(entries)>1 else None,'tv':stat([e['tv'] for e in entries]) if entries else None,'accuracy_damage':stat([e['accuracy_damage'] for e in entries]) if entries else None})
        hy=[r['causal'][name]['hysteresis']['mean'] for r in rows]
        result['hysteresis'][name]={k:stat([h[k] for h in hy]) for k in hy[0]}
    result['local_order_controls']={}
    for name in ARMS:
        valid=validity[name]=='V+'
        result['local_order_controls'][name]={}
        for stream in rows[0]['native'][name]['order_controls']:
            result['local_order_controls'][name][stream]={}
            for target in ('key','answer'):
                result['local_order_controls'][name][stream][target]={k:qualify(signed(values(rows,lambda r:r['native'][name]['order_controls'][stream][target][k]),True),valid) for k in ('frozen_reverse_gap','vs_integrated','vs_first','vs_final','vs_shuf')}
    pairs=[('RING','DISCONNECTED'),('RANDOM','DISCONNECTED'),('FULL','DISCONNECTED'),('LEARNED','FULL'),('RING_HET','RING'),('FULL_HET','FULL')]+[(name,'MONOLITHIC') for name in ARMS if name!='MONOLITHIC']
    result['topology_contrasts']={}
    for a,b in pairs:
        ok=validity[a]=='V+' and validity[b]=='V+'
        def contrast(fn,accuracy=True):return qualify(signed([fn(r,a)-fn(r,b) for r in rows],accuracy),ok)
        result['topology_contrasts'][a+'__minus__'+b]={
            'DATA':contrast(lambda r,n:r['architecture'][n]['test']['DATA_ACC']),
            'RETURN':contrast(lambda r,n:r['architecture'][n]['test']['RETURN_EARLY_ACC']),
            'H1':contrast(lambda r,n:r['native'][n]['history']['JOINT']['key']['H1']),
            'HS':contrast(lambda r,n:r['native'][n]['history']['JOINT']['key']['HS']),
            'complementarity':None if b=='MONOLITHIC' or a=='MONOLITHIC' else contrast(lambda r,n:r['native'][n]['complementarity']['joint_minus_best'])}
    result['claim_boundary']='Descriptive synthetic anomaly screen. Joint accessibility is not synergy; ablation dependence is not unique architectural necessity; no independent phase substance, universal code, or architecture novelty established.'
    return result


def markdown(r):
    lines=['# R9-S1 Final Aggregate — Recurrent Swarm Dynamics','','Frozen eight-family anomaly screen. Accuracy is an interpretation floor, not an architecture ranking.','','## Validity','','| Arm | Params | DATA | Return-early | Gate |','|---|---:|---:|---:|---|']
    for name in ARMS:
        a=r['architecture'][name]
        lines.append(f"| {name} | {a['params']['mean']:.0f} | {a['test']['DATA_ACC']['mean']:.4f} | {a['test']['RETURN_EARLY_ACC']['mean']:.4f} | {a['validity']} |")
    lines+=['','## Primary anomaly screen','','| Arm | Joint history | Complementarity | Communication |','|---|---|---|---|']
    for name in ARMS:
        h=r['history'][name]['JOINT']['key']['candidate'];c=r['complementarity'][name];m=r['communication'][name]
        lines.append(f"| {name} | {h} | {c['candidate'] if c else 'N/A'} | {m['candidate'] if m else 'N/A'} |")
    lines+=['','All signed comparisons, seed vectors, confidence intervals, negative results, topology contrasts, native organization, ablations, and intervention matrices are in FINAL_RESULT.json. Complete individual records are preserved in FAMILY_RESULTS.json and the original family artifacts. Screening labels are not independently confirmed discoveries.','','## Boundaries','',r['claim_boundary'],'','No main-branch merge was performed by this workflow.']
    return '\n'.join(lines)+'\n'


def write(root,outdir,expected_commit):
    rows=load(root,expected_commit);r=aggregate(rows)
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(rows,indent=2,sort_keys=True,allow_nan=False)
    (out/'FAMILY_RESULTS.json').write_text(raw)
    (out/'FINAL_RESULT.json').write_text(json.dumps(r,indent=2,sort_keys=True,allow_nan=False))
    (out/'FINAL_RESULT.md').write_text(markdown(r))
    manifests={'experiment':'R9-S1','implementation_commit':expected_commit,'seeds':list(SEEDS),'source':rows[0]['source'],'family_record_sha256':hashlib.sha256(raw.encode()).hexdigest(),'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}}
    (out/'MANIFEST.json').write_text(json.dumps(manifests,indent=2,sort_keys=True))
    print('R9-S1 frozen cross-seed classification complete',flush=True)


def self_check():
    assert signed([.1]*8,True)['class']=='A+'
    assert signed([-.1]*8,True)['class']=='A-'
    assert signed([.1,-.1]*4,True)['class']=='A0'
    assert signed([.001]*8,True)['class']=='A0'
    assert qualify(signed([.1]*8),False)['qualified_class']=='LOW_COMPETENCE_A+'
    try:signed([.1]*7);raise AssertionError('missing seed accepted')
    except ValueError:pass
    try:signed([float('nan')]*8);raise AssertionError('nonfinite accepted')
    except ValueError:pass
    return True


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root');ap.add_argument('--outdir');ap.add_argument('--expected-commit');ap.add_argument('--self-check',action='store_true');args=ap.parse_args()
    if args.self_check:assert self_check();print('Classifier self-check PASSED')
    else:
        if not all((args.root,args.outdir,args.expected_commit)):ap.error('root, outdir, expected-commit required')
        write(args.root,args.outdir,args.expected_commit)

if __name__=='__main__':main()
