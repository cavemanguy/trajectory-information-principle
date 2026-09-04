import argparse, csv, hashlib, json, math, platform, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

N_REL=8; N_VAL=16; EMB=8; HIDDEN=32; LATENT=16; STEPS=12
N_ANALYSIS=1024; BATCH=128; ETA=1e-12
PRIMARY_SEEDS=(7,19,43)
EPS_FRACS=(0.001,0.003,0.01,0.03,0.1,0.3)
ANALYSIS_BANK_SEED=20260904
TIME_SHUFFLE_SEED=20260906
EXPECTED_MEMORY_SHA='aea7617fbf9817cecdd29e28d55f4ecdd88678ca56a3312629c42d8f60a3c2a7'
EXPECTED_PERM_SHA='8cae4c20d0efc961714bcb37ad286903d9f1abef970d1add4de5328756d0801d'
EXPECTED_TIME_SHA='143c2843cd97da4a874964717464c603c595839e8d31d634afbc18a9cec2a468'
EXPECTED_DIR_SHA='d64f9a1707174557fec37dc9535cfd7f806c86b48314c9d092a4e6685eee6769'


def sha_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save_json(p,obj):
    Path(p).write_text(json.dumps(obj,indent=2,sort_keys=True,default=str))

class Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.rel_emb=nn.Embedding(N_REL,EMB)
        self.val_emb=nn.Embedding(N_VAL,EMB)
        self.enc=nn.GRU(EMB*2,HIDDEN,batch_first=True)
        self.to_h=nn.Linear(HIDDEN,LATENT)
        self.F=nn.Sequential(nn.Linear(LATENT,HIDDEN),nn.GELU(),nn.Linear(HIDDEN,LATENT),nn.Tanh())
        self.head0=nn.ModuleList([nn.Linear(LATENT,N_VAL) for _ in range(N_REL)])
        self.headT=nn.ModuleList([nn.Linear(LATENT,N_VAL) for _ in range(N_REL)])
    def encode(self,y,perms):
        b=len(y)
        rel=torch.arange(N_REL,device=y.device).expand(b,-1).gather(1,perms)
        val=y.gather(1,perms)
        x=torch.cat([self.rel_emb(rel),self.val_emb(val)],-1)
        _,s=self.enc(x)
        return torch.tanh(self.to_h(s[-1]))
    def trajectory(self,h0):
        hs=[h0]; h=h0
        for _ in range(STEPS):
            h=self.F(h); hs.append(h)
        return torch.stack(hs,1)


def load_core(core_path,summary_path):
    summary=json.loads(Path(summary_path).read_text())
    actual=sha_file(core_path)
    if summary.get('checkpoint_sha256') != actual:
        raise RuntimeError(f'core hash mismatch: expected {summary.get("checkpoint_sha256")} actual {actual}')
    ck=torch.load(core_path,map_location='cpu',weights_only=False)
    model=Core(); model.load_state_dict(ck['state_dict']); model.eval()
    for p in model.parameters(): p.requires_grad=False
    return model,summary,actual


def make_fixed_banks():
    rng=np.random.default_rng(ANALYSIS_BANK_SEED)
    memories=rng.integers(0,N_VAL,size=(N_ANALYSIS,N_REL),dtype=np.int64)
    perms=np.stack([rng.permutation(N_REL) for _ in range(N_ANALYSIS)]).astype(np.int64)
    rng=np.random.default_rng(TIME_SHUFFLE_SEED)
    tps=[]; base=np.arange(STEPS+1)
    for _ in range(N_ANALYSIS):
        while True:
            p=rng.permutation(STEPS+1)
            if np.all(p != base):
                tps.append(p); break
    time_perm=np.stack(tps).astype(np.int64)
    H=np.array([[1.0]],dtype=np.float64)
    while H.shape[0] < LATENT:
        H=np.block([[H,H],[H,-H]])
    directions=H/math.sqrt(LATENT)
    hashes={
        'memory_sha256':sha_array(memories),
        'encoding_permutation_sha256':sha_array(perms),
        'time_derangement_sha256':sha_array(time_perm),
        'direction_bank_sha256':sha_array(directions),
    }
    expected={
        'memory_sha256':EXPECTED_MEMORY_SHA,
        'encoding_permutation_sha256':EXPECTED_PERM_SHA,
        'time_derangement_sha256':EXPECTED_TIME_SHA,
        'direction_bank_sha256':EXPECTED_DIR_SHA,
    }
    if hashes != expected:
        raise RuntimeError(f'fixed-bank hash mismatch: {hashes}')
    return memories,perms,time_perm,directions,hashes


def generate_trajectories(core,memories,perms):
    ys=torch.from_numpy(memories).long(); ps=torch.from_numpy(perms).long(); chunks=[]
    with torch.no_grad():
        for a in range(0,N_ANALYSIS,BATCH):
            h0=core.encode(ys[a:a+BATCH],ps[a:a+BATCH])
            chunks.append(core.trajectory(h0))
    return torch.cat(chunks,0).contiguous()


def _nearest_norm_match(sn,si,st,q,m):
    p=int(np.searchsorted(sn,q)); best=None; left=p-1; right=p
    while left >= 0 or right < len(sn):
        dl=abs(float(sn[left])-float(q)) if left >= 0 else float('inf')
        dr=abs(float(sn[right])-float(q)) if right < len(sn) else float('inf')
        if best is not None and min(dl,dr) > best[0]:
            break
        if dl <= dr:
            cand=left; left-=1
        else:
            cand=right; right+=1
        if 0 <= cand < len(sn) and si[cand] != m:
            err=abs(float(sn[cand])-float(q))
            key=(err,int(si[cand]),int(st[cand]))
            if best is None or key < best:
                best=key
    if best is None: raise RuntimeError('failed norm match')
    return best[1],best[2]


def build_norm_matches(tr,cross_time):
    norms=torch.linalg.vector_norm(tr,dim=-1).cpu().numpy()
    out_i=np.empty((N_ANALYSIS,STEPS+1),dtype=np.int64)
    out_t=np.empty((N_ANALYSIS,STEPS+1),dtype=np.int64)
    all_i=np.repeat(np.arange(N_ANALYSIS),STEPS+1)
    all_t=np.tile(np.arange(STEPS+1),N_ANALYSIS)
    flat_norm=norms.reshape(-1)
    for t in range(STEPS+1):
        valid=(all_t != t) if cross_time else (all_t == t)
        pool_n=flat_norm[valid]; pool_i=all_i[valid]; pool_t=all_t[valid]
        order=np.argsort(pool_n,kind='mergesort')
        sn=pool_n[order]; si=pool_i[order]; st=pool_t[order]
        for m in range(N_ANALYSIS):
            ii,tt=_nearest_norm_match(sn,si,st,norms[m,t],m)
            out_i[m,t]=ii; out_t[m,t]=tt
    return out_i,out_t


def estimate_j_for_states(core,h,directions,eps):
    V=torch.from_numpy(directions).to(dtype=h.dtype,device=h.device)
    K=V.shape[0]; outs=[]
    with torch.no_grad():
        for a in range(0,len(h),BATCH):
            x=h[a:a+BATCH]
            xp=(x[:,None,:] + eps*V[None,:,:]).reshape(-1,LATENT)
            xm=(x[:,None,:] - eps*V[None,:,:]).reshape(-1,LATENT)
            rp=core.F(xp).reshape(len(x),K,LATENT)
            rm=core.F(xm).reshape(len(x),K,LATENT)
            response=(rp-rm)/(2.0*eps)
            J=response.transpose(1,2) @ V
            outs.append(J.cpu())
    return torch.cat(outs,0)


def all_time_j(core,tr,directions,eps):
    arr=[]
    for t in range(STEPS+1):
        arr.append(estimate_j_for_states(core,tr[:,t],directions,eps))
    return torch.stack(arr,1)


def frob(x): return torch.linalg.vector_norm(x,dim=(-2,-1))

def pair_metrics(A,B):
    na=frob(A); nb=frob(B); d=frob(A-B)
    r=d/(0.5*(na+nb)+ETA)
    cos=(A*B).sum(dim=(-2,-1))/(na*nb+ETA)
    return d,r,cos,na,nb

def meanop_metrics(A,B):
    a=A.mean(0); b=B.mean(0)
    na=torch.linalg.vector_norm(a); nb=torch.linalg.vector_norm(b); d=torch.linalg.vector_norm(a-b)
    r=d/(0.5*(na+nb)+ETA)
    cos=(a*b).sum()/(na*nb+ETA)
    return a,b,float(d),float(r),float(cos),float(na),float(nb)

def op_diagnostics(A,B):
    a,b,_,_,_,_,_=meanop_metrics(A,B)
    sv=torch.linalg.svdvals(a)
    p=sv/(sv.sum()+ETA)
    eff=float(torch.exp(-(p*torch.log(p+ETA)).sum()))
    anis=float(sv[0]/(sv.mean()+ETA))
    sym=(a+a.T)/2; anti=(a-a.T)/2; n=torch.linalg.vector_norm(a)+ETA
    _,Ua=torch.linalg.eigh(a@a.T)
    _,Ub=torch.linalg.eigh(b@b.T)
    Ua=Ua[:,-4:]; Ub=Ub[:,-4:]
    align=float(torch.linalg.vector_norm(Ua.T@Ub)**2/4.0)
    return {
        'effective_rank_meanop':eff,
        'anisotropy_s1_over_mean_s':anis,
        'symmetric_fraction_meanop':float(torch.linalg.vector_norm(sym)/n),
        'antisymmetric_fraction_meanop':float(torch.linalg.vector_norm(anti)/n),
        'top4_left_subspace_alignment':align,
        'singular_values_meanop':[float(x) for x in sv],
    }


def bootstrap_advantage(seed,t,ei,r_cross,r_same,nboot=2000):
    x=(r_cross-r_same).cpu().numpy()
    rng=np.random.default_rng((seed*1000003 + t*1009 + ei*97 + 20260907) % (2**32))
    vals=np.empty(nboot,dtype=np.float64); n=len(x)
    for b in range(nboot): vals[b]=x[rng.integers(0,n,n)].mean()
    return float(x.mean()),[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]


def gather_time(J,time_perm,t):
    idx=torch.from_numpy(time_perm[:,t]).long()
    return J[torch.arange(N_ANALYSIS),idx]


def gather_state(J,mi,mt,t):
    ii=torch.from_numpy(mi[:,t]).long(); tt=torch.from_numpy(mt[:,t]).long()
    return J[ii,tt]


def run_seed(args):
    if args.seed not in PRIMARY_SEEDS: raise ValueError('seed must be preregistered 7, 19, or 43')
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    core,core_summary,core_sha=load_core(args.core_path,args.summary_path)
    memories,perms,time_perm,directions,bank_hashes=make_fixed_banks()
    tr=generate_trajectories(core,memories,perms)
    scale=float(torch.linalg.vector_norm(tr[:,0],dim=-1).median())
    cross_i,cross_t=build_norm_matches(tr,True)
    same_i,same_t=build_norm_matches(tr,False)
    dir_perm=np.roll(np.arange(LATENT),-5)
    V=torch.from_numpy(directions).float(); P=torch.eye(LATENT)[:,torch.from_numpy(dir_perm).long()]
    Q=V.T @ P @ V
    cells=[]
    for ei,frac in enumerate(EPS_FRACS):
        eps=frac*scale
        J=all_time_j(core,tr,directions,eps)
        Jh=all_time_j(core,tr,directions,eps/2.0)
        Jd=all_time_j(core,tr,directions,eps*2.0)
        for t in range(STEPS+1):
            A=J[:,t]
            Btemp=gather_time(J,time_perm,t)
            Bcross=gather_state(J,cross_i,cross_t,t)
            Bsame=gather_state(J,same_i,same_t,t)
            Bdir=A @ Q
            dt,rt,ct,nat_n,temp_n=pair_metrics(A,Btemp)
            dx,rx,cx,_,cross_n=pair_metrics(A,Bcross)
            dw,rw,cw,_,same_n=pair_metrics(A,Bsame)
            dd,rd,cd,_,dir_n=pair_metrics(A,Bdir)
            _,_,mdt,mrt,mct,mna,mnb=meanop_metrics(A,Btemp)
            _,_,mdx,mrx,mcx,_,mxb=meanop_metrics(A,Bcross)
            _,_,mdw,mrw,mcw,_,mwb=meanop_metrics(A,Bsame)
            _,_,mdd,mrd,mcd,_,mdb=meanop_metrics(A,Bdir)
            conv_h=frob(A-Jh[:,t])/(frob(A)+ETA)
            conv_d=frob(A-Jd[:,t])/(frob(A)+ETA)
            adv,adv_ci=bootstrap_advantage(args.seed,t,ei,rx,rw)
            diag=op_diagnostics(A,Btemp)
            cell={
                'seed':args.seed,'t':t,'epsilon_fraction':frac,'epsilon_absolute':eps,
                'native_state_norm_mean':float(torch.linalg.vector_norm(tr[:,t],dim=-1).mean()),
                'native_response_norm_pair_mean':float(nat_n.mean()),
                'temporal_response_norm_pair_mean':float(temp_n.mean()),
                'cross_time_radius_response_norm_pair_mean':float(cross_n.mean()),
                'same_time_radius_response_norm_pair_mean':float(same_n.mean()),
                'direction_response_norm_pair_mean':float(dir_n.mean()),
                'pair_D_temporal':float(dt.mean()),'pair_R_temporal':float(rt.mean()),'pair_cos_temporal':float(ct.mean()),
                'pair_D_cross_time_radius':float(dx.mean()),'pair_R_cross_time_radius':float(rx.mean()),'pair_cos_cross_time_radius':float(cx.mean()),
                'pair_D_same_time_radius':float(dw.mean()),'pair_R_same_time_radius':float(rw.mean()),'pair_cos_same_time_radius':float(cw.mean()),
                'pair_D_direction':float(dd.mean()),'pair_R_direction':float(rd.mean()),'pair_cos_direction':float(cd.mean()),
                'meanop_D_temporal':mdt,'meanop_R_temporal':mrt,'meanop_cos_temporal':mct,
                'meanop_D_cross_time_radius':mdx,'meanop_R_cross_time_radius':mrx,'meanop_cos_cross_time_radius':mcx,
                'meanop_D_same_time_radius':mdw,'meanop_R_same_time_radius':mrw,'meanop_cos_same_time_radius':mcw,
                'meanop_D_direction':mdd,'meanop_R_direction':mrd,'meanop_cos_direction':mcd,
                'meanop_native_norm':mna,'meanop_temporal_norm':mnb,'meanop_cross_time_radius_norm':mxb,'meanop_same_time_radius_norm':mwb,'meanop_direction_norm':mdb,
                'pair_R_cross_minus_same':adv,'pair_R_advantage_bootstrap_ci95':adv_ci,
                'pair_cos_same_minus_cross':float(cw.mean()-cx.mean()),
                'convergence_rel_half_mean':float(conv_h.mean()),
                'convergence_rel_double_mean':float(conv_d.mean()),
                'numerically_stable_10pct':bool(float(conv_h.mean()) <= .10 and float(conv_d.mean()) <= .10),
            }
            cell.update(diag); cells.append(cell)
        del J,Jh,Jd
    config={
        'experiment':'JTP-1','status':'executed_frozen_after_preoutcome_amendment_1','seed':args.seed,
        'source_core_checkpoint_sha256':core_sha,
        'source_core_summary':core_summary,
        'analysis_n':N_ANALYSIS,'times':list(range(STEPS+1)),
        'epsilon_fractions':list(EPS_FRACS),'epsilon_scale_definition':'median ||h0|| over fixed analysis bank',
        'epsilon_scale_value':scale,'direction_bank':'16x16 normalized Sylvester-Hadamard, rows as directions',
        'direction_response_permutation':'cyclic response-column shift by 5',
        'cross_time_radius_match':'nearest latent norm among different-memory, different-time states',
        'same_time_radius_match':'nearest latent norm among different-memory states at the same trajectory time',
        'temporal_control':'per-memory fixed-point-free random permutation of the 13 time indices',
        'finite_difference':'symmetric central difference; every primary cell checked at eps/2 and 2eps',
        'bank_hashes':bank_hashes,
        'trajectory_sha256':hashlib.sha256(tr.numpy().tobytes()).hexdigest(),
        'cross_match_memory_index_sha256':sha_array(cross_i),'cross_match_time_index_sha256':sha_array(cross_t),
        'same_match_memory_index_sha256':sha_array(same_i),'same_match_time_index_sha256':sha_array(same_t),
        'torch_version':torch.__version__,'numpy_version':np.__version__,'python':sys.version,'platform':platform.platform(),
    }
    save_json(out/'manifest.json',config); save_json(out/'cells.json',cells)
    flatkeys=[k for k,v in cells[0].items() if not isinstance(v,(list,dict))]
    with (out/'cell_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=flatkeys); w.writeheader()
        for c in cells: w.writerow({k:c[k] for k in flatkeys})
    summary={
        'seed':args.seed,'core_sha256':core_sha,'epsilon_scale':scale,
        'stable_cells':sum(c['numerically_stable_10pct'] for c in cells),
        'total_cells':len(cells),
        'max_meanop_R_temporal':max(c['meanop_R_temporal'] for c in cells),
        'max_pair_R_advantage':max(c['pair_R_cross_minus_same'] for c in cells),
    }
    save_json(out/'summary.json',summary)
    print(json.dumps(summary,indent=2))


def rankcorr(x,y):
    x=np.asarray(x); y=np.asarray(y)
    rx=np.empty_like(np.argsort(x),dtype=float); ry=np.empty_like(np.argsort(y),dtype=float)
    rx[np.argsort(x,kind='mergesort')]=np.arange(len(x)); ry[np.argsort(y,kind='mergesort')]=np.arange(len(y))
    return float(np.corrcoef(rx,ry)[0,1])


def aggregate(args):
    root=Path(args.aggregate); files=list(root.rglob('cells.json'))
    if len(files) != 3: raise RuntimeError(f'expected 3 seed cell files, found {len(files)}')
    byseed={}
    for f in files:
        cells=json.loads(f.read_text()); byseed[int(cells[0]['seed'])]=cells
    if set(byseed) != set(PRIMARY_SEEDS): raise RuntimeError(byseed.keys())
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    rows=[]; consensus=[]
    for ei,frac in enumerate(EPS_FRACS):
        for t in range(STEPS+1):
            cs=[next(c for c in byseed[s] if c['t']==t and c['epsilon_fraction']==frac) for s in PRIMARY_SEEDS]
            row={'t':t,'epsilon_fraction':frac}
            for key in ('meanop_D_temporal','meanop_R_temporal','meanop_cos_temporal','meanop_R_cross_time_radius','meanop_cos_cross_time_radius','meanop_R_same_time_radius','meanop_cos_same_time_radius','pair_R_cross_minus_same','pair_cos_same_minus_cross','native_response_norm_pair_mean','convergence_rel_half_mean','convergence_rel_double_mean'):
                vals=[c[key] for c in cs]; row[key+'_seedmean']=float(np.mean(vals)); row[key+'_seedmin']=float(np.min(vals)); row[key+'_seedmax']=float(np.max(vals))
            stable=all(c['numerically_stable_10pct'] for c in cs)
            geom=all(c['pair_R_cross_minus_same'] >= .05 and c['pair_R_advantage_bootstrap_ci95'][0] > 0 and c['pair_cos_same_minus_cross'] > 0 for c in cs)
            row['stable_all_seeds']=stable; row['geometric_cell_all_seeds']=bool(stable and geom)
            rows.append(row)
            if row['geometric_cell_all_seeds']: consensus.append((t,ei,row['pair_R_cross_minus_same_seedmean']))
    if consensus:
        cells_set={(t,e) for t,e,_ in consensus}; visited=set(); comps=[]
        for node in cells_set:
            if node in visited: continue
            stack=[node]; comp=[]; visited.add(node)
            while stack:
                u=stack.pop(); comp.append(u)
                for v in ((u[0]-1,u[1]),(u[0]+1,u[1]),(u[0],u[1]-1),(u[0],u[1]+1)):
                    if v in cells_set and v not in visited: visited.add(v); stack.append(v)
            comps.append(comp)
        biggest=max(comps,key=len)
    else: biggest=[]
    qualifies_C=(len(consensus)>=4 and len({t for t,_,_ in consensus})>=2 and len({e for _,e,_ in consensus})>=2)
    qualifies_D=False; peak=None
    if qualifies_C:
        peak=max(consensus,key=lambda z:z[2]); pt,pe,pv=peak
        def val(t,e):
            rr=next(r for r in rows if r['t']==t and r['epsilon_fraction']==EPS_FRACS[e])
            return rr['pair_R_cross_minus_same_seedmean']
        eps_boundary=max(val(pt,0),val(pt,len(EPS_FRACS)-1))
        time_boundary=max(val(0,pe),val(STEPS,pe))
        qualifies_D=(0<pt<STEPS and 0<pe<len(EPS_FRACS)-1 and len(biggest)>=4 and pv>=1.5*eps_boundary and pv>=1.5*time_boundary)
    if qualifies_D: outcome='D — bounded trajectory–perturbation regime'
    elif qualifies_C: outcome='C — time-specific geometric structure beyond radius-matched within-time variation'
    else:
        stable_rows=[r for r in rows if r['stable_all_seeds']]
        if not stable_rows: outcome='A — no valid trajectory-dependent response structure'
        else:
            corr=rankcorr([r['meanop_D_temporal_seedmean'] for r in stable_rows],[r['native_response_norm_pair_mean_seedmean'] for r in stable_rows])
            maxadv=max(r['pair_R_cross_minus_same_seedmean'] for r in stable_rows)
            if abs(corr)>=.80 or maxadv < .05: outcome='B — sensitivity/contraction or generic state-geometry explanation'
            else: outcome='A — no preregistered time-specific geometric trajectory structure'
    result={
        'experiment':'JTP-1','primary_seeds':list(PRIMARY_SEEDS),'preoutcome_amendment':1,'outcome':outcome,
        'consensus_geometric_cells':len(consensus),'largest_connected_consensus_component':len(biggest),
        'peak_consensus_cell':None if peak is None else {'t':peak[0],'epsilon_fraction':EPS_FRACS[peak[1]],'pair_R_advantage_seedmean':peak[2]},
        'decision_rule':{
            'numerical_stability':'mean relative operator change <= 0.10 for both eps/2 and 2eps in all seeds',
            'geometric_cell':'all seeds: cross-time radius-matched R minus same-time radius-matched R >=0.05, bootstrap CI lower bound >0, cosine(same)-cosine(cross)>0, and stable',
            'C':'at least 4 geometric cells spanning >=2 times and >=2 epsilon levels',
            'D':'C plus largest connected component >=4, interior peak, and peak >=1.5x both epsilon-boundary and time-boundary effects',
            'B':'if C fails, stable-grid |rank corr(D,response norm)|>=0.80 or maximum cross-time-over-same-time radius-matched normalized advantage <0.05',
        }
    }
    save_json(out/'aggregate_result.json',result); save_json(out/'aggregate_cells.json',rows)
    keys=list(rows[0].keys())
    with (out/'aggregate_cells.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
    md=['# JTP-1 Final Result','',f"**Primary classification:** {outcome}",'',f"Consensus geometric cells: {len(consensus)} / {len(rows)}",f"Largest connected consensus component: {len(biggest)}"]
    if peak: md += [f"Peak consensus cell: t={peak[0]}, epsilon fraction={EPS_FRACS[peak[1]]}, seed-mean cross-time-over-same-time radius-matched normalized advantage={peak[2]:.6f}"]
    md += ['','This classification was produced by the frozen decision rule after PREOUTCOME_AMENDMENT_1, committed before replacement-run result inspection.','','See `aggregate_cells.csv`, each seed artifact, the original preregistration, and the pre-outcome amendment for the full diagnostics and claim boundary.']
    (out/'FINAL_RESULT.md').write_text('\n'.join(md)+'\n')
    print(json.dumps(result,indent=2))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int); ap.add_argument('--core-path'); ap.add_argument('--summary-path'); ap.add_argument('--outdir',required=True); ap.add_argument('--aggregate')
    args=ap.parse_args()
    if args.aggregate: aggregate(args)
    else:
        if args.seed is None or not args.core_path or not args.summary_path: ap.error('seed/core-path/summary-path required')
        run_seed(args)
if __name__=='__main__': main()
