"""R9-S1 frozen runner. Scientific runs accept no tuning overrides."""
import argparse
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from experiments.r9_t1 import run_r9_t1 as task
from experiments.r9_s1.models import ARMS,N,H,build_arm,seed_for
from experiments.r9_s1 import diagnostics as dx

VERSION='R9-S1-v1'
PREREG='ca4da167c3be7f7b4ff98e11bf969ff41041eb94'
TASK_BLOB='b9ba7615f5cf70354b6c9b7da6cb13fabea567d6'
SEEDS=(3109,3137,3163,3191,3221,3251,3271,3301)
TRAIN_STEPS=1000;BATCH=32;EVAL_EPISODES=1024;PROBE_EPISODES=256;INTERVENTION_EPISODES=256;PAIRED_EPISODES=128
LR=0.001;WD=0.0001;CLIP=1.0
SOURCE_FILES=('experiments/r9_s1/PREREGISTRATION.md','experiments/r9_s1/models.py','experiments/r9_s1/diagnostics.py','experiments/r9_s1/run_r9_s1.py','experiments/r9_s1/classify_r9_s1.py','experiments/r9_s1/requirements.txt')


def setup():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled=False
    torch.manual_seed(1);np.random.seed(1)


def source_manifest(expected_commit=None):
    path=Path(task.__file__).resolve();data=path.read_bytes()
    blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    if blob!=TASK_BLOB:raise RuntimeError('Frozen R9-T1 task source mismatch: '+blob)
    files={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCE_FILES}
    sha=os.environ.get('GITHUB_SHA') or expected_commit
    if expected_commit and sha!=expected_commit:raise RuntimeError('Workflow commit mismatch')
    return {'task_git_blob':blob,'task_sha256':hashlib.sha256(data).hexdigest(),'preregistration_commit':PREREG,'implementation_commit':sha,'source_sha256':files,'protocol_version':VERSION}


def batch(seed,name,n):return task.make_batch(seed_for(seed,name),n)


def finite(obj):
    if isinstance(obj,dict):return all(finite(v) for v in obj.values())
    if isinstance(obj,(list,tuple)):return all(finite(v) for v in obj)
    if isinstance(obj,(int,float,np.number)) and not isinstance(obj,(bool,np.bool_)):return math.isfinite(float(obj))
    return True


def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))


def train(model,seed,smoke=False):
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
    trace=[];model.train()
    for step in range(2 if smoke else TRAIN_STEPS):
        b=batch(seed,f'train_{step}',4 if smoke else BATCH)
        opt.zero_grad(set_to_none=True)
        logits=model(b['types'],b['payload'])
        mask=b['target']>=0
        loss=F.cross_entropy(logits[mask],b['target'][mask])
        if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),CLIP)
        if not torch.isfinite(grad):raise RuntimeError('nonfinite gradient')
        opt.step()
        if not smoke and (step%100==0 or step==TRAIN_STEPS-1):
            trace.append({'step':step,'loss':float(loss.detach()),'gradient_norm':float(grad.detach())})
    return trace,opt


def task_semantics(b):
    typ=b['types'];pay=b['payload'];target=b['target'];ak=b['active_key']
    mask=target>=0
    if not bool((target[mask]==(pay[mask]+ak[mask])%task.N_VAL).all()):return False
    for i in range(len(typ)):
        acts=pay[i][typ[i]==task.ACTIVATE].tolist()
        defs=pay[i][typ[i]==task.REGIME].tolist()
        if defs!=[0,1,2] or acts!=[0,1,2]:return False
        seen={}
        for t in range(typ.shape[1]):
            k=int(typ[i,t]);p=int(pay[i,t])
            if k==task.REGIME:
                if t+1>=typ.shape[1] or int(typ[i,t+1])!=task.KEY:return False
                seen[p]=int(pay[i,t+1])
            if k==task.ACTIVATE:
                if p not in seen or int(typ[i,t+1])==task.KEY:return False
    return True


def smoke(seed,outdir,manifest):
    setup();b=batch(seed,'smoke',4);assert task_semantics(b)
    checks={};counts={}
    for name in ARMS:
        model=build_arm(name,seed);counts[name]=sum(p.numel() for p in model.parameters())
        train(model,seed,True)
        model.eval()
        with torch.no_grad():
            y,h=model(b['types'],b['payload'],True)
            assert y.shape==(4,task.SEQ_LEN,task.N_VAL)
            assert h.shape==(4,task.SEQ_LEN,64) if name=='MONOLITHIC' else h.shape==(4,task.SEQ_LEN,N,H)
            assert bool(torch.isfinite(y).all() and torch.isfinite(h).all())
            for cut in (24,48):
                yy,hh=model(b['types'][:,:cut],b['payload'][:,:cut],True)
                assert torch.allclose(yy,y[:,:cut],atol=1e-6,rtol=1e-6)
                assert torch.allclose(hh,h[:,:cut],atol=1e-6,rtol=1e-6)
            if name!='MONOLITHIC':
                e=model.encode(b['types'],b['payload']);state=e.new_zeros(4,N,H)
                manual=[]
                for t in range(task.SEQ_LEN):
                    state=model.step(state,e[:,t],b['types'][:,t]);manual.append(state)
                assert torch.allclose(torch.stack(manual,1),h,atol=1e-6,rtol=1e-6)
                assert bool(torch.diag(model.graph).eq(0).all())
                if name=='DISCONNECTED':assert bool(model.graph.eq(0).all())
                elif name=='LEARNED':
                    a=model.routing(h[:,5]);assert a.shape==(4,N,N)
                    assert bool((a>0).sum(-1).eq(2).all())
                    assert torch.allclose(a.sum(-1),torch.ones(4,N),atol=1e-6)
                else:
                    assert torch.allclose(model.graph.sum(-1),torch.ones(N),atol=1e-6)
                    assert bool((model.graph>0).sum(-1).eq(7 if name.startswith('FULL') else 2).all())
                p=[torch.arange(4) for _ in range(task.SEQ_LEN)]
                yi,_=model(b['types'],b['payload'],True,'shuffle',p)
                assert torch.allclose(y,yi,atol=1e-6,rtol=1e-6)
                if name=='DISCONNECTED':
                    yz,_=model(b['types'],b['payload'],True,'zero')
                    assert torch.allclose(y,yz,atol=1e-6,rtol=1e-6)
                if name in ('FULL','FULL_HET'):
                    yr,_=model(b['types'],b['payload'],True,'rewire')
                    assert torch.allclose(y,yr,atol=1e-6,rtol=1e-6)
        checks[name]=True
    assert len({counts[n] for n in ('DISCONNECTED','RING','RANDOM','FULL')})==1
    from experiments.r9_s1.classify_r9_s1 import self_check
    assert self_check()
    out={'experiment':'R9-S1','protocol_version':VERSION,'smoke_only':True,'seed':seed,'all_valid':True,'checks':checks,'parameter_counts':counts,'task_semantics':True,'causal_prefix':True,'synchronous_equivalence':True,'deterministic_replay':True,'source':manifest,'scientific_metrics_emitted':False}
    save(Path(outdir)/'smoke_summary.json',out)
    print('R9-S1 outcome-free smoke PASSED',flush=True)


def run_family(seed,outdir,manifest):
    if seed not in SEEDS:raise ValueError('Seed not preregistered')
    setup();out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    models={};architecture={}
    for name in ARMS:
        model=build_arm(name,seed)
        tr,opt=train(model,seed)
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'seed':seed,'arm':name,'source':manifest,'train_trace':tr},out/(name+'.pt'))
        models[name]=model
        architecture[name]={'params':sum(p.numel() for p in model.parameters()),'gradient_bearing_params':sum(p.numel() for p in model.parameters() if p.grad is not None),'graph':model.graph.tolist() if name!='MONOLITHIC' else None,'rewired_graph':model.rewired.tolist() if name!='MONOLITHIC' else None,'train_trace':tr}
        print('trained '+name,flush=True)
    test=batch(seed,'test',EVAL_EPISODES)
    probes=[batch(seed,'probe_'+name,PROBE_EPISODES) for name in ('train','validation','test')]
    intervention=batch(seed,'intervention',INTERVENTION_EPISODES)
    if not task_semantics(test):raise RuntimeError('task semantics failure')
    native={};fits={}
    for name,model in models.items():
        model.eval()
        logits,_=dx.trace(model,test)
        architecture[name]['test']=dx.metrics(logits,test)
        native[name],fits[name]=dx.native_diagnostics(model,probes,seed_for(seed,'diagnostic_'+name))
        print('native diagnostics complete '+name,flush=True)
    causal={}
    for name,model in models.items():
        causal[name]=dx.causal_diagnostics(model,intervention,seed_for(seed,'causal_'+name),fits[name])
        print('causal diagnostics complete '+name,flush=True)
    result={'experiment':'R9-S1','protocol_version':VERSION,'seed':seed,'frozen_seed':True,'source':manifest,'architecture':architecture,'native':native,'causal':causal,'configuration':{'train_steps':TRAIN_STEPS,'batch':BATCH,'evaluation_episodes':EVAL_EPISODES,'probe_episodes_per_split':PROBE_EPISODES,'intervention_episodes':INTERVENTION_EPISODES,'paired_history_episodes':PAIRED_EPISODES,'seeds':list(SEEDS),'lr':LR,'weight_decay':WD,'gradient_clip':CLIP,'ridge_lambdas':list(dx.LAMBDAS)},'environment':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform()},'all_valid':True}
    result['all_valid']=finite(result)
    if not result['all_valid']:raise RuntimeError('nonfinite family result')
    save(out/'family_result.json',result)
    print('R9-S1 family complete seed='+str(seed),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--outdir',required=True);ap.add_argument('--expected-commit');ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    manifest=source_manifest(args.expected_commit)
    if args.smoke:smoke(args.seed,args.outdir,manifest)
    else:run_family(args.seed,args.outdir,manifest)

if __name__=='__main__':main()
