"""Frozen R9-S1 native and separately justified causal measurements."""
import itertools
import math
import numpy as np
import torch
from torch.nn import functional as F
from experiments.r9_t1 import run_r9_t1 as task
from experiments.r9_s1.models import N,H,seed_for

LAMBDAS=(0.0001,0.01,1.0,100.0)


def trace(model,b,mode='native',perms=None,silence=None,readout_zero=None,chunk=64):
    model.eval(); logits=[]; states=[]
    if perms is not None:chunk=len(b['types'])
    with torch.no_grad():
        for start in range(0,len(b['types']),chunk):
            end=start+chunk
            pp=None if perms is None else [p[start:end] for p in perms]
            y,h=model(b['types'][start:end],b['payload'][start:end],True,mode,pp,silence,readout_zero)
            logits.append(y.cpu());states.append(h.cpu())
    return torch.cat(logits),torch.cat(states)


def metrics(logits,b):
    pred=logits.argmax(-1);out={}
    for name,mask in [('DATA',b['target']>=0),('NEW_EARLY',b['new_early']),('RETURN_EARLY',b['return_early']),('STEADY',b['steady']),('LONG_GAP',b['long_gap'])]:
        n=int(mask.sum());out[name+'_N']=n
        out[name+'_ACC']=float((pred[mask]==b['target'][mask]).float().mean()) if n else None
    mask=b['target']>=0
    out['CROSS_ENTROPY']=float(F.cross_entropy(logits[mask],b['target'][mask]))
    return out


def group_perms(b,seed):
    rng=np.random.default_rng(seed);typ=b['types'].numpy();B,T=typ.shape
    p=np.tile(np.arange(B),(T,1))
    for t in range(T):
        for kind in range(task.N_TYPES):
            idx=np.flatnonzero(typ[:,t]==kind)
            if len(idx)>1:p[t,idx]=rng.permutation(idx)
    return torch.from_numpy(p.copy()).long()


def streams(h):
    if h.ndim==3:return {'JOINT':h.numpy()}
    return {'JOINT':h.reshape(h.shape[0],h.shape[1],-1).numpy(),**{f'N{i}':h[:,:,i].numpy() for i in range(N)}}


def selected(b,target):
    mask=(b['active_key']>=0);mask[:,:2]=False
    if target=='answer':mask &= b['target']>=0
    return mask.numpy()


def label(b,target):
    return b['active_key'].numpy() if target=='key' else b['target'].numpy()


def standardize(X,mu=None,sd=None):
    X=np.asarray(X,np.float64)
    if mu is None:mu=X.mean(0);sd=np.maximum(X.std(0),1e-6)
    z=(X-mu)/sd
    return z,mu,sd


def _scores(X,fit):
    z,_,_=standardize(X,fit['mu'],fit['sd'])
    return np.column_stack([z,np.ones(len(z))])@fit['w']


def _ce(scores,y):
    z=scores-scores.max(1,keepdims=True)
    logp=z-np.log(np.exp(z).sum(1,keepdims=True))
    return float(-logp[np.arange(len(y)),y].mean())


def assess(fit,X,y):
    s=_scores(X,fit)
    return {'acc':float(np.mean(s.argmax(1)==y)),'ce':_ce(s,y),'n':int(len(y))}


def fit_probe(X,y,V,v,return_fit=False):
    X,mu,sd=standardize(X); V,_,_=standardize(V,mu,sd)
    X=np.column_stack([X,np.ones(len(X))]);V=np.column_stack([V,np.ones(len(V))])
    Y=np.eye(task.N_VAL,dtype=np.float64)[y]
    G=X.T@X/len(X); C=X.T@Y/len(X)
    penalty=np.eye(G.shape[0]);penalty[-1,-1]=0
    best=None
    for lam in LAMBDAS:
        w=np.linalg.solve(G+lam*penalty,C)
        sc=V@w;ce=_ce(sc,v)
        if best is None or ce<best[0]:best=(ce,lam,w)
    fit={'mu':mu,'sd':sd,'w':best[2],'lambda':best[1]}
    out={'validation_ce':best[0],'validation_acc':float(np.mean((V@best[2]).argmax(1)==v)), 'lambda':best[1],'dim':int(X.shape[1]-1),'train_n':int(len(y)),'validation_n':int(len(v))}
    return (out,fit) if return_fit else out


def probe3(Xs,ys,return_fit=False):
    info,fit=fit_probe(Xs[0],ys[0],Xs[1],ys[1],True)
    result={**info,**assess(fit,Xs[2],ys[2])}
    return (result,fit) if return_fit else result


def flatten_features(h,b,target,variant='state',perm=None):
    d=np.zeros_like(h);d[:,1:]=h[:,1:]-h[:,:-1]
    dp=np.zeros_like(h);dp[:,2:]=d[:,1:-1]
    zero=np.zeros_like(h)
    if variant=='state':x=h
    elif variant=='one':x=np.concatenate([h,d],-1)
    elif variant=='multi':x=np.concatenate([h,d,dp],-1)
    elif variant=='disp':x=np.concatenate([d,dp],-1)
    elif variant=='shuf':
        p=perm.numpy().T;ds=d[p,np.arange(h.shape[1])[None,:]]
        x=np.concatenate([h,ds],-1)
    elif variant=='zero':x=np.concatenate([h,zero],-1)
    elif variant=='dup':x=np.concatenate([h,h],-1)
    elif variant=='ordered':x=np.concatenate([h,dp,d],-1)
    elif variant=='reversed':x=np.concatenate([h,d,dp],-1)
    elif variant=='integrated':x=np.concatenate([h,dp+d],-1)
    elif variant=='first':x=np.concatenate([h,dp],-1)
    elif variant=='final':x=np.concatenate([h,d],-1)
    else:raise ValueError(variant)
    mask=selected(b,target);y=label(b,target)[mask]
    return x[mask],y


def history_probes(traces,batches,perms,targets=('key','answer')):
    all_results={};fits={}
    for stream in streams(traces[0]):
        hs=[streams(h)[stream] for h in traces]
        all_results[stream]={};fits[stream]={}
        for target in targets:
            results={};localfits={}
            variants=('state','one','multi','disp','shuf','zero','dup')
            for variant in variants:
                data=[flatten_features(h,b,target,variant,p) for h,b,p in zip(hs,batches,perms)]
                result,fit=probe3([z[0] for z in data],[z[1] for z in data],True)
                results[variant]=result;localfits[variant]=fit
            results['H1']=results['one']['acc']-results['state']['acc']
            results['HS']=results['one']['acc']-results['shuf']['acc']
            results['H2']=results['multi']['acc']-results['state']['acc']
            results['H_DUP']=results['one']['acc']-results['dup']['acc']
            all_results[stream][target]=results;fits[stream][target]=localfits
    return all_results,fits


def order_controls(traces,batches,perms):
    out={}
    for stream in streams(traces[0]):
        hs=[streams(h)[stream] for h in traces];out[stream]={}
        for target in ('key','answer'):
            data=[flatten_features(h,b,target,'ordered',p) for h,b,p in zip(hs,batches,perms)]
            native,fit=probe3([z[0] for z in data],[z[1] for z in data],True)
            test_h,test_b,test_p=hs[2],batches[2],perms[2]
            # Keep the native reader fixed for reversal; retraining on permuted columns is an equivalent linear model.
            X,y=flatten_features(test_h,test_b,target,'reversed',test_p)
            native_rev=assess(fit,X,y)
            result={'ordered':native,'frozen_reversed':native_rev,'frozen_reverse_gap':native['acc']-native_rev['acc']}
            for variant in ('integrated','first','final','shuf'):
                data=[flatten_features(h,b,target,variant,p) for h,b,p in zip(hs,batches,perms)]
                result[variant]=probe3([z[0] for z in data],[z[1] for z in data])
                result['vs_'+variant]=native['acc']-result[variant]['acc']
            out[stream][target]=result
    return out


def pad64(x):
    if x.shape[-1]>64:raise ValueError('feature exceeds64')
    return np.pad(x,((0,0),(0,64-x.shape[-1])))


def complementarity(traces,batches,perms,history):
    if traces[0].ndim==3:return None
    ss=[streams(h) for h in traces];ys=[label(b,'key')[selected(b,'key')] for b in batches]
    nodes=[[s[f'N{i}'][selected(b,'key')] for s,b in zip(ss,batches)] for i in range(N)]
    nodeinfo=[]
    for i in range(N):
        r=probe3([pad64(z) for z in nodes[i]],ys);nodeinfo.append(r)
    best=max(range(N),key=lambda i:(nodeinfo[i]['validation_acc'],-i))
    pairs={}
    for i,j in itertools.combinations(range(N),2):
        data=[pad64(np.concatenate([nodes[i][k],nodes[j][k]],1)) for k in range(3)]
        pairs[f'{i},{j}']=probe3(data,ys)
    bestpair=max(pairs,key=lambda k:(pairs[k]['validation_acc'],-int(k.split(',')[0]),-int(k.split(',')[1])))
    dups=[];zeros=[];shufs=[];joints=[]
    for s,b,p in zip(ss,batches,perms):
        mask=selected(b,'key');joint=s['JOINT'];anchor=s[f'N{best}']
        mixed=joint.reshape(joint.shape[0],joint.shape[1],N,H).copy()
        q=p.numpy().T
        for j in range(N):
            if j!=best:mixed[:,:,j]=mixed[q,np.arange(joint.shape[1])[None,:],j]
        a=anchor[mask];dups.append(np.tile(a,(1,8)));zeros.append(pad64(a));shufs.append(mixed.reshape(joint.shape)[mask]);joints.append(joint[mask])
    joint=probe3(joints,ys);shuf=probe3(shufs,ys);dup=probe3(dups,ys);zero=probe3(zeros,ys)
    return {'nodes':nodeinfo,'best_node':best,'best_node_padded':nodeinfo[best],'best_pair':bestpair,'best_pair_result':pairs[bestpair], 'joint':joint,'shuffled_pair':shuf,'duplicated_best':dup,'zero_padded_best':zero,'joint_minus_best':joint['acc']-nodeinfo[best]['acc'],'joint_minus_shuffle':joint['acc']-shuf['acc'],'joint_minus_pair':joint['acc']-pairs[bestpair]['acc']}


def cka(a,b):
    a=a-a.mean(0);b=b-b.mean(0)
    x=np.linalg.norm(a.T@b,'fro')**2
    den=np.linalg.norm(a.T@a,'fro')*np.linalg.norm(b.T@b,'fro')
    return float(x/max(den,1e-12))


def _corr(x,y):
    x=x-x.mean();y=y-y.mean();den=np.linalg.norm(x)*np.linalg.norm(y)
    return float(np.dot(x,y)/den) if den>1e-12 else 0.0


def native_organization(h,b,perms,fits):
    if h.ndim==3:return {'joint_state_norm':float(torch.linalg.vector_norm(h,dim=-1).mean()),'node_statistics':None,'cka':None,'lag_correlations':None,'reactivation':reactivation(h,b,fits)}
    h=h.numpy();d=np.zeros_like(h);d[:,1:]=h[:,1:]-h[:,:-1]
    mask=selected(b,'key');norm=np.linalg.norm(h,axis=-1);speed=np.linalg.norm(d,axis=-1)
    X=h[mask];idx=np.linspace(0,len(X)-1,min(2048,len(X))).astype(int);X=X[idx]
    matrix=np.array([[cka(X[:,i],X[:,j]) for j in range(N)] for i in range(N)])
    valid=(b['active_key']>=0).numpy();valid[:,:2]=False
    per=perms.numpy().T;sh=speed[per,np.arange(speed.shape[1])[None,:]]
    rev=speed.copy()
    for i in range(len(rev)):
        length=int((b['types'][i]!=task.PAD).sum())
        rev[i,:length]=rev[i,:length][::-1]
    lags={}
    for label,z in [('native',speed),('shuffled',sh),('reversed',rev)]:
        mats=[]
        for lag in range(5):
            m=valid[:,:valid.shape[1]-lag] & valid[:,lag:]
            mats.append([[_corr(speed[:,:speed.shape[1]-lag,i][m],z[:,lag:,j][m]) for j in range(N)] for i in range(N)])
        lags[label]=mats
    return {'node_statistics':[{'state_norm_mean':float(norm[:,:,i][mask].mean()),'state_norm_std':float(norm[:,:,i][mask].std()),'displacement_mean':float(speed[:,:,i][mask].mean()),'displacement_std':float(speed[:,:,i][mask].std())} for i in range(N)],'cka':matrix.tolist(),'lag_correlations':lags,'reactivation':reactivation(torch.from_numpy(h),b,fits)}


def reactivation(h,b,fits):
    ss=streams(h);out={};typ=b['types'].numpy();ret=b['return_early'].numpy()
    rank=np.full(typ.shape,-1,np.int64)
    for i in range(len(typ)):
        r=0
        for t in range(typ.shape[1]):
            if typ[i,t]==task.ACTIVATE:r=0
            if ret[i,t] and r<4:rank[i,t]=r;r+=1
    for stream,arr in ss.items():
        fit=fits[stream]['key']['state'];values=[]
        for j in range(4):
            m=rank==j;values.append(assess(fit,arr[m],b['active_key'].numpy()[m]))
        out[stream]={'rank':values,'last_minus_first':values[3]['acc']-values[0]['acc']}
    return out


def native_diagnostics(model,batches,seed):
    traces=[trace(model,b)[1] for b in batches]
    perms=[group_perms(b,seed_for(seed,f'probe_shuffle_{j}')) for j,b in enumerate(batches)]
    hist,fits=history_probes(traces,batches,perms)
    result={'history':hist,'complementarity':complementarity(traces,batches,perms,hist),'order_controls':order_controls(traces,batches,perms),'organization':native_organization(traces[2],batches[2],perms[2],fits)}
    return result,fits


def communication_controls(model,b,seed,native=None):
    if native is None:native=trace(model,b)[0]
    base=metrics(native,b)
    if model.kind=='MONOLITHIC':
        return {'native':base,'controls':None,'contrasts':None,'note':'No inter-node communication in monolithic control.'}
    perms=group_perms(b,seed_for(seed,'message_shuffle'))
    out={}
    for mode in ('zero','shuffle','rewire'):
        y,_=trace(model,b,mode,perms if mode=='shuffle' else None)
        out[mode]=metrics(y,b)
    return {'native':base,'controls':out,'contrasts':{k:{m:base[m]-v[m] for m in ('DATA_ACC','RETURN_EARLY_ACC','CROSS_ENTROPY')} for k,v in out.items()},'rewire_informative':model.kind not in ('DISCONNECTED','FULL','FULL_HET')}


def _key_access(fit,h,b,zero_node=None):
    x=streams(h)['JOINT']
    if zero_node is not None and h.ndim==4:
        x=h.clone();x[:,:,zero_node]=0;x=x.reshape(x.shape[0],x.shape[1],-1).numpy()
    mask=selected(b,'key')
    return assess(fit,x[mask],label(b,'key')[mask])


def node_ablations(model,b,fits,native=None):
    if model.kind=='MONOLITHIC':return None
    if native is None:native=trace(model,b)
    y,h=native;base=metrics(y,b)
    fit=fits['JOINT']['key']['state'];basekey=_key_access(fit,h,b)
    out=[]
    for i in range(N):
        ys,hs=trace(model,b,silence=i)
        sm=metrics(ys,b);sk=_key_access(fit,hs,b)
        with torch.no_grad():
            z=h.clone();z[:,:,i]=0
            yr=model.head(z.reshape(-1,N*H)).reshape(h.shape[0],h.shape[1],task.N_VAL)
        rm=metrics(yr,b);rk=_key_access(fit,h,b,i)
        out.append({'node':i,'outgoing_zero':sm,'readout_zero':rm,'outgoing_key':sk,'readout_key':rk,'outgoing_effect':{k:base[k]-sm[k] for k in ('DATA_ACC','RETURN_EARLY_ACC','CROSS_ENTROPY')},'readout_effect':{k:base[k]-rm[k] for k in ('DATA_ACC','RETURN_EARLY_ACC','CROSS_ENTROPY')},'outgoing_key_effect':basekey['acc']-sk['acc'],'readout_key_effect':basekey['acc']-rk['acc']})
    return {'native':base,'native_key':basekey,'nodes':out}


def _donors(b,seed,limit=1024):
    rng=np.random.default_rng(seed_for(seed,'donors'))
    typ=b['types'].numpy();payload=b['payload'].numpy();keys=b['active_key'].numpy()
    eligible=np.zeros_like(typ,dtype=bool)
    for t in range(2,typ.shape[1]-3):
        eligible[:,t]=(typ[:,t]==task.DATA)
        for k in range(1,4):eligible[:,t]&=(typ[:,t+k]==task.DATA)
    ids=np.argwhere(eligible)
    if len(ids)>limit:ids=ids[np.sort(rng.choice(len(ids),limit,replace=False))]
    out={}
    for kind in ('different_key','same_key'):
        rows=[]
        for i,t in ids:
            pool=np.flatnonzero((typ[:,t]==typ[i,t])&(payload[:,t]==payload[i,t])&((keys[:,t]!=keys[i,t]) if kind=='different_key' else (keys[:,t]==keys[i,t])))
            pool=pool[pool!=i]
            if len(pool):rows.append((i,t,int(rng.choice(pool))))
        out[kind]=np.asarray(rows,np.int64).reshape(-1,3)
    return out


def influence(model,b,seed,native=None):
    if model.kind=='MONOLITHIC':return None
    if native is None:native=trace(model,b)
    logits,hs=native
    donors=_donors(b,seed)
    result={}
    for kind,rows in donors.items():
        if not len(rows):
            result[kind]={'n':0,'nodes':None,'reason':'no matching donors'}
            continue
        bi=torch.from_numpy(rows[:,0]);ti=torch.from_numpy(rows[:,1]);di=torch.from_numpy(rows[:,2])
        h0=hs[bi,ti];hd=hs[di,ti]
        native3=hs[bi,ti+3];nativep=logits[bi,ti+3].softmax(-1)
        targets=b['target'][bi,ti+3]
        with torch.no_grad():
            enc=model.encode(b['types'],b['payload'])
            future_e=torch.stack([enc[bi,ti+k] for k in range(1,4)],1)
            future_t=torch.stack([b['types'][bi,ti+k] for k in range(1,4)],1)
        nodes=[]
        for node in range(N):
            denom=torch.linalg.vector_norm(hd[:,node]-h0[:,node],dim=-1)
            valid=denom>1e-7
            if not bool(valid.any()):
                nodes.append({'n':0,'response':None,'tv':None,'accuracy_damage':None});continue
            h=h0[valid].clone();h[:,node]=hd[valid,node]
            with torch.no_grad():
                for k in range(3):h=model.step(h,future_e[valid,k],future_t[valid,k])
                pred=model.readout(h).softmax(-1)
            diff=torch.linalg.vector_norm(h-native3[valid],dim=-1)/denom[valid,None]
            tv=0.5*(pred-nativep[valid]).abs().sum(-1)
            nativecorrect=(nativep[valid].argmax(-1)==targets[valid]).float()
            cfcorrect=(pred.argmax(-1)==targets[valid]).float()
            nodes.append({'n':int(valid.sum()),'response':diff.mean(0).tolist(),'tv':float(tv.mean()),'accuracy_damage':float(nativecorrect.mean()-cfcorrect.mean()),'initial_norm_mean':float(denom[valid].mean())})
        result[kind]={'n':int(len(rows)),'nodes':nodes}
    return result


def paired_history_batch(seed,n):
    b=task.make_batch(seed_for(seed,'paired_history'),n)
    q={k:v.clone() for k,v in b.items()}
    rng=np.random.default_rng(seed_for(seed,'paired_history_payload'))
    checkpoints=[]
    for i in range(n):
        acts=(b['types'][i]==task.ACTIVATE).nonzero(as_tuple=False).flatten()
        t=int(acts[-1]);checkpoints.append(t)
        old=(b['types'][i,:t]==task.DATA).nonzero(as_tuple=False).flatten()
        if len(old):
            values=torch.from_numpy(rng.integers(0,task.N_VAL,len(old),dtype=np.int64))
            q['payload'][i,old]=values
            q['target'][i,old]=(values+q['active_key'][i,old])%task.N_VAL
    return b,q,checkpoints


def hysteresis(model,seed,n=128):
    a,b,cp=paired_history_batch(seed,n)
    ya,ha=trace(model,a);yb,hb=trace(model,b)
    aa=ha.reshape(n,ha.shape[1],-1);bb=hb.reshape(n,hb.shape[1],-1)
    rows=[]
    for i,t in enumerate(cp):
        norm=float(torch.linalg.vector_norm(aa[i,t]-bb[i,t])/(0.5*(torch.linalg.vector_norm(aa[i,t])+torch.linalg.vector_norm(bb[i,t]))+1e-8))
        future=(a['target'][i,t:]>=0)
        pa=ya[i,t:][future].softmax(-1);pb=yb[i,t:][future].softmax(-1)
        tv=float((pa-pb).abs().sum(-1).mean()/2)
        target=a['target'][i,t:][future]
        ca=float((pa.argmax(-1)==target).float().mean());cb=float((pb.argmax(-1)==target).float().mean())
        rows.append({'checkpoint_distance':norm,'terminal_distance':float(torch.linalg.vector_norm(aa[i,-1]-bb[i,-1])/(0.5*(torch.linalg.vector_norm(aa[i,-1])+torch.linalg.vector_norm(bb[i,-1]))+1e-8)),'future_tv':tv,'correctness_gap':cb-ca,'future_accuracy_a':ca,'future_accuracy_b':cb})
    return {'n':n,'mean':{k:float(np.mean([r[k] for r in rows])) for k in rows[0]},'rows':rows,'paired_semantics_verified':bool(torch.equal(a['active_key'],b['active_key']) and torch.equal(a['types'],b['types']) and all(torch.equal(a['payload'][i,t:],b['payload'][i,t:]) for i,t in enumerate(cp)))}


def causal_diagnostics(model,b,seed,fits):
    native=trace(model,b)
    return {'communication':communication_controls(model,b,seed,native[0]),'node_ablations':node_ablations(model,b,fits,native),'influence':influence(model,b,seed,native),'hysteresis':hysteresis(model,seed)}
