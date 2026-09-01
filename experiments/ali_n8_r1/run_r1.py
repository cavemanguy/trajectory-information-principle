import argparse, csv, hashlib, json, os, platform, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from run_core import Core, derive_seed, make_memories, make_perms, set_seed

N_REL=8; N_VAL=16; Q_EMB=8; HIDDEN=32; LATENT=16; EPS=1e-8
PRIMARY=(5,17,31)
BATCH=256; LR=1e-3; WD=1e-4; MAX_EPOCHS=100; PATIENCE=12; MIN_DELTA=1e-4; CLIP=1.0
SIZES={'train':20000,'val':2500,'test':5000}

def init_module(m):
    if isinstance(m, nn.Embedding): nn.init.normal_(m.weight,0.0,0.02)
    elif isinstance(m,nn.Linear): nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

def normalize(v): return v/(v.norm(dim=-1,keepdim=True)+EPS)

class QueryEmbed(nn.Module):
    def __init__(self): super().__init__(); self.emb=nn.Embedding(N_REL,Q_EMB); self.apply(init_module)
    def forward(self,q): return self.emb(q)

class MLP(nn.Module):
    def __init__(self,inp,out=N_VAL):
        super().__init__(); self.net=nn.Sequential(nn.Linear(inp,HIDDEN),nn.GELU(),nn.Linear(HIDDEN,out)); self.apply(init_module)
    def forward(self,x): return self.net(x)

class ALISystem(nn.Module):
    def __init__(self,kind):
        super().__init__(); self.kind=kind; self.qe=QueryEmbed(); inp={'mq':LATENT+Q_EMB,'q':Q_EMB,'m':LATENT}[kind]
        self.policy=MLP(inp,LATENT); self.reader=MLP(LATENT+Q_EMB,N_VAL)
    def direction(self,m,q):
        e=self.qe(q); x=torch.cat([m,e],-1) if self.kind=='mq' else e if self.kind=='q' else m
        return normalize(self.policy(x))
    def logits(self,m,q,F,alpha,direction_override=None):
        e=self.qe(q); v=self.direction(m,q) if direction_override is None else direction_override
        r=(F(m+alpha*v)-F(m-alpha*v))/(2*alpha)
        return self.reader(torch.cat([r,e],-1)),v,r

class FixedSystem(nn.Module):
    def __init__(self,learned,seed):
        super().__init__(); self.qe=QueryEmbed(); self.reader=MLP(LATENT+Q_EMB,N_VAL)
        g=torch.Generator().manual_seed(seed); raw=torch.randn(LATENT,generator=g)
        if learned: self.v=nn.Parameter(raw)
        else: self.v=nn.Parameter(normalize(raw.unsqueeze(0)).squeeze(0),requires_grad=False)
    def logits(self,m,q,F,alpha):
        v=normalize(self.v.unsqueeze(0)).expand(m.size(0),-1); r=(F(m+alpha*v)-F(m-alpha*v))/(2*alpha)
        return self.reader(torch.cat([r,self.qe(q)],-1)),v,r

class DirectSystem(nn.Module):
    def __init__(self): super().__init__(); self.qe=QueryEmbed(); self.reader=MLP(LATENT+Q_EMB,N_VAL)
    def logits(self,x,q): return self.reader(torch.cat([x,self.qe(q)],-1))

class ZeroSystem(nn.Module):
    def __init__(self): super().__init__(); self.qe=QueryEmbed(); self.reader=MLP(LATENT+Q_EMB,N_VAL)
    def logits(self,q): return self.reader(torch.cat([torch.zeros(q.size(0),LATENT,device=q.device),self.qe(q)],-1))

def sha_tensor(x): return hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()
def sha_file(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save_json(p,x): Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,default=str))

def load_core(core_path, summary_path):
    core=Core(); ck=torch.load(core_path,map_location='cpu',weights_only=False); core.load_state_dict(ck['state_dict']); core.eval()
    for p in core.parameters(): p.requires_grad=False
    summary=json.loads(Path(summary_path).read_text()); alpha=float(summary['alpha'])
    expected=summary.get('checkpoint_sha256'); actual=sha_file(core_path)
    if expected and expected!=actual: raise RuntimeError(f'core checkpoint hash mismatch: {actual} != {expected}')
    return core,alpha,summary,actual

def split_data(seed,names=('train','val','test')):
    return {n:make_memories(SIZES[n],derive_seed(seed,'memory_'+n)) for n in names}

def encode_split(core,values,seed,name):
    perm_name='perm_train_alpha' if name=='train' else 'perm_'+name
    perms=make_perms(len(values),derive_seed(seed,perm_name)); out=[]
    with torch.no_grad():
        for a in range(0,len(values),BATCH): out.append(core.encode(values[a:a+BATCH],perms[a:a+BATCH]))
    return torch.cat(out),perms

def expand(values,m):
    n=len(values); idx=torch.arange(n).repeat_interleave(N_REL); q=torch.arange(N_REL).repeat(n); y=values.reshape(-1)
    return m[idx],q,y,idx

def F_all(core,m):
    out=[]
    with torch.no_grad():
        for a in range(0,len(m),BATCH): out.append(core.F(m[a:a+BATCH]))
    return torch.cat(out)

def acc_eval(forward,arrays):
    preds=[]; ys=[]
    with torch.no_grad():
        n=len(arrays[-1])
        for a in range(0,n,BATCH):
            b=[x[a:a+BATCH] for x in arrays]; lg=forward(*b[:-1]); preds.append(lg.argmax(1)); ys.append(b[-1])
    p=torch.cat(preds); y=torch.cat(ys); return float((p==y).float().mean()),p,y

def train_model(seed,name,model,tr,va,forward,out):
    set_seed(derive_seed(seed,name+'_init')); model.apply(init_module)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=LR,weight_decay=WD); ce=nn.CrossEntropyLoss()
    best=-1.0; best_ep=0; state=None; stale=0; hist=[]
    for ep in range(1,MAX_EPOCHS+1):
        model.train(); n=len(tr[-1]); g=torch.Generator().manual_seed(derive_seed(seed,f'{name}_batch_{ep}')); order=torch.randperm(n,generator=g)
        total_loss=0.0; seen=0
        for a in range(0,n,BATCH):
            ix=order[a:a+BATCH]; b=[x[ix] for x in tr]; opt.zero_grad(set_to_none=True); lg=forward(model,*b[:-1]); loss=ce(lg,b[-1]); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),CLIP); opt.step(); total_loss += loss.detach().item()*len(ix); seen += len(ix)
        model.eval(); va_acc,_,_=acc_eval(lambda *x:forward(model,*x),va)
        hist.append({'epoch':ep,'train_loss':total_loss/seen,'validation_accuracy':va_acc}); print(f'{name} epoch={ep:03d} val={va_acc:.6f}',flush=True)
        if va_acc>best+MIN_DELTA:
            best=va_acc; best_ep=ep; stale=0; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale += 1
            if stale>=PATIENCE: break
    if state is None: raise RuntimeError(f'no checkpoint selected for {name}')
    model.load_state_dict(state); torch.save({'state_dict':state,'epoch':best_ep,'validation_accuracy':best},out/f'{name}.pt'); save_json(out/f'{name}_history.json',hist)
    return model,{'best_epoch':best_ep,'best_validation_accuracy':best,'parameter_count':sum(p.numel() for p in model.parameters())}

def dirs_for(sysm,m,q):
    out=[]
    with torch.no_grad():
        for a in range(0,len(q),BATCH): out.append(sysm.direction(m[a:a+BATCH],q[a:a+BATCH]))
    return torch.cat(out)

def load_model(path,model):
    ck=torch.load(path,map_location='cpu',weights_only=False); model.load_state_dict(ck['state_dict']); model.eval(); return model

def phase_train(seed,core_path,summary_path,out):
    core,alpha,core_summary,core_hash=load_core(core_path,summary_path); data=split_data(seed,('train','val'))
    train_m,_=encode_split(core,data['train'],seed,'train'); val_m,_=encode_split(core,data['val'],seed,'val')
    tr=expand(data['train'],train_m); va=expand(data['val'],val_m); tr3=tr[:3]; va3=va[:3]
    manifest={'phase':'train_select','seed':seed,'alpha':alpha,'core_checkpoint_sha256':core_hash,'dataset_hashes':{n:sha_tensor(v) for n,v in data.items()},'latent_hashes':{'train':sha_tensor(train_m),'val':sha_tensor(val_m)},'systems':{}}
    systems={}
    for kind in ('mq','q','m'):
        model=ALISystem(kind); fw=lambda mod,m,q: mod.logits(m,q,core.F,alpha)[0]
        model,info=train_model(seed,kind,model,tr3,va3,fw,out); systems[kind]=model; manifest['systems'][kind]=info
    for nm,learned in [('fixed',True),('random',False)]:
        model=FixedSystem(learned,derive_seed(seed,nm+'_vector')); fw=lambda mod,m,q: mod.logits(m,q,core.F,alpha)[0]
        model,info=train_model(seed,nm,model,tr3,va3,fw,out); systems[nm]=model; manifest['systems'][nm]=info
    train_f,val_f=F_all(core,train_m),F_all(core,val_m)
    for nm,trx,vax in [('direct_m',train_m,val_m),('direct_f',train_f,val_f)]:
        trd=expand(data['train'],trx)[:3]; vad=expand(data['val'],vax)[:3]; model=DirectSystem(); fw=lambda mod,x,q:mod.logits(x,q)
        model,info=train_model(seed,nm,model,trd,vad,fw,out); systems[nm]=model; manifest['systems'][nm]=info
    zero=ZeroSystem(); fwz=lambda mod,q:mod.logits(q); zero,info=train_model(seed,'zero',zero,(tr[1],tr[2]),(va[1],va[2]),fwz,out); systems['zero']=zero; manifest['systems']['zero']=info
    for kind in ('mq','m'):
        tv=dirs_for(systems[kind],tr[0],tr[1]); vv=dirs_for(systems[kind],va[0],va[1]); leak=DirectSystem(); fw=lambda mod,x,q:mod.logits(x,q)
        leak,info=train_model(seed,'leak_'+kind,leak,(tv,tr[1],tr[2]),(vv,va[1],va[2]),fw,out); manifest['systems']['leak_'+kind]=info
    save_json(out/'train_manifest.json',manifest)
    print(json.dumps(manifest,indent=2),flush=True)

def response_by_direction(core,alpha,m,v):
    out=[]
    with torch.no_grad():
        for a in range(0,len(m),BATCH):
            mm=m[a:a+BATCH]; vv=v.unsqueeze(0).expand(len(mm),-1); out.append((core.F(mm+alpha*vv)-core.F(mm-alpha*vv))/(2*alpha))
    return torch.cat(out)

def phase_decode(seed,direction,core_path,summary_path,main_dir,out):
    core,alpha,_,core_hash=load_core(core_path,summary_path); data=split_data(seed,('train','val')); tm,_=encode_split(core,data['train'],seed,'train'); vm,_=encode_split(core,data['val'],seed,'val')
    qsys=load_model(Path(main_dir)/'q.pt',ALISystem('q'))
    with torch.no_grad(): qdirs=qsys.direction(torch.zeros(N_REL,LATENT),torch.arange(N_REL))
    rt=response_by_direction(core,alpha,tm,qdirs[direction]); rv=response_by_direction(core,alpha,vm,qdirs[direction])
    manifest={'phase':'decode_train_select','seed':seed,'direction':direction,'core_checkpoint_sha256':core_hash,'query_system_sha256':sha_file(Path(main_dir)/'q.pt'),'decoders':{}}
    for target in range(N_REL):
        name=f'decode_{target}_{direction}'; dec=MLP(LATENT,N_VAL); fw=lambda mod,x:mod(x)
        dec,info=train_model(seed,name,dec,(rt,data['train'][:,target]),(rv,data['val'][:,target]),fw,out); manifest['decoders'][str(target)]=info
    save_json(out/f'decode_direction_{direction}_manifest.json',manifest); print(json.dumps(manifest,indent=2),flush=True)

def save_preds(path,p,y,q=None,mem=None):
    with Path(path).open('w',newline='') as f:
        w=csv.writer(f); hdr=['target','prediction']
        if q is not None: hdr.append('query')
        if mem is not None: hdr.append('memory_index')
        w.writerow(hdr)
        for i in range(len(y)):
            row=[int(y[i]),int(p[i])]
            if q is not None: row.append(int(q[i]))
            if mem is not None: row.append(int(mem[i]))
            w.writerow(row)

def phase_final(seed,core_path,summary_path,main_dir,decoders_dir,out):
    core,alpha,core_summary,core_hash=load_core(core_path,summary_path)
    required=['mq','q','m','fixed','random','direct_m','direct_f','zero','leak_mq','leak_m']
    for n in required:
        if not (Path(main_dir)/f'{n}.pt').exists(): raise FileNotFoundError(f'missing selected checkpoint {n}.pt')
    for j in range(N_REL):
        for i in range(N_REL):
            if not (Path(decoders_dir)/f'decode_{i}_{j}.pt').exists(): raise FileNotFoundError(f'missing decode_{i}_{j}.pt')
    data=split_data(seed,('test',)); test=data['test']; test_m,test_perm=encode_split(core,test,seed,'test'); te=expand(test,test_m)
    systems={
        'mq':load_model(Path(main_dir)/'mq.pt',ALISystem('mq')),
        'q':load_model(Path(main_dir)/'q.pt',ALISystem('q')),
        'm':load_model(Path(main_dir)/'m.pt',ALISystem('m')),
        'fixed':load_model(Path(main_dir)/'fixed.pt',FixedSystem(True,derive_seed(seed,'fixed_vector'))),
        'random':load_model(Path(main_dir)/'random.pt',FixedSystem(False,derive_seed(seed,'random_vector'))),
        'direct_m':load_model(Path(main_dir)/'direct_m.pt',DirectSystem()),
        'direct_f':load_model(Path(main_dir)/'direct_f.pt',DirectSystem()),
        'zero':load_model(Path(main_dir)/'zero.pt',ZeroSystem()),
        'leak_mq':load_model(Path(main_dir)/'leak_mq.pt',DirectSystem()),
        'leak_m':load_model(Path(main_dir)/'leak_m.pt',DirectSystem()),
    }
    result={'experiment':'ALI-N8-R1','seed':seed,'phase':'final_test','alpha':alpha,'core_checkpoint_sha256':core_hash,'core_summary':core_summary,'test_dataset_sha256':sha_tensor(test),'test_latent_sha256':sha_tensor(test_m),'systems':{}}
    for name in ('mq','q','m'):
        fw=lambda m,q,s=systems[name]:s.logits(m,q,core.F,alpha)[0]; acc,p,y=acc_eval(fw,te[:3]); save_preds(out/f'{name}_predictions.csv',p,y,te[1],te[3]); result['systems'][name]={'test_accuracy':acc}
    for name in ('fixed','random'):
        fw=lambda m,q,s=systems[name]:s.logits(m,q,core.F,alpha)[0]; acc,p,y=acc_eval(fw,te[:3]); save_preds(out/f'{name}_predictions.csv',p,y,te[1],te[3]); result['systems'][name]={'test_accuracy':acc}
    test_f=F_all(core,test_m)
    for name,x in [('direct_m',test_m),('direct_f',test_f)]:
        ted=expand(test,x); fw=lambda xx,q,s=systems[name]:s.logits(xx,q); acc,p,y=acc_eval(fw,ted[:3]); save_preds(out/f'{name}_predictions.csv',p,y,ted[1],ted[3]); result['systems'][name]={'test_accuracy':acc}
    acc,p,y=acc_eval(lambda q:systems['zero'].logits(q),(te[1],te[2])); save_preds(out/'zero_predictions.csv',p,y,te[1],te[3]); result['systems']['zero']={'test_accuracy':acc}
    for kind in ('mq','m'):
        vv=dirs_for(systems[kind],te[0],te[1]); leak=systems['leak_'+kind]; acc,p,y=acc_eval(lambda x,q:leak.logits(x,q),(vv,te[1],te[2])); save_preds(out/f'leak_{kind}_predictions.csv',p,y,te[1],te[3]); result['systems']['leak_'+kind]={'test_accuracy':acc}
    donor=torch.roll(torch.arange(len(test_m)),1); wrong_v=[]; native_v=[]
    with torch.no_grad():
        for a in range(0,len(te[2]),BATCH):
            mi=te[3][a:a+BATCH]; q=te[1][a:a+BATCH]; native_v.append(systems['mq'].direction(te[0][a:a+BATCH],q)); wrong_v.append(systems['mq'].direction(test_m[donor[mi]],q))
    native_v=torch.cat(native_v); wrong_v=torch.cat(wrong_v)
    def override_preds(vs):
        pp=[]
        with torch.no_grad():
            for a in range(0,len(te[2]),BATCH): pp.append(systems['mq'].logits(te[0][a:a+BATCH],te[1][a:a+BATCH],core.F,alpha,vs[a:a+BATCH])[0].argmax(1))
        return torch.cat(pp)
    pn,pw=override_preds(native_v),override_preds(wrong_v); an=float((pn==te[2]).float().mean()); aw=float((pw==te[2]).float().mean())
    result['wrong_memory']={'native_accuracy':an,'wrong_memory_accuracy':aw,'prediction_change_rate':float((pn!=pw).float().mean()),'paired_accuracy_difference':an-aw}
    with torch.no_grad(): qdirs=systems['q'].direction(torch.zeros(N_REL,LATENT),torch.arange(N_REL)); cosine=qdirs@qdirs.T
    np.savetxt(out/'query_direction_cosine.csv',cosine.numpy(),delimiter=',')
    native=np.zeros((N_REL,N_REL)); native_counts=np.zeros((N_REL,N_REL),dtype=np.int64)
    for i in range(N_REL):
        mask=te[1]==i; mm,qq,yy=te[0][mask],te[1][mask],te[2][mask]
        for j in range(N_REL):
            pp=[]
            with torch.no_grad():
                for a in range(0,len(yy),BATCH):
                    v=qdirs[j].unsqueeze(0).expand(min(BATCH,len(yy)-a),-1); pp.append(systems['q'].logits(mm[a:a+BATCH],qq[a:a+BATCH],core.F,alpha,v)[0].argmax(1))
            p=torch.cat(pp); native_counts[i,j]=int((p==yy).sum()); native[i,j]=native_counts[i,j]/len(yy)
    np.savetxt(out/'native_swap_accuracy.csv',native,delimiter=','); np.savetxt(out/'native_swap_counts.csv',native_counts,delimiter=',',fmt='%d')
    per_native=[float(native[i,i]-np.mean(np.delete(native[i],i))) for i in range(N_REL)]; result['native_diagonal_advantage_per_relation']=per_native; result['D_native']=float(np.mean(per_native))
    decode=np.zeros((N_REL,N_REL)); decode_counts=np.zeros((N_REL,N_REL),dtype=np.int64)
    for j in range(N_REL):
        re=response_by_direction(core,alpha,test_m,qdirs[j])
        for i in range(N_REL):
            dec=load_model(Path(decoders_dir)/f'decode_{i}_{j}.pt',MLP(LATENT,N_VAL)); acc,p,y=acc_eval(lambda x,d=dec:d(x),(re,test[:,i])); decode[i,j]=acc; decode_counts[i,j]=int((p==y).sum())
    np.savetxt(out/'decode_accuracy.csv',decode,delimiter=','); np.savetxt(out/'decode_counts.csv',decode_counts,delimiter=',',fmt='%d')
    per_decode=[float(decode[i,i]-np.mean(np.delete(decode[i],i))) for i in range(N_REL)]; result['decode_diagonal_advantage_per_relation']=per_decode; result['D_decode']=float(np.mean(per_decode))
    correct_m=np.zeros(N_REL,dtype=np.int64); correct_z=np.zeros(N_REL,dtype=np.int64)
    with torch.no_grad():
        for a in range(0,len(test),BATCH):
            y=test[a:a+BATCH]; m,z,lm,lz=core(y,test_perm[a:a+BATCH])
            for i in range(N_REL): correct_m[i]+=int((lm[i].argmax(1)==y[:,i]).sum()); correct_z[i]+=int((lz[i].argmax(1)==y[:,i]).sum())
    result['core_integrity_test']={'m_head_accuracies':(correct_m/len(test)).tolist(),'F_head_accuracies':(correct_z/len(test)).tolist(),'m_mean':float((correct_m/len(test)).mean()),'F_mean':float((correct_z/len(test)).mean())}
    result['checkpoint_hashes']={n:sha_file(Path(main_dir)/f'{n}.pt') for n in required}; result['decoder_checkpoint_hashes']={f'{i}_{j}':sha_file(Path(decoders_dir)/f'decode_{i}_{j}.pt') for j in range(N_REL) for i in range(N_REL)}
    save_json(out/'final_summary.json',result); print(json.dumps(result,indent=2),flush=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--phase',choices=['train','decode','final'],required=True); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--core-path',required=True); ap.add_argument('--core-summary',required=True); ap.add_argument('--main-dir'); ap.add_argument('--decoders-dir'); ap.add_argument('--direction',type=int); ap.add_argument('--outdir',required=True); args=ap.parse_args()
    if args.seed not in PRIMARY: raise SystemExit('R1 only accepts primary seeds 5, 17, 31')
    if args.phase=='decode' and (args.direction is None or not 0<=args.direction<N_REL): raise SystemExit('--direction 0..7 required for decode')
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    env={'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),'phase':args.phase,'seed':args.seed,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}; save_json(out/'environment.json',env)
    if args.phase=='train': phase_train(args.seed,args.core_path,args.core_summary,out)
    elif args.phase=='decode': phase_decode(args.seed,args.direction,args.core_path,args.core_summary,args.main_dir,out)
    else: phase_final(args.seed,args.core_path,args.core_summary,args.main_dir,args.decoders_dir,out)

if __name__=='__main__': main()
