import argparse, csv, hashlib, json, platform, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

N_REL=8; N_VAL=16; EMB=8; HIDDEN=32; LATENT=16; STEPS=12; OBS=8; EPS=1e-8
PRIMARY=(7,19,43)
SIZES={'train':20000,'val':2500,'test':5000}
BATCH=256; LR=1e-3; WD=1e-4; MAX_EPOCHS=100; PATIENCE=12; MIN_DELTA=1e-4; CLIP=1.0


def derive_seed(seed,name):
    h=hashlib.sha256(f'Observer-R2|{seed}|{name}'.encode()).digest()
    return int.from_bytes(h[:4],'big')

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True,warn_only=True)

def init_linear_embedding(m):
    if isinstance(m,nn.Embedding): nn.init.normal_(m.weight,0.0,0.02)
    elif isinstance(m,nn.Linear): nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

def init_gru(gru):
    for name,p in gru.named_parameters():
        if 'weight_ih' in name: nn.init.xavier_uniform_(p)
        elif 'weight_hh' in name:
            for b in p.chunk(3,0): nn.init.orthogonal_(b)
        elif 'bias' in name: nn.init.zeros_(p)

def save_json(p,x): Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,default=str))
def sha_file(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sha_tensor(x): return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def make_memories(n,seed):
    g=torch.Generator().manual_seed(seed)
    return torch.randint(0,N_VAL,(n,N_REL),generator=g)

def make_perms(n,seed):
    g=torch.Generator().manual_seed(seed)
    return torch.stack([torch.randperm(N_REL,generator=g) for _ in range(n)])

def make_time_perms(n,seed):
    g=torch.Generator().manual_seed(seed)
    return torch.stack([torch.randperm(STEPS,generator=g) for _ in range(n)])

class Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.rel_emb=nn.Embedding(N_REL,EMB); self.val_emb=nn.Embedding(N_VAL,EMB)
        self.enc=nn.GRU(EMB*2,HIDDEN,batch_first=True); self.to_h=nn.Linear(HIDDEN,LATENT)
        self.F=nn.Sequential(nn.Linear(LATENT,HIDDEN),nn.GELU(),nn.Linear(HIDDEN,LATENT),nn.Tanh())
        self.head0=nn.ModuleList([nn.Linear(LATENT,N_VAL) for _ in range(N_REL)])
        self.headT=nn.ModuleList([nn.Linear(LATENT,N_VAL) for _ in range(N_REL)])
        self.apply(init_linear_embedding); init_gru(self.enc)
    def encode(self,y,perms):
        b=len(y); rel=torch.arange(N_REL).expand(b,-1).gather(1,perms); val=y.gather(1,perms)
        x=torch.cat([self.rel_emb(rel),self.val_emb(val)],-1); _,s=self.enc(x)
        return torch.tanh(self.to_h(s[-1]))
    def trajectory(self,h0):
        hs=[h0]; h=h0
        for _ in range(STEPS): h=self.F(h); hs.append(h)
        return torch.stack(hs,1)
    def forward(self,y,perms):
        h0=self.encode(y,perms); tr=self.trajectory(h0); hT=tr[:,-1]
        return tr,[h(h0) for h in self.head0],[h(hT) for h in self.headT]

class Heads(nn.Module):
    def __init__(self,inp):
        super().__init__(); self.heads=nn.ModuleList([nn.Sequential(nn.Linear(inp,32),nn.GELU(),nn.Linear(32,N_VAL)) for _ in range(N_REL)]); self.apply(init_linear_embedding)
    def forward(self,x): return [h(x) for h in self.heads]

class ObserverSystem(nn.Module):
    def __init__(self,input_dim,mode='native'):
        super().__init__(); self.mode=mode; self.cell=nn.GRUCell(input_dim,OBS); self.readout=Heads(OBS)
        self.apply(init_linear_embedding); init_gru(self.cell)
    def forward(self,seq):
        o=torch.zeros(len(seq),OBS,device=seq.device)
        for t in range(seq.size(1)):
            prev=torch.zeros_like(o) if self.mode=='reset' else o
            o=self.cell(seq[:,t],prev)
        return self.readout(o),o

class SnapshotSystem(nn.Module):
    def __init__(self,inp): super().__init__(); self.readout=Heads(inp)
    def forward(self,x): return self.readout(x)


def geometry(tr):
    h=tr[:,:-1]; hn=tr[:,1:]; dh=hn-h
    speed=torch.linalg.vector_norm(dh,dim=-1,keepdim=True)
    direction=dh/(speed+EPS)
    radius=torch.linalg.vector_norm(h,dim=-1,keepdim=True)
    radial=torch.linalg.vector_norm(hn,dim=-1,keepdim=True)-radius
    g=torch.cat([direction,speed,radius,radial],-1)
    turn=torch.zeros(len(tr),STEPS,1)
    turn[:,1:,0]=torch.nn.functional.cosine_similarity(dh[:,:-1],dh[:,1:],dim=-1)
    return {'g':g,'direction':direction,'speed':speed,'radius':radius,'radial_change':radial,'turn':turn,'dh':dh}

def permute_seq(seq,perms):
    return seq.gather(1,perms.unsqueeze(-1).expand(-1,-1,seq.size(-1)))

def eval_logits(logits,y):
    preds=torch.stack([z.argmax(1) for z in logits],1); corr=(preds==y)
    return float(corr.float().mean()),corr.float().mean(0).tolist(),preds

def evaluate_model(model,kind,arrays):
    model.eval(); ps=[]; ys=[]
    with torch.no_grad():
        n=len(arrays['y'])
        for a in range(0,n,BATCH):
            y=arrays['y'][a:a+BATCH]
            if kind in ('observer','reset','shuffled','direction','speed','state'):
                lg,_=model(arrays['x'][a:a+BATCH])
            else: lg=model(arrays['x'][a:a+BATCH])
            ps.append(torch.stack([z.argmax(1) for z in lg],1)); ys.append(y)
    p=torch.cat(ps); y=torch.cat(ys); c=(p==y)
    return float(c.float().mean()),c.float().mean(0).tolist(),p,y

def train_system(seed,name,model,kind,tr,va,out):
    set_seed(derive_seed(seed,name+'_init')); model.apply(init_linear_embedding)
    if hasattr(model,'cell'): init_gru(model.cell)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD); ce=nn.CrossEntropyLoss()
    best=-1.; state=None; best_ep=0; stale=0; hist=[]
    for ep in range(1,MAX_EPOCHS+1):
        model.train(); n=len(tr['y']); g=torch.Generator().manual_seed(derive_seed(seed,f'{name}_batch_{ep}')); order=torch.randperm(n,generator=g)
        total=0.; seen=0
        for a in range(0,n,BATCH):
            ix=order[a:a+BATCH]; x=tr['x'][ix]; y=tr['y'][ix]; opt.zero_grad(set_to_none=True)
            if kind in ('observer','reset','shuffled','direction','speed','state'): lg,_=model(x)
            else: lg=model(x)
            loss=sum(ce(lg[i],y[:,i]) for i in range(N_REL))/N_REL; loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),CLIP); opt.step()
            total+=loss.item()*len(ix); seen+=len(ix)
        va_acc,_,_,_=evaluate_model(model,kind,va); hist.append({'epoch':ep,'train_loss':total/seen,'validation_accuracy':va_acc}); print(f'{name} epoch={ep:03d} val={va_acc:.6f}',flush=True)
        if va_acc>best+MIN_DELTA:
            best=va_acc; best_ep=ep; stale=0; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale+=1
            if stale>=PATIENCE: break
    if state is None: raise RuntimeError(name)
    model.load_state_dict(state); torch.save({'state_dict':state,'epoch':best_ep,'validation_accuracy':best},out/f'{name}.pt'); save_json(out/f'{name}_history.json',hist)
    return {'best_epoch':best_ep,'best_validation_accuracy':best,'parameter_count':sum(p.numel() for p in model.parameters())}

def evaluate_core(model,y,perms):
    model.eval(); c0=np.zeros(N_REL,dtype=np.int64); cT=np.zeros(N_REL,dtype=np.int64)
    with torch.no_grad():
        for a in range(0,len(y),BATCH):
            yy=y[a:a+BATCH]; tr,l0,lT=model(yy,perms[a:a+BATCH])
            for i in range(N_REL): c0[i]+=int((l0[i].argmax(1)==yy[:,i]).sum()); cT[i]+=int((lT[i].argmax(1)==yy[:,i]).sum())
    a0=c0/len(y); aT=cT/len(y); return float(np.r_[a0,aT].mean()),a0.tolist(),aT.tolist()

def phase_core(seed,out):
    train=make_memories(SIZES['train'],derive_seed(seed,'memory_train')); val=make_memories(SIZES['val'],derive_seed(seed,'memory_val')); vp=make_perms(len(val),derive_seed(seed,'perm_val'))
    set_seed(derive_seed(seed,'core_init')); model=Core(); opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD); ce=nn.CrossEntropyLoss()
    best=-1.; stale=0; best_ep=0; state=None; hist=[]
    for ep in range(1,MAX_EPOCHS+1):
        model.train(); tp=make_perms(len(train),derive_seed(seed,f'perm_train_{ep}')); g=torch.Generator().manual_seed(derive_seed(seed,f'core_batch_{ep}')); order=torch.randperm(len(train),generator=g)
        for a in range(0,len(train),BATCH):
            ix=order[a:a+BATCH]; y=train[ix]; p=tp[ix]; opt.zero_grad(set_to_none=True); _,l0,lT=model(y,p); loss=sum(ce(l0[i],y[:,i])+ce(lT[i],y[:,i]) for i in range(N_REL))/N_REL; loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),CLIP); opt.step()
        metric,a0,aT=evaluate_core(model,val,vp); hist.append({'epoch':ep,'validation_metric':metric,'h0':a0,'h12':aT}); print(f'core epoch={ep:03d} val={metric:.6f}',flush=True)
        if metric>best+MIN_DELTA:
            best=metric; best_ep=ep; stale=0; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; torch.save({'state_dict':state,'epoch':best_ep,'validation_metric':best},out/'core_best.pt')
        else:
            stale+=1
            if stale>=PATIENCE: break
    save_json(out/'core_history.json',hist); ck=sha_file(out/'core_best.pt'); save_json(out/'core_summary.json',{'seed':seed,'best_epoch':best_ep,'best_validation_metric':best,'checkpoint_sha256':ck,'test_generated':False,'test_evaluated':False,'status':'core_complete_frozen_test_unseen'})

def load_core(path,summary_path):
    s=json.loads(Path(summary_path).read_text()); actual=sha_file(path)
    if s['checkpoint_sha256']!=actual: raise RuntimeError('core hash mismatch')
    m=Core(); ck=torch.load(path,map_location='cpu',weights_only=False); m.load_state_dict(ck['state_dict']); m.eval()
    for p in m.parameters(): p.requires_grad=False
    return m,s

def make_split_features(core,seed,name):
    y=make_memories(SIZES[name],derive_seed(seed,'memory_'+name)); p=make_perms(len(y),derive_seed(seed,'perm_'+name)); chunks=[]
    with torch.no_grad():
        for a in range(0,len(y),BATCH): chunks.append(core.trajectory(core.encode(y[a:a+BATCH],p[a:a+BATCH])))
    tr=torch.cat(chunks); geo=geometry(tr); tperm=make_time_perms(len(y),derive_seed(seed,'time_shuffle_'+name)); return y,p,tr,geo,tperm

def build_arrays(tr,geo,tperm,y,system):
    if system=='observer': x=geo['g']
    elif system=='snapshot': x=geo['g'][:,-1]
    elif system=='reset': x=geo['g']
    elif system=='shuffled': x=permute_seq(geo['g'],tperm)
    elif system=='direction': x=geo['direction']
    elif system=='speed': x=geo['speed']
    elif system=='state': x=torch.cat([geo['g'],tr[:,:-1]],-1)
    elif system=='direct_h0': x=tr[:,0]
    elif system=='direct_h12': x=tr[:,-1]
    else: raise ValueError(system)
    return {'x':x,'y':y}

def system_ctor(name):
    if name=='observer': return ObserverSystem(19,'native'),'observer'
    if name=='snapshot': return SnapshotSystem(19),'snapshot'
    if name=='reset': return ObserverSystem(19,'reset'),'reset'
    if name=='shuffled': return ObserverSystem(19,'native'),'shuffled'
    if name=='direction': return ObserverSystem(16,'native'),'direction'
    if name=='speed': return ObserverSystem(1,'native'),'speed'
    if name=='state': return ObserverSystem(35,'native'),'state'
    if name in ('direct_h0','direct_h12'): return SnapshotSystem(16),name
    raise ValueError(name)

def phase_train(seed,core_path,summary_path,out):
    core,s=load_core(core_path,summary_path); ty,tp,ttr,tg,tperm=make_split_features(core,seed,'train'); vy,vp,vtr,vg,vperm=make_split_features(core,seed,'val')
    manifest={'seed':seed,'core_checkpoint_sha256':s['checkpoint_sha256'],'test_generated':False,'test_evaluated':False,'systems':{},'train_dataset_sha256':sha_tensor(ty),'val_dataset_sha256':sha_tensor(vy),'train_trajectory_sha256':sha_tensor(ttr),'val_trajectory_sha256':sha_tensor(vtr)}
    for name in ('observer','snapshot','reset','shuffled','direction','speed','state','direct_h0','direct_h12'):
        model,kind=system_ctor(name); info=train_system(seed,name,model,kind,build_arrays(ttr,tg,tperm,ty,name),build_arrays(vtr,vg,vperm,vy,name),out); manifest['systems'][name]=info
    save_json(out/'train_manifest.json',manifest)

def load_system(path,name):
    model,kind=system_ctor(name); ck=torch.load(path,map_location='cpu',weights_only=False); model.load_state_dict(ck['state_dict']); model.eval(); return model,kind

def save_predictions(path,p,y):
    with Path(path).open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['memory_index','relation','target','prediction'])
        for m in range(len(y)):
            for r in range(N_REL): w.writerow([m,r,int(y[m,r]),int(p[m,r])])

def bootstrap_diff(seed,name,pA,pB,y,nboot=10000):
    a=(pA==y).float().mean(1).numpy(); b=(pB==y).float().mean(1).numpy(); d=a-b; rng=np.random.default_rng(derive_seed(seed,'bootstrap_'+name)); vals=np.empty(nboot)
    n=len(d)
    for i in range(nboot): vals[i]=d[rng.integers(0,n,n)].mean()
    return {'observed':float(d.mean()),'ci95_percentile':[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))],'n_bootstrap':nboot}

def phase_final(seed,core_path,summary_path,main_dir,out):
    core,s=load_core(core_path,summary_path); names=('observer','snapshot','reset','shuffled','direction','speed','state','direct_h0','direct_h12')
    for n in names:
        if not (Path(main_dir)/f'{n}.pt').exists(): raise FileNotFoundError(n)
    y,p,tr,geo,tperm=make_split_features(core,seed,'test'); result={'experiment':'Observer-R2','seed':seed,'core_checkpoint_sha256':s['checkpoint_sha256'],'test_dataset_sha256':sha_tensor(y),'test_trajectory_sha256':sha_tensor(tr),'systems':{},'checkpoint_hashes':{}}
    preds={}
    for n in names:
        model,kind=load_system(Path(main_dir)/f'{n}.pt',n); acc,per,pred,_=evaluate_model(model,kind,build_arrays(tr,geo,tperm,y,n)); result['systems'][n]={'test_accuracy':acc,'per_relation_accuracy':per}; preds[n]=pred; result['checkpoint_hashes'][n]=sha_file(Path(main_dir)/f'{n}.pt'); save_predictions(out/f'{n}_predictions.csv',pred,y)
    result['paired']={
        'history_vs_snapshot':bootstrap_diff(seed,'history_vs_snapshot',preds['observer'],preds['snapshot'],y),
        'history_vs_reset':bootstrap_diff(seed,'history_vs_reset',preds['observer'],preds['reset'],y),
        'history_vs_shuffled':bootstrap_diff(seed,'history_vs_shuffled',preds['observer'],preds['shuffled'],y),
    }
    metric,a0,aT=evaluate_core(core,y,p); result['core_integrity_test']={'mean_16_heads':metric,'h0_head_accuracies':a0,'h12_head_accuracies':aT}
    k=min(512,len(y)); np.savez_compressed(out/'trajectory_log_first512.npz',h=tr[:k].numpy(),dh=geo['dh'][:k].numpy(),direction=geo['direction'][:k].numpy(),speed=geo['speed'][:k].numpy(),radius=geo['radius'][:k].numpy(),radial_change=geo['radial_change'][:k].numpy(),turn=geo['turn'][:k].numpy())
    model,_=load_system(Path(main_dir)/'observer.pt','observer'); states=[]; o=torch.zeros(k,OBS)
    with torch.no_grad():
        states.append(o.clone())
        for t in range(STEPS): o=model.cell(geo['g'][:k,t],o); states.append(o.clone())
    np.save(out/'observer_states_first512.npy',torch.stack(states,1).numpy())
    save_json(out/'final_summary.json',result); print(json.dumps(result,indent=2),flush=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--phase',choices=['core','train','final'],required=True); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--outdir',required=True); ap.add_argument('--core-path'); ap.add_argument('--core-summary'); ap.add_argument('--main-dir'); args=ap.parse_args()
    if args.seed not in PRIMARY: raise SystemExit('Observer-R2 primary runner accepts only seeds 7, 19, 43')
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True); save_json(out/'environment.json',{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),'phase':args.phase,'seed':args.seed,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    if args.phase=='core': phase_core(args.seed,out)
    elif args.phase=='train': phase_train(args.seed,args.core_path,args.core_summary,out)
    else: phase_final(args.seed,args.core_path,args.core_summary,args.main_dir,out)

if __name__=='__main__': main()
