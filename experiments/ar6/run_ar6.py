import os,sys,math,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.linear_model import Ridge
os.environ['OMP_NUM_THREADS']='1';torch.set_num_threads(4)
sys.path.insert(0,'/mnt/data/ar1');import run_core as ar
ROOT=Path('/mnt/data/ar6');RES=ROOT/'results';RES.mkdir(parents=True,exist_ok=True);CK=Path('/mnt/data/ar2/checkpoints')
SEEDS=[7,19,31,43,59];PROG=np.linspace(0,.9,10);TOL=.002524

def rs(*x):return int.from_bytes(hashlib.sha256('|'.join(map(str,x)).encode()).digest()[:4],'big')
def load(seed):ar.CK=CK;return ar.load(seed)[0]
def split(m,seed,name,n):
 X,C,Y,ids=ar.make_split(seed,name,n);ch,_=ar.choose_conv(m,seed);z=m.encode(X).detach();_,_,tau,done,pred,states=ar.converge(m,X,ch['eps'],ch['K'],save_all=True);return X,C.numpy(),Y.numpy(),z,tau.numpy(),done.numpy(),pred.numpy(),states.detach()
def sample_states(st,tau):
 a=st.numpy();N,T,D=a.shape;sp=np.empty((N,10,D),np.float32)
 for j,p in enumerate(PROG):
  x=p*tau;lo=np.floor(x).astype(int);hi=np.minimum(np.ceil(x).astype(int),T-1);w=x-lo;sp[:,j]=(1-w[:,None])*a[np.arange(N),lo]+w[:,None]*a[np.arange(N),hi]
 d=sp[:,1:]-sp[:,:-1];u=d/(np.linalg.norm(d,axis=2,keepdims=True)+1e-8);return sp,u

def fit_K(U,Y,p):
 Ks=[];As=[]
 for y in range(16):
  ix=Y==y;mod=Ridge(alpha=1e-3).fit(U[ix,p],U[ix,p+1]);A=mod.coef_.astype(np.float32);K=A-np.eye(A.shape[0],dtype=np.float32)
  eff=np.mean(np.linalg.norm(U[ix,p]@K.T,axis=1)/(np.linalg.norm(U[ix,p],axis=1)+1e-8));K=K/max(eff,1e-6);Ks.append(K);As.append(A)
 return np.stack(Ks),np.stack(As)
def jac(m,s,z):
 s=s.detach().requires_grad_(True);return torch.autograd.functional.jacobian(lambda q:m.step(q[None],z[None])[0],s,vectorize=True).detach().numpy()
def stepmod(m,s,z,M):
 g=m.step(s,z);d=g-s;return s+torch.einsum('bij,bj->bi',M,d)
def centroid(U,Y,C,p):
 ce=np.zeros((16,4,U.shape[-1]),np.float32)
 for y in range(16):
  for c in range(4):ce[y,c]=U[(Y==y)&(C==c),p].mean(0)
 return ce
def reader_acc(X,Y,C,ce):
 pr=np.zeros(len(X),int)
 for y in range(16):
  ii=np.where(Y==y)[0];pr[ii]=((X[ii,None]-ce[y][None])**2).sum(2).argmin(1)
 return float((pr==C).mean())
def calibrate(seed,m,tr,val,Utr):
 rows=[]
 for p in [3,4,5]:
  K,_=fit_K(Utr,tr[2],p);N=min(512,len(val[1]));Y=val[2][:N];z=val[3][:N];tau=val[4][:N];st=val[7][:N];tt=np.maximum(1,np.floor(tau*PROG[p+1]).astype(int));base=st[np.arange(N),tt].clone();yn=val[6][:N]
  for alpha in [.005,.01,.02,.05,.1]:
   Ms=torch.tensor(np.eye(base.shape[1])[None]+alpha*K[Y],dtype=torch.float32);s=base.clone()
   with torch.no_grad():
    for k in range(2):s=stepmod(m,s,z,Ms)
    for k in range(60):s=m.step(s,z)
    y2=m.nearest_logits(s).argmax(1).numpy();native=base.clone()
    for k in range(62):native=m.step(native,z)
    ep=torch.linalg.vector_norm(s-native,dim=1).numpy()
   jc=[]
   for i in range(min(32,N)):
    J=jac(m,base[i],z[i]);M=np.eye(J.shape[0])+alpha*K[Y[i]];Jp=np.eye(J.shape[0])+M@(J-np.eye(J.shape[0]));jc.append(np.linalg.norm(Jp-J,'fro')/(np.linalg.norm(J,'fro')+1e-12))
   rows.append(dict(seed=seed,p=p,alpha=alpha,J_change=np.mean(jc),Y_same=np.mean(y2==yn),convergence=1.0,endpoint_pass=np.mean(ep<TOL),endpoint_mean=np.mean(ep)))
 df=pd.DataFrame(rows);df.to_csv(RES/f'seed{seed}_calibration.csv',index=False)
 ok=df[(df.J_change>=.0036)&(df.Y_same>=.99)&(df.convergence>=.99)&(df.endpoint_pass>=.99)]
 if not len(ok):return None,df
 mx=ok.alpha.max();q=ok[ok.alpha==mx].copy();q['md']=(q.p-4).abs();r=q.sort_values(['md','p']).iloc[0];return (int(r.p),float(r.alpha),float(r.J_change)),df

def analyze(seed):
 print('seed',seed,flush=True);m=load(seed);m.eval();tr=split(m,seed,'train',ar.train_n);val=split(m,seed,'val',ar.val_n);sptr,Utr=sample_states(tr[7],tr[4]);sel,cal=calibrate(seed,m,tr,val,Utr)
 if sel is None:return dict(seed=seed,status='NO_SAFE_OPERATOR_WINDOW')
 p,alpha,jchg=sel;K,A=fit_K(Utr,tr[2],p);te=split(m,seed,'test',ar.test_n);sp0,U0=sample_states(te[7],te[4]);N=min(1024,len(te[1]));Y=te[2][:N];C=te[1][:N];z=te[3][:N];tau=te[4][:N];st=te[7][:N];D=st.shape[-1]
 tt=np.maximum(1,np.floor(tau*PROG[p+1]).astype(int));conds={}
 rng=np.random.default_rng(rs('randK',seed));Kr=rng.normal(size=K.shape).astype(np.float32)
 for y in range(16):
  Kr[y]/=(np.linalg.norm(Kr[y],'fro')+1e-8);Kr[y]*=np.linalg.norm(K[y],'fro')
 for name,sgn,KK in [('advance',1,K),('retard',-1,K),('random',1,Kr),('zero',0,K)]:
  s=st[np.arange(N),tt].clone();Ms=torch.tensor(np.eye(D)[None]+sgn*alpha*KK[Y],dtype=torch.float32);seq=[s.clone()]
  with torch.no_grad():
   for k in range(2):s=stepmod(m,s,z,Ms);seq.append(s.clone())
   for k in range(58):s=m.step(s,z);seq.append(s.clone())
  conds[name]=(torch.stack(seq,1),s)
 native=conds['zero'][0];rows=[]
 for name in ['advance','retard','random']:
  arr=conds[name][0]
  for h in [2,4,8,16,32]:
   if h>=arr.shape[1]:continue
   dn=native[:,h]-native[:,h-1];dm=arr[:,h]-arr[:,h-1];un=dn/(torch.linalg.vector_norm(dn,dim=1,keepdim=True)+1e-8);um=dm/(torch.linalg.vector_norm(dm,dim=1,keepdim=True)+1e-8);du=um-un
   rows.append(dict(seed=seed,condition=name,h=h,mean_du=float(torch.linalg.vector_norm(du,dim=1).mean()),mean_cos=float((un*um).sum(1).mean())))
 pd.DataFrame(rows).to_csv(RES/f'seed{seed}_persistence.csv',index=False)
 pres=[]
 for name in ['advance','retard','random']:
  ep=torch.linalg.vector_norm(conds[name][1]-conds['zero'][1],dim=1);yp=m.nearest_logits(conds[name][1]).argmax(1).numpy();pres.append(dict(condition=name,Y_same=np.mean(yp==te[6][:N]),endpoint_pass=float((ep<TOL).float().mean()),endpoint_mean=float(ep.mean()),endpoint_p99=float(torch.quantile(ep,.99))))
 pd.DataFrame(pres).to_csv(RES/f'seed{seed}_preservation.csv',index=False)
 Jrows=[]
 for i in range(min(64,N)):
  y=Y[i];s0=st[i,tt[i]].detach();zi=z[i];J=jac(m,s0,zi);M=np.eye(D)+alpha*K[y];Jmod=np.eye(D)+M@(J-np.eye(D))
  with torch.no_grad():sn=m.step(s0[None],zi[None])[0];sm=stepmod(m,s0[None],zi[None],torch.tensor(M[None],dtype=torch.float32))[0];sn2=m.step(sn[None],zi[None])[0];sm2=stepmod(m,sm[None],zi[None],torch.tensor(M[None],dtype=torch.float32))[0]
  d1=(sm-sn).numpy();act=(sm2-sn2).numpy();Jnat1=jac(m,sn,zi);Jm1=jac(m,sm,zi);Jmod1=np.eye(D)+M@(Jm1-np.eye(D));pn=Jnat1@d1;pm=Jmod1@d1
  co=lambda a,b:float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
  Jrows.append(dict(seed=seed,index=i,J_change=np.linalg.norm(Jmod-J,'fro')/(np.linalg.norm(J,'fro')+1e-12),nativeJ_cos=co(pn,act),modifiedJ_cos=co(pm,act),modified_minus_native=co(pm,act)-co(pn,act)))
 jdf=pd.DataFrame(Jrows);jdf.to_csv(RES/f'seed{seed}_modifiedJ.csv',index=False)
 base=st[np.arange(N),tt].clone();g=m.step(base,z);d=g-base;compat=[]
 ces={q:centroid(Utr,tr[2],tr[1],q) for q in [max(0,p-1),p,min(8,p+1)]}
 for name,sgn,KK in [('advance',1,K),('zero',0,K),('retard',-1,K),('random',1,Kr)]:
  M=torch.tensor(np.eye(D)[None]+sgn*alpha*KK[Y],dtype=torch.float32);dm=torch.einsum('bij,bj->bi',M,d);u=(dm/(torch.linalg.vector_norm(dm,dim=1,keepdim=True)+1e-8)).detach().numpy();rr={'condition':name}
  for q,ce in ces.items():rr[f'reader_{q}']=reader_acc(u,Y,C,ce)
  compat.append(rr)
 pd.DataFrame(compat).to_csv(RES/f'seed{seed}_reader_compat.csv',index=False)
 def fitcanon(q,ref=4):
  mods=[]
  for y in range(16):mods.append(Ridge(alpha=1e-3).fit(Utr[tr[2]==y,q],Utr[tr[2]==y,ref]))
  return mods
 mods=fitcanon(p);can=[]
 for name,sgn,KK in [('advance',1,K),('zero',0,K),('retard',-1,K),('random',1,Kr)]:
  M=torch.tensor(np.eye(D)[None]+sgn*alpha*KK[Y],dtype=torch.float32);dm=torch.einsum('bij,bj->bi',M,d);u=(dm/(torch.linalg.vector_norm(dm,dim=1,keepdim=True)+1e-8)).detach().numpy();cc=np.empty_like(u)
  for y in range(16):ix=Y==y;cc[ix]=mods[y].predict(u[ix])
  can.append((name,u,cc))
 zraw=dict((n,u) for n,u,c in can);zcan=dict((n,c) for n,u,c in can);raw_adv=np.mean(np.linalg.norm(zraw['advance']-zraw['zero'],axis=1));raw_ret=np.mean(np.linalg.norm(zraw['retard']-zraw['zero'],axis=1));can_adv=np.mean(np.linalg.norm(zcan['advance']-zcan['zero'],axis=1));can_ret=np.mean(np.linalg.norm(zcan['retard']-zcan['zero'],axis=1));signcos=np.mean(np.sum((zraw['advance']-zraw['zero'])*(-(zraw['retard']-zraw['zero'])),1)/(np.linalg.norm(zraw['advance']-zraw['zero'],axis=1)*np.linalg.norm(zraw['retard']-zraw['zero'],axis=1)+1e-8))
 adv=[x for x in compat if x['condition']=='advance'][0];ret=[x for x in compat if x['condition']=='retard'][0];zero=[x for x in compat if x['condition']=='zero'][0];pr=[x for x in pres if x['condition']=='advance'][0]
 out=dict(seed=seed,status='OK',stage_bin=p,alpha=alpha,validation_J_change=jchg,test_J_change=jdf.J_change.mean(),modifiedJ_cos=jdf.modifiedJ_cos.mean(),nativeJ_cos=jdf.nativeJ_cos.mean(),modified_minus_native=jdf.modified_minus_native.mean(),raw_adv_change=raw_adv,raw_ret_change=raw_ret,canonical_adv_change=can_adv,canonical_ret_change=can_ret,sign_reversal_cos=signcos,advance_earlier=adv[f'reader_{max(0,p-1)}'],advance_native=adv[f'reader_{p}'],advance_later=adv[f'reader_{min(8,p+1)}'],zero_earlier=zero[f'reader_{max(0,p-1)}'],zero_native=zero[f'reader_{p}'],zero_later=zero[f'reader_{min(8,p+1)}'],retard_earlier=ret[f'reader_{max(0,p-1)}'],retard_native=ret[f'reader_{p}'],retard_later=ret[f'reader_{min(8,p+1)}'],Y_same=pr['Y_same'],endpoint_pass=pr['endpoint_pass'],endpoint_mean=pr['endpoint_mean'])
 pd.DataFrame([out]).to_csv(RES/f'seed{seed}_summary.csv',index=False);print(out,flush=True);return out
if __name__=='__main__':
 rows=[analyze(s) for s in SEEDS];pd.DataFrame(rows).to_csv(RES/'cross_seed_summary.csv',index=False)
