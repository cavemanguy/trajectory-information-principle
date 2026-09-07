"""R9-S1 synchronous recurrent swarms. No scientific outcome-dependent parameters."""
import hashlib
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from experiments.r9_t1 import run_r9_t1 as task

N, H, M, E = 8, 8, 4, 16
ARMS = ('DISCONNECTED','RING','RANDOM','FULL','LEARNED','RING_HET','FULL_HET','MONOLITHIC')


def seed_for(seed, name):
    d = hashlib.sha256(f'r9-s1|{seed}|{name}'.encode()).digest()
    return int.from_bytes(d[:8], 'little') % (2**31-1)


def adjacency(kind, seed):
    a = np.zeros((N,N), np.float32)
    if kind == 'DISCONNECTED':
        return torch.from_numpy(a)
    if kind in ('RING','RING_HET'):
        for i in range(N):
            a[i,(i-1)%N] = a[i,(i+1)%N] = 0.5
    elif kind in ('FULL','FULL_HET'):
        a[:] = (np.ones((N,N),np.float32)-np.eye(N,dtype=np.float32))/(N-1)
    elif kind == 'RANDOM':
        rng = np.random.default_rng(seed)
        for _ in range(10000):
            a[:] = 0
            for i in range(N):
                a[i,rng.choice([j for j in range(N) if j != i],2,replace=False)] = 0.5
            reach = (a > 0).astype(np.int64)
            r = np.eye(N,dtype=np.int64)
            for _ in range(N):
                r = ((r + r @ reach)>0).astype(np.int64)
            if bool(r.all()):
                break
        else:
            raise RuntimeError('no strongly connected random graph')
    else:
        raise ValueError(kind)
    return torch.from_numpy(a.copy())


class Cells(nn.Module):
    def __init__(self, hetero=False):
        super().__init__()
        k = N if hetero else 1
        self.wx = nn.Parameter(torch.empty(k,3*H,H+M))
        self.wh = nn.Parameter(torch.empty(k,3*H,H))
        self.bx = nn.Parameter(torch.zeros(k,3*H))
        self.bh = nn.Parameter(torch.zeros(k,3*H))
        for w in (self.wx,self.wh):
            for i in range(k):
                nn.init.uniform_(w[i],-1/math.sqrt(H),1/math.sqrt(H))
    def forward(self, x, h):
        a = torch.einsum('bni,noi->bno',x,self.wx) if self.wx.shape[0] == N else F.linear(x,self.wx[0])
        b = torch.einsum('bni,noi->bno',h,self.wh) if self.wh.shape[0] == N else F.linear(h,self.wh[0])
        ax = a + self.bx
        bh = b + self.bh
        ar, az, an = ax.chunk(3,-1)
        br, bz, bn = bh.chunk(3,-1)
        r = torch.sigmoid(ar+br)
        z = torch.sigmoid(az+bz)
        n = torch.tanh(an+r*bn)
        return (1-z)*n+z*h


class Swarm(nn.Module):
    def __init__(self, kind, graph_seed):
        super().__init__()
        self.kind = kind
        self.encoder = task.TokenEncoder(E)
        self.input_proj = nn.Linear(E,H)
        self.hetero = kind.endswith('_HET')
        self.cells = Cells(self.hetero)
        self.message = nn.Parameter(torch.empty(N if self.hetero else 1,M,H))
        self.message_bias = nn.Parameter(torch.zeros(N if self.hetero else 1,M))
        nn.init.uniform_(self.message,-1/math.sqrt(H),1/math.sqrt(H))
        self.head = nn.Sequential(nn.Linear(N*H,128),nn.GELU(),nn.Linear(128,task.N_VAL))
        self.register_buffer('ports',torch.tensor([task.KEY,task.KEY,task.REGIME,task.REGIME,task.DATA,task.DATA,task.DISTRACTOR,task.DISTRACTOR]))
        self.register_buffer('activate_ports',torch.tensor([False,False,True,True,False,False,False,False]))
        self.register_buffer('graph',adjacency('DISCONNECTED' if kind == 'LEARNED' else kind,graph_seed))
        self.register_buffer('rewired',adjacency('RANDOM' if kind not in ('DISCONNECTED','RANDOM','RING','RING_HET') else 'RING',seed_for(graph_seed,'rewire')))
        if kind in ('RING','RING_HET'):
            self.rewired = adjacency('RANDOM',seed_for(graph_seed,'rewire'))
        if kind == 'RANDOM':
            self.rewired = adjacency('RANDOM',seed_for(graph_seed,'rewire'))
        if kind in ('FULL','FULL_HET'):
            self.rewired = self.graph.clone()
        if kind == 'LEARNED':
            self.q = nn.Linear(H,4,bias=False)
            self.k = nn.Linear(H,4,bias=False)
        self.probe_streams = ['JOINT']+[f'N{i}' for i in range(N)]
    def encode(self, types, payload):
        return self.encoder(types,payload)
    def local_input(self, e, typ):
        mask = (typ[:,None] == self.ports[None,:]) | ((typ[:,None] == task.ACTIVATE) & self.activate_ports[None,:])
        return self.input_proj(e)[:,None,:] * mask[:,:,None].to(e.dtype)
    def messages(self,h):
        if self.hetero:
            return torch.einsum('bni,nmi->bnm',h,self.message)+self.message_bias[None]
        return F.linear(h,self.message[0],self.message_bias[0])
    def routing(self,h,mode='native'):
        if mode == 'zero':
            return self.graph
        if mode == 'rewire':
            return self.rewired
        if self.kind != 'LEARNED':
            return self.graph
        q,k = self.q(h),self.k(h)
        scores = torch.matmul(q,k.transpose(1,2))/math.sqrt(4)
        eye = torch.eye(N,dtype=torch.bool,device=h.device)
        scores = scores.masked_fill(eye[None],float('-inf'))
        top = torch.topk(scores,2,dim=-1).indices
        mask = torch.zeros_like(scores,dtype=torch.bool).scatter_(-1,top,True)
        return torch.softmax(scores.masked_fill(~mask,float('-inf')),dim=-1)
    def step(self,h,e,typ,mode='native',perm=None,silence=None):
        msg = self.messages(h)
        if silence is not None:
            msg = msg.clone(); msg[:,silence] = 0
        if mode == 'shuffle':
            if perm is None: raise ValueError('shuffle requires permutation')
            msg = msg.gather(0,perm[:,None,None].expand_as(msg))
        if mode == 'zero' or self.kind == 'DISCONNECTED':
            agg = torch.zeros(h.shape[0],N,M,device=h.device,dtype=h.dtype)
        else:
            a = self.routing(h,mode)
            agg = torch.einsum('ij,bjm->bim',a,msg) if a.ndim == 2 else torch.bmm(a,msg)
        z = torch.cat([self.local_input(e,typ),agg],-1)
        new = self.cells(z,h)
        return torch.where((typ != task.PAD)[:,None,None],new,h)
    def readout(self,h,readout_zero=None):
        if readout_zero is not None:
            h = h.clone(); h[:,readout_zero] = 0
        return self.head(h.reshape(h.shape[0],-1))
    def forward(self,types,payload,return_states=False,mode='native',perms=None,silence=None,readout_zero=None):
        e = self.encode(types,payload)
        h = e.new_zeros(types.shape[0],N,H)
        states=[]; logits=[]
        for t in range(types.shape[1]):
            h = self.step(h,e[:,t],types[:,t],mode,None if perms is None else perms[t],silence)
            states.append(h)
            logits.append(self.readout(h,readout_zero))
        logits=torch.stack(logits,1)
        return (logits,torch.stack(states,1)) if return_states else logits


class Monolithic(nn.Module):
    kind='MONOLITHIC'
    probe_streams=['JOINT']
    def __init__(self):
        super().__init__()
        self.encoder=task.TokenEncoder(E)
        self.rnn=nn.GRU(E,64,batch_first=True)
        self.head=nn.Sequential(nn.Linear(64,128),nn.GELU(),nn.Linear(128,task.N_VAL))
    def encode(self,types,payload): return self.encoder(types,payload)
    def step(self,h,e,typ,mode='native',perm=None,silence=None):
        new=self.rnn(e[:,None],h[None])[1][0]
        return torch.where((typ != task.PAD)[:,None],new,h)
    def readout(self,h,readout_zero=None): return self.head(h)
    def forward(self,types,payload,return_states=False,mode='native',perms=None,silence=None,readout_zero=None):
        e=self.encode(types,payload); h=e.new_zeros(types.shape[0],64); states=[]; logits=[]
        for t in range(types.shape[1]):
            h=self.step(h,e[:,t],types[:,t]); states.append(h); logits.append(self.readout(h))
        logits=torch.stack(logits,1)
        return (logits,torch.stack(states,1)) if return_states else logits


def build_arm(name,seed):
    torch.manual_seed(seed_for(seed,'init_'+name))
    return Monolithic() if name == 'MONOLITHIC' else Swarm(name,seed_for(seed,'graph'))
