"""AFO-1A: reference task, learned recurrent source, and isolated replay adapter."""
from __future__ import annotations
import copy
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from experiments.afo_1.protocol import Action, canonical

PROTOCOL = 'AFO-1A-v1'
SEEDS = (4109, 4133, 4153, 4177, 4201, 4219, 4241, 4261)
START, MODE, DATA, PAD = range(4)
NVAL, SEQ_LEN, HIDDEN = 8, 32, 64
STEPS, BATCH, EVAL_N = 1600, 64, 1024
OP_NAMES = ('LOAD', 'ADD', 'XOR', 'ROTL_XOR')


def derive_seed(seed, name):
    d = hashlib.sha256(f'{PROTOCOL}|{seed}|{name}'.encode()).digest()
    return int.from_bytes(d[:8], 'little') % (2**31 - 1)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def apply_op(a, mode, x):
    a, mode, x = int(a), int(mode), int(x)
    if not (0 <= a < 8 and 0 <= x < 8 and 0 <= mode < 4):
        raise ValueError('invalid reference operation')
    return (x, (a+x) % 8, a ^ x, (((a << 1) | (a >> 2)) & 7) ^ x)[mode]


def make_batch(seed, n):
    rng = np.random.default_rng(seed)
    typ = np.full((n, SEQ_LEN), PAD, np.int64)
    payload = np.full((n, SEQ_LEN), 8, np.int64)
    target = np.full((n, SEQ_LEN), -100, np.int64)
    op = np.full((n, SEQ_LEN), -1, np.int64)
    before = np.full((n, SEQ_LEN), -1, np.int64)
    after = np.full((n, SEQ_LEN), -1, np.int64)
    block = np.full((n, SEQ_LEN), -1, np.int64)
    initial = rng.integers(0, 8, size=n)
    modes = rng.integers(0, 4, size=(n, 3))
    values = rng.integers(0, 8, size=(n, 3, 8))
    typ[:, 0], payload[:, 0] = START, initial
    acc = initial.copy()
    pos = 1
    for b in range(3):
        typ[:, pos], payload[:, pos] = MODE, modes[:, b]
        pos += 1
        for k in range(8):
            x = values[:, b, k]
            old = acc.copy()
            acc = np.asarray([apply_op(a, m, v) for a, m, v in zip(acc, modes[:, b], x)], dtype=np.int64)
            typ[:, pos], payload[:, pos] = DATA, x
            target[:, pos], op[:, pos] = acc, modes[:, b]
            before[:, pos], after[:, pos], block[:, pos] = old, acc, b
            pos += 1
    assert pos == 28
    return {'types': typ, 'payload': payload, 'target': target, 'op': op,
            'before': before, 'after': after, 'block': block}


def model_inputs(batch):
    return torch.as_tensor(batch['types'], dtype=torch.long), torch.as_tensor(batch['payload'], dtype=torch.long)


class Source(nn.Module):
    def __init__(self):
        super().__init__()
        self.type_emb = nn.Embedding(4, 16)
        self.payload_emb = nn.Embedding(9, 16)
        self.norm = nn.LayerNorm(32)
        self.cell = nn.GRUCell(32, HIDDEN)
        self.head = nn.Sequential(nn.Linear(HIDDEN, 64), nn.GELU(), nn.Linear(64, 8))

    def transition(self, h, typ, payload):
        x = self.norm(torch.cat([self.type_emb(typ), self.payload_emb(payload)], -1))
        new = self.cell(x, h)
        return torch.where((typ != PAD).unsqueeze(-1), new, h)

    def forward(self, types, payload, return_states=False):
        h = torch.zeros(types.shape[0], HIDDEN, dtype=self.type_emb.weight.dtype, device=types.device)
        states, logits = [], []
        for t in range(types.shape[1]):
            h = self.transition(h, types[:, t], payload[:, t])
            states.append(h)
            logits.append(self.head(h))
        logits = torch.stack(logits, 1)
        if return_states:
            return logits, torch.stack(states, 1)
        return logits


def parameter_fingerprint(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        t = value.detach().cpu().contiguous()
        h.update(canonical([name, list(t.shape), str(t.dtype)]))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def save_checkpoint(model, path):
    path = Path(path)
    torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checkpoint(path):
    model = Source()
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class SourceAdapter:
    """Owns a frozen model and complete deterministic continuation state."""
    def __init__(self, model, hidden=None, index=0):
        self.model = model.eval()
        if self.model.training:
            raise RuntimeError('source must be in evaluation mode')
        if any(p.device.type != 'cpu' for p in self.model.parameters()):
            raise ValueError('AFO-1A adapter requires CPU parameters')
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.h = torch.zeros(1, HIDDEN) if hidden is None else hidden.detach().clone().reshape(1, HIDDEN)
        self.index = int(index)
        self._fingerprint = parameter_fingerprint(self.model)

    @property
    def fingerprint(self):
        return self._fingerprint

    def _check_frozen(self):
        if self.model.training or parameter_fingerprint(self.model) != self._fingerprint:
            raise RuntimeError('frozen model parameters or evaluation mode changed')

    def snapshot(self):
        self._check_frozen()
        return {'h': self.h.detach().reshape(-1).tolist(), 'index': self.index}

    def restore(self, snapshot):
        if (type(snapshot) is not dict or set(snapshot) != {'h', 'index'} or
            type(snapshot['index']) is not int or snapshot['index'] < 0 or
            type(snapshot['h']) is not list or len(snapshot['h']) != HIDDEN):
            raise ValueError('invalid complete-state snapshot')
        h = torch.tensor(snapshot['h'], dtype=torch.float32).reshape(1, HIDDEN)
        if not torch.isfinite(h).all():
            raise ValueError('nonfinite state')
        self.h, self.index = h.clone(), int(snapshot['index'])

    def observe(self):
        return tuple(self.h.detach().reshape(-1).tolist())

    def step(self, event):
        self._check_frozen()
        if type(event) is not dict or set(event) != {'type', 'payload'}:
            raise ValueError('event must contain only type and payload')
        if type(event['type']) is not int or type(event['payload']) is not int:
            raise ValueError('event fields must be integers')
        typ, payload = event['type'], event['payload']
        legal = ((typ == START and 0 <= payload < 8) or
                 (typ == MODE and 0 <= payload < 4) or
                 (typ == DATA and 0 <= payload < 8) or
                 (typ == PAD and payload == 8))
        if not legal:
            raise ValueError('illegal event')
        with torch.no_grad():
            t = torch.tensor([typ]); p = torch.tensor([payload])
            self.h = self.model.transition(self.h, t, p).detach().clone()
            logits_tensor = self.model.head(self.h)
            if not torch.isfinite(self.h).all() or not torch.isfinite(logits_tensor).all():
                raise FloatingPointError('nonfinite frozen continuation')
            logits = logits_tensor.reshape(-1).tolist()
        self.index += 1
        return {'prediction': int(np.argmax(logits)), 'logits': logits}

    def perturb(self, action):
        self._check_frozen()
        if not isinstance(action, Action):
            raise TypeError('expected a validated Action')
        if action.kind == 'zero':
            return
        if action.kind != 'vector' or len(action.values) != HIDDEN:
            raise ValueError('only bounded 64D state vectors are legal')
        with torch.no_grad():
            self.h = self.h + torch.tensor(action.values, dtype=torch.float32).reshape(1, HIDDEN)
        if not torch.isfinite(self.h).all():
            raise ValueError('nonfinite perturbed state')

    def fork(self):
        m = Source()
        m.load_state_dict(copy.deepcopy(self.model.state_dict()))
        fork = SourceAdapter(m, index=self.index)
        if fork.h.data_ptr() == self.h.data_ptr() or any(
            a.data_ptr() == b.data_ptr() for a, b in zip(self.model.parameters(), fork.model.parameters())
        ):
            raise RuntimeError('fork shares mutable storage')
        if fork.fingerprint != self.fingerprint:
            raise RuntimeError('fork parameter fingerprint mismatch')
        return fork
