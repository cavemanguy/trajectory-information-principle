from __future__ import annotations

import math

import numpy as np
import torch

from experiments.r9_t1 import run_r9_t1 as base
from experiments.r9_h1 import diagnostics as dx

N_VAL = base.N_VAL


def _probe(train_X, train_y, test_X, test_y):
    fit = base.fit_ridge(train_X, train_y)
    return base.ridge_acc(fit, test_X, test_y)


def answer_history(model, train_batch, test_batch, shuffle_seed):
    _, tr_states = dx._state_pack(model, train_batch)
    _, te_states = dx._state_pack(model, test_batch)
    tr_valid = train_batch["target"] >= 0
    te_valid = test_batch["target"] >= 0
    tr_valid[:, :2] = False
    te_valid[:, :2] = False
    tr_idx = tr_valid.nonzero(as_tuple=False)
    te_idx = te_valid.nonzero(as_tuple=False)
    tr_y = train_batch["target"][tr_idx[:, 0], tr_idx[:, 1]].numpy()
    te_y = test_batch["target"][te_idx[:, 0], te_idx[:, 1]].numpy()
    rng = np.random.default_rng(shuffle_seed)
    out = {}
    for stream in model.probe_streams:
        tr_h, tr_d1, tr_d2 = dx._features(tr_states, tr_idx, stream)
        te_h, te_d1, te_d2 = dx._features(te_states, te_idx, stream)
        ptr = rng.permutation(len(tr_h))
        pte = rng.permutation(len(te_h))
        state = _probe(tr_h, tr_y, te_h, te_y)
        one = _probe(
            np.concatenate([tr_h, tr_d1], 1), tr_y,
            np.concatenate([te_h, te_d1], 1), te_y,
        )
        multi = _probe(
            np.concatenate([tr_h, tr_d1, tr_d2], 1), tr_y,
            np.concatenate([te_h, te_d1, te_d2], 1), te_y,
        )
        shuf = _probe(
            np.concatenate([tr_h, tr_d1[ptr]], 1), tr_y,
            np.concatenate([te_h, te_d1[pte]], 1), te_y,
        )
        out[stream] = {
            "state_acc": state,
            "one_step_acc": one,
            "multi_step_acc": multi,
            "shuffled_acc": shuf,
            "h1": one - state,
            "h2": multi - state,
            "hshuf": one - shuf,
            "n_test": int(len(te_y)),
        }
    return out


def _ordered_features(states, idx, stream):
    s = states[stream]
    b, t = idx[:, 0], idx[:, 1]
    h0 = s[b, t].numpy()
    h1 = s[b, t - 1].numpy()
    h2 = s[b, t - 2].numpy()
    h3 = s[b, t - 3].numpy()
    d0 = h0 - h1
    d1 = h1 - h2
    d2 = h2 - h3
    return d0, d1, d2


def local_order_controls(model, train_batch, test_batch, shuffle_seed):
    _, tr_states = dx._state_pack(model, train_batch)
    _, te_states = dx._state_pack(model, test_batch)
    tr_rr = dx.return_rank(train_batch)
    te_rr = dx.return_rank(test_batch)
    tr_idx = ((tr_rr >= 0) & (torch.arange(tr_rr.shape[1]).view(1, -1) >= 3)).nonzero(as_tuple=False)
    te_idx = ((te_rr >= 0) & (torch.arange(te_rr.shape[1]).view(1, -1) >= 3)).nonzero(as_tuple=False)
    tr_y = train_batch["active_key"][tr_idx[:, 0], tr_idx[:, 1]].numpy()
    te_y = test_batch["active_key"][te_idx[:, 0], te_idx[:, 1]].numpy()
    rng = np.random.default_rng(shuffle_seed)
    out = {}
    for stream in model.probe_streams:
        tr0, tr1, tr2 = _ordered_features(tr_states, tr_idx, stream)
        te0, te1, te2 = _ordered_features(te_states, te_idx, stream)
        ordered_tr = np.concatenate([tr2, tr1, tr0], 1)
        ordered_te = np.concatenate([te2, te1, te0], 1)
        reversed_tr = np.concatenate([tr0, tr1, tr2], 1)
        reversed_te = np.concatenate([te0, te1, te2], 1)
        tr_stack = np.stack([tr2, tr1, tr0], 1)
        te_stack = np.stack([te2, te1, te0], 1)
        for arr in (tr_stack, te_stack):
            for i in range(len(arr)):
                arr[i] = arr[i, rng.permutation(3)]
        shuffled_tr = tr_stack.reshape(len(tr_stack), -1)
        shuffled_te = te_stack.reshape(len(te_stack), -1)
        integrated_tr = tr0 + tr1 + tr2
        integrated_te = te0 + te1 + te2
        out[stream] = {
            "ordered_acc": _probe(ordered_tr, tr_y, ordered_te, te_y),
            "reversed_acc": _probe(reversed_tr, tr_y, reversed_te, te_y),
            "shuffled_order_acc": _probe(shuffled_tr, tr_y, shuffled_te, te_y),
            "integrated_acc": _probe(integrated_tr, tr_y, integrated_te, te_y),
            "first_transient_acc": _probe(tr2, tr_y, te2, te_y),
            "final_transient_acc": _probe(tr0, tr_y, te0, te_y),
            "n_test": int(len(te_y)),
        }
    return out


def _metrics_from_logits(logits, batch):
    pred = logits.argmax(-1)
    out = {}
    for key, mask in [
        ("DATA_ACC", batch["target"] >= 0),
        ("RETURN_EARLY_ACC", batch["return_early"]),
    ]:
        a, n = dx.acc_on_mask(pred, batch["target"], mask)
        out[key] = a
        out[key + "_N"] = n
    return out


def parallel_branch_ablation(model, batch):
    model.eval()
    with torch.no_grad():
        x = model.encoder(batch["types"], batch["payload"])
        B, T, D = x.shape
        h = torch.zeros(B, D, device=x.device)
        rs = []
        for t in range(T):
            h = model.rnn(x[:, t], h)
            rs.append(h)
        rs = torch.stack(rs, 1)
        tr = model.tr(x + model.pos[:T])
        native = model.head(model.fuse(torch.cat([rs, tr], -1)))
        no_rnn = model.head(model.fuse(torch.cat([torch.zeros_like(rs), tr], -1)))
        no_t = model.head(model.fuse(torch.cat([rs, torch.zeros_like(tr)], -1)))
    m0 = _metrics_from_logits(native, batch)
    mr = _metrics_from_logits(no_rnn, batch)
    mt = _metrics_from_logits(no_t, batch)
    return {
        "native": m0,
        "bypass_rnn": mr,
        "bypass_transformer": mt,
        "rnn_contribution_return": m0["RETURN_EARLY_ACC"] - mr["RETURN_EARLY_ACC"],
        "transformer_contribution_return": m0["RETURN_EARLY_ACC"] - mt["RETURN_EARLY_ACC"],
    }


def _direction_selectivity(resp, keys, max_n=4096):
    X = resp.reshape(-1, resp.shape[-1]).numpy()
    y = keys.reshape(-1).numpy()
    valid = y >= 0
    X = X[valid]
    y = y[valid]
    if len(X) > max_n:
        idx = np.linspace(0, len(X) - 1, max_n).astype(int)
        X = X[idx]
        y = y[idx]
    norm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    Xn = X / norm
    cents = []
    present = []
    for k in range(N_VAL):
        if np.sum(y == k) < 3:
            continue
        c = Xn[y == k].mean(0)
        c /= np.linalg.norm(c) + 1e-8
        cents.append(c)
        present.append(k)
    if len(cents) < 2:
        return float("nan")
    C = np.stack(cents)
    sims = Xn @ C.T
    same = []
    other = []
    lookup = {k: i for i, k in enumerate(present)}
    for i, k in enumerate(y):
        if k not in lookup:
            continue
        j = lookup[k]
        same.append(sims[i, j])
        other.append((sims[i].sum() - sims[i, j]) / (len(present) - 1))
    return float(np.mean(same) - np.mean(other))


def perturbation_response(model, train_batch, test_batch, kind):
    if kind == "TEACHER":
        _, tr_native = dx._state_pack(model, train_batch, "native")
        _, tr_zero = dx._state_pack(model, train_batch, "zero")
        _, te_native = dx._state_pack(model, test_batch, "native")
        _, te_zero = dx._state_pack(model, test_batch, "zero")
        tr_resp = tr_native["RNN"] - tr_zero["RNN"]
        te_resp = te_native["RNN"] - te_zero["RNN"]
        _, te_shuf = dx._state_pack(model, test_batch, "shuffled")
        _, te_rand = dx._state_pack(model, test_batch, "random")
        shuf_resp = te_shuf["RNN"] - te_zero["RNN"]
        rand_resp = te_rand["RNN"] - te_zero["RNN"]
    elif kind == "INTERROGATOR":
        _, tr_native = dx._state_pack(model, train_batch, "native")
        _, te_native = dx._state_pack(model, test_batch, "native")
        _, te_shuf = dx._state_pack(model, test_batch, "shuffled")
        _, te_rand = dx._state_pack(model, test_batch, "random")
        tr_resp = tr_native["RESPONSE"]
        te_resp = te_native["RESPONSE"]
        shuf_resp = te_shuf["RESPONSE"]
        rand_resp = te_rand["RESPONSE"]
    else:
        raise ValueError(kind)

    tr_mask = train_batch["active_key"] >= 0
    te_mask = test_batch["active_key"] >= 0
    tri = tr_mask.nonzero(as_tuple=False)
    tei = te_mask.nonzero(as_tuple=False)
    trX = tr_resp[tri[:, 0], tri[:, 1]].numpy()
    teX = te_resp[tei[:, 0], tei[:, 1]].numpy()
    shy = shuf_resp[tei[:, 0], tei[:, 1]].numpy()
    ray = rand_resp[tei[:, 0], tei[:, 1]].numpy()
    tr_y = train_batch["active_key"][tri[:, 0], tri[:, 1]].numpy()
    te_y = test_batch["active_key"][tei[:, 0], tei[:, 1]].numpy()
    fit = base.fit_ridge(trX, tr_y)
    native_key = base.ridge_acc(fit, teX, te_y)
    shuf_key = base.ridge_acc(fit, shy, te_y)
    rand_key = base.ridge_acc(fit, ray, te_y)
    return {
        "response_norm_native": float(torch.linalg.vector_norm(te_resp, dim=-1)[te_mask].mean()),
        "response_norm_shuffled": float(torch.linalg.vector_norm(shuf_resp, dim=-1)[te_mask].mean()),
        "response_norm_random": float(torch.linalg.vector_norm(rand_resp, dim=-1)[te_mask].mean()),
        "response_key_acc_native": native_key,
        "response_key_acc_shuffled": shuf_key,
        "response_key_acc_random": rand_key,
        "response_key_gain_vs_shuffled": native_key - shuf_key,
        "direction_selectivity": _direction_selectivity(te_resp, test_batch["active_key"]),
    }
