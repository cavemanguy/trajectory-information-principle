from __future__ import annotations

import math
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from experiments.r9_t1 import run_r9_t1 as base

N_VAL = base.N_VAL


def acc_on_mask(pred, target, mask):
    m = mask & (target >= 0)
    n = int(m.sum())
    if n == 0:
        return float("nan"), 0
    return float((pred[m] == target[m]).float().mean()), n


def task_metrics(model, batch, control_mode="native"):
    model.eval()
    with torch.no_grad():
        logits = model(batch["types"], batch["payload"], control_mode=control_mode)
    pred = logits.argmax(-1)
    data = batch["target"] >= 0
    out = {}
    for key, mask in [
        ("DATA_ACC", data),
        ("NEW_EARLY_ACC", batch["new_early"]),
        ("RETURN_EARLY_ACC", batch["return_early"]),
        ("STEADY_ACC", batch["steady"]),
        ("LONG_GAP_ACC", batch["long_gap"]),
    ]:
        a, n = acc_on_mask(pred, batch["target"], mask)
        out[key] = a
        out[key + "_N"] = n
    out["CROSS_ENTROPY"] = float(base.masked_loss(logits, batch["target"]))
    return out


def _state_pack(model, batch, mode="native"):
    model.eval()
    with torch.no_grad():
        logits, states = model(batch["types"], batch["payload"], return_states=True, control_mode=mode)
    return logits.detach().cpu(), {k: v.detach().cpu() for k, v in states.items()}


def _valid_indices(batch, lag=2):
    valid = batch["active_key"] >= 0
    valid[:, :lag] = False
    return valid.nonzero(as_tuple=False)


def _features(states, idx, stream):
    s = states[stream]
    b = idx[:, 0]
    t = idx[:, 1]
    cur = s[b, t].numpy()
    prev = s[b, t - 1].numpy()
    prev2 = s[b, t - 2].numpy()
    d1 = cur - prev
    d2 = prev - prev2
    return cur, d1, d2


def _targets(batch, idx, key="active_key"):
    return batch[key][idx[:, 0], idx[:, 1]].numpy()


def _probe_train_test(train_X, train_y, test_X, test_y):
    fit = base.fit_ridge(train_X, train_y)
    return base.ridge_acc(fit, test_X, test_y)


def history_accessibility(model, train_batch, test_batch, shuffle_seed):
    _, tr_states = _state_pack(model, train_batch)
    _, te_states = _state_pack(model, test_batch)
    tr_idx = _valid_indices(train_batch, 2)
    te_idx = _valid_indices(test_batch, 2)
    tr_y = _targets(train_batch, tr_idx)
    te_y = _targets(test_batch, te_idx)
    rng = np.random.default_rng(shuffle_seed)
    out = {}
    for stream in model.probe_streams:
        tr_h, tr_d1, tr_d2 = _features(tr_states, tr_idx, stream)
        te_h, te_d1, te_d2 = _features(te_states, te_idx, stream)
        ptr = rng.permutation(len(tr_h))
        pte = rng.permutation(len(te_h))
        state_acc = _probe_train_test(tr_h, tr_y, te_h, te_y)
        one_acc = _probe_train_test(
            np.concatenate([tr_h, tr_d1], 1), tr_y,
            np.concatenate([te_h, te_d1], 1), te_y,
        )
        multi_acc = _probe_train_test(
            np.concatenate([tr_h, tr_d1, tr_d2], 1), tr_y,
            np.concatenate([te_h, te_d1, te_d2], 1), te_y,
        )
        shuf_acc = _probe_train_test(
            np.concatenate([tr_h, tr_d1[ptr]], 1), tr_y,
            np.concatenate([te_h, te_d1[pte]], 1), te_y,
        )
        disp_acc = _probe_train_test(
            np.concatenate([tr_d1, tr_d2], 1), tr_y,
            np.concatenate([te_d1, te_d2], 1), te_y,
        )
        out[stream] = {
            "state_acc": state_acc,
            "one_step_acc": one_acc,
            "multi_step_acc": multi_acc,
            "shuffled_acc": shuf_acc,
            "displacement_only_acc": disp_acc,
            "h1": one_acc - state_acc,
            "h2": multi_acc - state_acc,
            "hshuf": one_acc - shuf_acc,
            "n_test": int(len(te_y)),
        }
    return out, tr_states, te_states, tr_idx, te_idx


def complementarity(model, train_batch, test_batch, tr_states, te_states, tr_idx, te_idx, shuffle_seed):
    if model.pair_streams is None:
        return None
    a, b = model.pair_streams
    tr_y = _targets(train_batch, tr_idx)
    te_y = _targets(test_batch, te_idx)
    tr_a, _, _ = _features(tr_states, tr_idx, a)
    te_a, _, _ = _features(te_states, te_idx, a)
    tr_b, _, _ = _features(tr_states, tr_idx, b)
    te_b, _, _ = _features(te_states, te_idx, b)
    rng = np.random.default_rng(shuffle_seed)
    ptr = rng.permutation(len(tr_a))
    pte = rng.permutation(len(te_a))
    aa = _probe_train_test(tr_a, tr_y, te_a, te_y)
    ab = _probe_train_test(tr_b, tr_y, te_b, te_y)
    joint = _probe_train_test(
        np.concatenate([tr_a, tr_b], 1), tr_y,
        np.concatenate([te_a, te_b], 1), te_y,
    )
    shuf = _probe_train_test(
        np.concatenate([tr_a, tr_b[ptr]], 1), tr_y,
        np.concatenate([te_a, te_b[pte]], 1), te_y,
    )
    rel_tr = np.stack([np.linalg.norm(tr_a, axis=1), np.linalg.norm(tr_b, axis=1)], 1)
    rel_te = np.stack([np.linalg.norm(te_a, axis=1), np.linalg.norm(te_b, axis=1)], 1)
    rel = _probe_train_test(rel_tr, tr_y, rel_te, te_y)
    return {
        "stream_a": a,
        "stream_b": b,
        "a_acc": aa,
        "b_acc": ab,
        "joint_acc": joint,
        "shuffled_pair_acc": shuf,
        "relation_norm_acc": rel,
        "comp": joint - max(aa, ab),
        "comp_shuf": joint - shuf,
    }


def cross_module(history, model):
    streams = list(model.probe_streams)
    first = streams[0]
    last = streams[-1]
    return {
        "first_stream": first,
        "last_stream": last,
        "first_key_acc": history[first]["state_acc"],
        "last_key_acc": history[last]["state_acc"],
        "xform": history[last]["state_acc"] - history[first]["state_acc"],
    }


def return_rank(batch):
    B, T = batch["types"].shape
    rr = torch.full((B, T), -1, dtype=torch.long)
    for i in range(B):
        in_return = False
        rank = 0
        for t in range(T):
            typ = int(batch["types"][i, t])
            if typ == base.ACTIVATE:
                in_return = True
                rank = 0
            elif typ in (base.REGIME, base.KEY):
                if typ == base.REGIME:
                    in_return = False
            elif in_return and bool(batch["return_early"][i, t]):
                rr[i, t] = rank
                rank += 1
    return rr


def reactivation_diagnostic(model, train_batch, test_batch, tr_states, te_states, tr_idx):
    tr_y = _targets(train_batch, tr_idx)
    rr = return_rank(test_batch)
    out = {}
    for stream in model.probe_streams:
        tr_h, _, _ = _features(tr_states, tr_idx, stream)
        fit = base.fit_ridge(tr_h, tr_y)
        vals = []
        for rank in range(4):
            idx = (rr == rank).nonzero(as_tuple=False)
            if len(idx) == 0:
                vals.append(float("nan"))
                continue
            s = te_states[stream]
            X = s[idx[:, 0], idx[:, 1]].numpy()
            y = test_batch["active_key"][idx[:, 0], idx[:, 1]].numpy()
            vals.append(base.ridge_acc(fit, X, y))
        out[stream] = {
            "rank_acc": vals,
            "react": vals[3] - vals[0] if all(math.isfinite(v) for v in (vals[0], vals[3])) else float("nan"),
        }
    return out


def path_dependence(model, test_batch, logits, te_states):
    stream = model.probe_streams[-1]
    s = te_states[stream]
    pred = logits.argmax(-1)
    groups = []
    for k in range(N_VAL):
        for x in range(N_VAL):
            ctx = (test_batch["active_key"] == k) & (test_batch["payload"] == x)
            ni = (ctx & test_batch["new_early"]).nonzero(as_tuple=False)
            ri = (ctx & test_batch["return_early"]).nonzero(as_tuple=False)
            if len(ni) < 3 or len(ri) < 3:
                continue
            hn = s[ni[:, 0], ni[:, 1]]
            hr = s[ri[:, 0], ri[:, 1]]
            cn = hn.mean(0)
            cr = hr.mean(0)
            scale = 0.5 * (torch.linalg.vector_norm(cn) + torch.linalg.vector_norm(cr)) + 1e-6
            geom = float(torch.linalg.vector_norm(cn - cr) / scale)
            tn = test_batch["target"][ni[:, 0], ni[:, 1]]
            tr = test_batch["target"][ri[:, 0], ri[:, 1]]
            an = float((pred[ni[:, 0], ni[:, 1]] == tn).float().mean())
            ar = float((pred[ri[:, 0], ri[:, 1]] == tr).float().mean())
            groups.append((geom, ar - an))
    if not groups:
        return {"stream": stream, "matched_groups": 0, "geometry": float("nan"), "functional_gap": float("nan")}
    return {
        "stream": stream,
        "matched_groups": len(groups),
        "geometry": float(np.mean([g[0] for g in groups])),
        "functional_gap": float(np.mean([g[1] for g in groups])),
    }


def linear_cka(X, Y, max_n=1024):
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    if len(X) > max_n:
        idx = np.linspace(0, len(X) - 1, max_n).astype(int)
        X = X[idx]
        Y = Y[idx]
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    xy = np.linalg.norm(X.T @ Y, ord="fro") ** 2
    xx = np.linalg.norm(X.T @ X, ord="fro")
    yy = np.linalg.norm(Y.T @ Y, ord="fro")
    return float(xy / max(1e-12, xx * yy))


def pair_similarity(model, te_states, te_idx):
    if model.pair_streams is None:
        return None
    a, b = model.pair_streams
    Xa, _, _ = _features(te_states, te_idx, a)
    Xb, _, _ = _features(te_states, te_idx, b)
    return {"stream_a": a, "stream_b": b, "linear_cka": linear_cka(Xa, Xb)}


def controller_audit(model, train_batch, test_batch):
    if model.controller_signal is None:
        return None
    _, tr_states = _state_pack(model, train_batch)
    _, te_states = _state_pack(model, test_batch)
    tr_mask = train_batch["target"] >= 0
    te_mask = test_batch["target"] >= 0
    tri = tr_mask.nonzero(as_tuple=False)
    tei = te_mask.nonzero(as_tuple=False)
    sig = model.controller_signal
    Xtr = tr_states[sig][tri[:, 0], tri[:, 1]].numpy()
    Xte = te_states[sig][tei[:, 0], tei[:, 1]].numpy()
    ytr = train_batch["target"][tri[:, 0], tri[:, 1]].numpy()
    yte = test_batch["target"][tei[:, 0], tei[:, 1]].numpy()
    signal_acc = _probe_train_test(Xtr, ytr, Xte, yte)
    ptr = train_batch["payload"][tri[:, 0], tri[:, 1]].numpy()
    pte = test_batch["payload"][tei[:, 0], tei[:, 1]].numpy()
    ctr = np.eye(N_VAL)[ptr]
    cte = np.eye(N_VAL)[pte]
    context_acc = _probe_train_test(ctr, ytr, cte, yte)
    return {
        "signal": sig,
        "signal_target_acc": signal_acc,
        "payload_context_acc": context_acc,
        "signal_minus_context": signal_acc - context_acc,
        "n_test": int(len(yte)),
    }


def control_effects(model, test_batch):
    if not getattr(model, "control_capable", False):
        return None
    modes = ["native", "shuffled", "zero", "random", "wrong"]
    metrics = {m: task_metrics(model, test_batch, m) for m in modes}
    native = metrics["native"]
    return {
        "modes": metrics,
        "ctrl_shuf": native["RETURN_EARLY_ACC"] - metrics["shuffled"]["RETURN_EARLY_ACC"],
        "ctrl_zero": native["RETURN_EARLY_ACC"] - metrics["zero"]["RETURN_EARLY_ACC"],
        "ctrl_rand": native["RETURN_EARLY_ACC"] - metrics["random"]["RETURN_EARLY_ACC"],
        "ctrl_wrong": native["RETURN_EARLY_ACC"] - metrics["wrong"]["RETURN_EARLY_ACC"],
    }


def mechanistic_diagnostics(model, train_batch, test_batch, shuffle_seed):
    logits, _ = _state_pack(model, test_batch)
    history, tr_states, te_states, tr_idx, te_idx = history_accessibility(
        model, train_batch, test_batch, shuffle_seed
    )
    comp = complementarity(
        model, train_batch, test_batch, tr_states, te_states, tr_idx, te_idx, shuffle_seed + 1
    )
    react = reactivation_diagnostic(model, train_batch, test_batch, tr_states, te_states, tr_idx)
    return {
        "history": history,
        "complementarity": comp,
        "cross_module": cross_module(history, model),
        "reactivation": react,
        "path_dependence": path_dependence(model, test_batch, logits, te_states),
        "pair_similarity": pair_similarity(model, te_states, te_idx),
    }


def teacher_off_history(teacher_model, control_rnn, train_batch, test_batch, shuffle_seed):
    def one(model, mode):
        _, tr_states = _state_pack(model, train_batch, mode)
        _, te_states = _state_pack(model, test_batch, mode)
        tr_idx = _valid_indices(train_batch, 2)
        te_idx = _valid_indices(test_batch, 2)
        tr_y = _targets(train_batch, tr_idx)
        te_y = _targets(test_batch, te_idx)
        stream = "RNN"
        tr_h, tr_d1, _ = _features(tr_states, tr_idx, stream)
        te_h, te_d1, _ = _features(te_states, te_idx, stream)
        state = _probe_train_test(tr_h, tr_y, te_h, te_y)
        onea = _probe_train_test(
            np.concatenate([tr_h, tr_d1], 1), tr_y,
            np.concatenate([te_h, te_d1], 1), te_y,
        )
        return {"state": state, "one": onea, "h1": onea - state}
    return {
        "teacher_off": one(teacher_model, "zero"),
        "never_taught": one(control_rnn, "native"),
    }
