import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_IN = 16
N_CLASSES = 4
HIDDEN = 16
F_HIDDEN = 32
STEPS = 12
GAMMA = 0.75
NOISE_STD = 0.10
BATCH = 256
LR = 1e-3
WD = 1e-4
CLIP = 1.0

PRETRAIN_N = 12288
VAL_N = 3072
TEST_N = 6144
CAL_N = 4096
HOLD_N = 4096
SWITCH_N = 4096

PRETRAIN_EPOCHS = 80
CAL_EPOCHS = 20
HOLD_EPOCHS = 120
HOLD_CHECKS = (30, 60, 90, 120)
SWITCH_EPOCHS = 10
SWITCH_CHECKS = (1, 3, 5, 10)

GLOBAL_TASK_SEED = 20260905
BENCH_SEEDS = (1109, 1127, 1144, 1162, 1181, 1199, 1218, 1237, 1255, 1274, 1292, 1311)
CONDITIONS = ("FULL", "F1", "F2", "HEAD", "NOADAPT")
DRIFTS = {"A": -1.0, "B": +1.0}


def derive_seed(seed, name):
    h = hashlib.sha256(f"E1-CAL|{seed}|{name}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def clone_state(sd):
    return {k: v.detach().cpu().clone() for k, v in sd.items()}


def sha_state(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        t = sd[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def state_distance(sd, ref):
    total = 0.0
    for k in sd:
        d = sd[k].detach().cpu().double() - ref[k].detach().cpu().double()
        total += float(torch.sum(d * d))
    return math.sqrt(total)


def make_task_matrix():
    g = torch.Generator().manual_seed(GLOBAL_TASK_SEED)
    w = torch.randn(N_CLASSES, N_IN, generator=g)
    w = w - w.mean(dim=0, keepdim=True)
    return F.normalize(w, dim=1)


W_TASK = make_task_matrix()


def make_z(n, seed, name):
    g = torch.Generator().manual_seed(derive_seed(seed, f"z_{name}"))
    return torch.randn(n, N_IN, generator=g)


def labels_from_z(z):
    return torch.argmax(z @ W_TASK.T, dim=1)


def family_uv(seed):
    g = torch.Generator().manual_seed(derive_seed(seed, "family_uv"))
    a = torch.randn(N_CLASSES, generator=g)
    b = torch.randn(N_CLASSES, generator=g)
    u = a @ W_TASK
    u = u / torch.linalg.vector_norm(u)
    v = b @ W_TASK
    v = v - torch.dot(v, u) * u
    nv = torch.linalg.vector_norm(v)
    if float(nv) < 1e-6:
        v = torch.randn(N_IN, generator=g)
        v = v - torch.dot(v, u) * u
        nv = torch.linalg.vector_norm(v)
    v = v / nv
    return u, v


def observe(z, seed, name, u, v, drift):
    g = torch.Generator().manual_seed(derive_seed(seed, f"noise_{name}"))
    noise = NOISE_STD * torch.randn(z.shape, generator=g)
    if float(drift) == 0.0:
        return z + noise
    proj = z @ v
    return z + float(drift) * GAMMA * proj[:, None] * u[None, :] + noise


class RecurrentClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Linear(N_IN, HIDDEN)
        self.F1 = nn.Linear(HIDDEN, F_HIDDEN)
        self.F2 = nn.Linear(F_HIDDEN, HIDDEN)
        self.head = nn.Linear(HIDDEN, N_CLASSES)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.enc.weight)
        nn.init.zeros_(self.enc.bias)
        nn.init.xavier_uniform_(self.F1.weight)
        nn.init.zeros_(self.F1.bias)
        nn.init.xavier_uniform_(self.F2.weight)
        nn.init.zeros_(self.F2.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        h = torch.tanh(self.enc(x))
        for _ in range(STEPS):
            h = torch.tanh(self.F2(F.gelu(self.F1(h))))
        return self.head(h)


def set_trainable(model, condition):
    for p in model.parameters():
        p.requires_grad = False
    if condition == "FULL":
        for p in model.parameters():
            p.requires_grad = True
    elif condition == "F1":
        for p in model.F1.parameters():
            p.requires_grad = True
    elif condition == "F2":
        for p in model.F2.parameters():
            p.requires_grad = True
    elif condition == "HEAD":
        for p in model.head.parameters():
            p.requires_grad = True
    elif condition == "NOADAPT":
        pass
    else:
        raise ValueError(condition)


def trainable_count(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def make_optimizer(model):
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return None
    return torch.optim.AdamW(params, lr=LR, weight_decay=WD)


def train_epoch(model, opt, x, y, order_seed):
    if opt is None:
        return 0.0
    model.train()
    g = torch.Generator().manual_seed(order_seed)
    order = torch.randperm(len(x), generator=g)
    total = 0.0
    seen = 0
    trainable = [p for p in model.parameters() if p.requires_grad]
    for a in range(0, len(x), BATCH):
        ix = order[a:a+BATCH]
        opt.zero_grad(set_to_none=True)
        logits = model(x[ix])
        loss = F.cross_entropy(logits, y[ix])
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite loss")
        loss.backward()
        nn.utils.clip_grad_norm_(trainable, CLIP)
        opt.step()
        total += float(loss.detach()) * len(ix)
        seen += len(ix)
    return total / max(seen, 1)


def accuracy(model, x, y):
    model.eval()
    correct = 0
    with torch.no_grad():
        for a in range(0, len(x), BATCH):
            logits = model(x[a:a+BATCH])
            correct += int((torch.argmax(logits, dim=1) == y[a:a+BATCH]).sum())
    return float(correct / len(x))


def finite_obj(obj):
    if isinstance(obj, dict):
        return all(finite_obj(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(finite_obj(v) for v in obj)
    if isinstance(obj, (float, np.floating)):
        return math.isfinite(float(obj))
    return True


def make_data(seed, ntrain=PRETRAIN_N, nval=VAL_N, ntest=TEST_N, ncal=CAL_N, nhold=HOLD_N, nswitch=SWITCH_N):
    u, v = family_uv(seed)
    z_train = make_z(ntrain, seed, "pretrain_train")
    z_val = make_z(nval, seed, "pretrain_val")
    z_test = make_z(ntest, seed, "test")
    z_cal = make_z(ncal, seed, "cal")
    z_hold = make_z(nhold, seed, "hold")
    z_switch = make_z(nswitch, seed, "switch")
    ys = {
        "train": labels_from_z(z_train),
        "val": labels_from_z(z_val),
        "test": labels_from_z(z_test),
        "cal": labels_from_z(z_cal),
        "hold": labels_from_z(z_hold),
        "switch": labels_from_z(z_switch),
    }
    x = {
        "train0": observe(z_train, seed, "pretrain_train", u, v, 0.0),
        "val0": observe(z_val, seed, "pretrain_val", u, v, 0.0),
        "test0": observe(z_test, seed, "test", u, v, 0.0),
        "hold0": observe(z_hold, seed, "hold", u, v, 0.0),
    }
    for name, s in DRIFTS.items():
        x[f"test_{name}"] = observe(z_test, seed, "test", u, v, s)
        x[f"cal_{name}"] = observe(z_cal, seed, "cal", u, v, s)
        x[f"switch_{name}"] = observe(z_switch, seed, "switch", u, v, s)
    return x, ys, u, v


def run_condition(seed, drift_name, condition, base_state, base_hash, x, ys,
                  cal_epochs=CAL_EPOCHS, hold_epochs=HOLD_EPOCHS, hold_checks=HOLD_CHECKS,
                  switch_epochs=SWITCH_EPOCHS, switch_checks=SWITCH_CHECKS):
    model = RecurrentClassifier()
    model.load_state_dict(clone_state(base_state))
    if sha_state(model.state_dict()) != base_hash:
        raise RuntimeError(f"fork hash mismatch {drift_name}/{condition}")

    set_trainable(model, condition)
    ntrainable = trainable_count(model)
    opt = make_optimizer(model)

    base_neutral_acc = accuracy(model, x["test0"], ys["test"])
    base_drift_acc = accuracy(model, x[f"test_{drift_name}"], ys["test"])

    cal_batches = math.ceil(len(x[f"cal_{drift_name}"]) / BATCH)
    for ep in range(1, cal_epochs + 1):
        train_epoch(model, opt, x[f"cal_{drift_name}"], ys["cal"], derive_seed(seed, f"{drift_name}_cal_order_{ep}"))

    cal_state = clone_state(model.state_dict())
    cal_acc = accuracy(model, x[f"test_{drift_name}"], ys["test"])
    cal_neutral_acc = accuracy(model, x["test0"], ys["test"])
    cal_gain = cal_acc - base_drift_acc

    hold_records = []
    hold_batches = math.ceil(len(x["hold0"]) / BATCH)
    for ep in range(1, hold_epochs + 1):
        train_epoch(model, opt, x["hold0"], ys["hold"], derive_seed(seed, f"{drift_name}_hold_order_{ep}"))
        if ep in hold_checks:
            sd = clone_state(model.state_dict())
            return_acc = accuracy(model, x[f"test_{drift_name}"], ys["test"])
            neutral_acc = accuracy(model, x["test0"], ys["test"])
            hold_records.append({
                "hold_epoch": int(ep),
                "neutral_acc": neutral_acc,
                "return_acc": return_acc,
                "retained_gain": return_acc - base_drift_acc,
                "distance_from_post_calibration": state_distance(sd, cal_state),
                "distance_from_pretrained_base": state_distance(sd, base_state),
            })

    final_hold_state = clone_state(model.state_dict())
    opposite = "B" if drift_name == "A" else "A"
    switch_records = []
    switch_batches = math.ceil(len(x[f"switch_{opposite}"]) / BATCH)
    for ep in range(1, switch_epochs + 1):
        train_epoch(model, opt, x[f"switch_{opposite}"], ys["switch"], derive_seed(seed, f"{drift_name}_switch_to_{opposite}_order_{ep}"))
        if ep in switch_checks:
            switch_records.append({
                "switch_epoch": int(ep),
                "opposite_drift": opposite,
                "opposite_acc": accuracy(model, x[f"test_{opposite}"], ys["test"]),
            })

    online_steps = cal_epochs * cal_batches + hold_epochs * hold_batches + switch_epochs * switch_batches
    update_elements = int(online_steps * ntrainable)

    return {
        "condition": condition,
        "drift": drift_name,
        "fork_hash_ok": True,
        "trainable_params": int(ntrainable),
        "base_neutral_acc": base_neutral_acc,
        "base_drift_acc": base_drift_acc,
        "post_calibration_acc": cal_acc,
        "post_calibration_neutral_acc": cal_neutral_acc,
        "cal_gain": cal_gain,
        "explicit_checkpoint_oracle_acc": cal_acc,
        "post_calibration_state_sha256": sha_state(cal_state),
        "post_hold_state_sha256": sha_state(final_hold_state),
        "hold": hold_records,
        "switch": switch_records,
        "online_optimizer_steps": int(online_steps if opt is not None else 0),
        "trainable_parameter_element_updates": int(update_elements if opt is not None else 0),
    }


def run(seed, outdir, smoke=False):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    if smoke:
        ntrain, nval, ntest, ncal, nhold, nswitch = 512, 256, 512, 256, 256, 256
        pre_epochs, cal_epochs, hold_epochs, switch_epochs = 2, 1, 2, 1
        hold_checks, switch_checks = (1, 2), (1,)
    else:
        ntrain, nval, ntest, ncal, nhold, nswitch = PRETRAIN_N, VAL_N, TEST_N, CAL_N, HOLD_N, SWITCH_N
        pre_epochs, cal_epochs, hold_epochs, switch_epochs = PRETRAIN_EPOCHS, CAL_EPOCHS, HOLD_EPOCHS, SWITCH_EPOCHS
        hold_checks, switch_checks = HOLD_CHECKS, SWITCH_CHECKS

    x, ys, u, v = make_data(seed, ntrain, nval, ntest, ncal, nhold, nswitch)

    set_seed(derive_seed(seed, "init"))
    model = RecurrentClassifier()
    set_trainable(model, "FULL")
    opt = make_optimizer(model)
    for ep in range(1, pre_epochs + 1):
        train_epoch(model, opt, x["train0"], ys["train"], derive_seed(seed, f"pretrain_order_{ep}"))

    base_state = clone_state(model.state_dict())
    base_hash = sha_state(base_state)
    base_val_acc = accuracy(model, x["val0"], ys["val"])
    base_neutral_test_acc = accuracy(model, x["test0"], ys["test"])
    base_drift_acc = {d: accuracy(model, x[f"test_{d}"], ys["test"]) for d in DRIFTS}

    if smoke:
        paths = {}
        for d in DRIFTS:
            paths[d] = {}
            for c in CONDITIONS:
                paths[d][c] = run_condition(seed, d, c, base_state, base_hash, x, ys, cal_epochs=cal_epochs, hold_epochs=hold_epochs, hold_checks=hold_checks, switch_epochs=switch_epochs, switch_checks=switch_checks)
        save_json(out / "smoke_summary.json", {
            "status": "ok",
            "seed": int(seed),
            "base_hash": base_hash,
            "base_neutral_test_acc": base_neutral_test_acc,
            "conditions": {d: sorted(paths[d]) for d in paths},
            "hold_lengths": {d: {c: len(paths[d][c]["hold"]) for c in paths[d]} for d in paths},
            "switch_lengths": {d: {c: len(paths[d][c]["switch"]) for c in paths[d]} for d in paths},
        })
        return

    if seed not in BENCH_SEEDS:
        raise SystemExit(f"seed {seed} is not an E1 benchmark family")

    if base_neutral_test_acc < 0.80:
        save_json(out / "seed_summary.json", {
            "experiment": "E1-Persistent-Calibration",
            "seed": int(seed),
            "validity": {"base_neutral_accuracy": False, "fork_identity": True, "finite": True, "complete": False},
            "base": {"state_sha256": base_hash, "validation_acc": base_val_acc, "neutral_test_acc": base_neutral_test_acc, "drift_test_acc": base_drift_acc},
        })
        return

    results = {d: {} for d in DRIFTS}
    fork_ok = True
    for d in DRIFTS:
        for c in CONDITIONS:
            r = run_condition(seed, d, c, base_state, base_hash, x, ys)
            fork_ok = fork_ok and bool(r["fork_hash_ok"])
            results[d][c] = r
            print(f"seed={seed} drift={d} cond={c} base={r['base_drift_acc']:.4f} cal={r['post_calibration_acc']:.4f} hold120={r['hold'][-1]['return_acc']:.4f}", flush=True)

    complete = all(len(results[d][c]["hold"]) == len(HOLD_CHECKS) and len(results[d][c]["switch"]) == len(SWITCH_CHECKS) for d in DRIFTS for c in CONDITIONS)
    finite = finite_obj(results)

    counts = {}
    for c in CONDITIONS:
        m = RecurrentClassifier()
        set_trainable(m, c)
        counts[c] = trainable_count(m)

    summary = {
        "experiment": "E1-Persistent-Calibration",
        "seed": int(seed),
        "benchmark_seed": True,
        "task": {"gamma": GAMMA, "noise_std": NOISE_STD, "u": u.tolist(), "v": v.tolist(), "steps": STEPS},
        "base": {"state_sha256": base_hash, "validation_acc": base_val_acc, "neutral_test_acc": base_neutral_test_acc, "drift_test_acc": base_drift_acc},
        "trainable_parameter_counts": counts,
        "results": results,
        "validity": {"base_neutral_accuracy": bool(base_neutral_test_acc >= 0.80), "fork_identity": bool(fork_ok), "finite": bool(finite), "complete": bool(complete)},
    }
    summary["all_valid"] = bool(all(summary["validity"].values()))
    save_json(out / "seed_summary.json", summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run(args.seed, args.outdir, smoke=args.smoke)


if __name__ == "__main__":
    main()
