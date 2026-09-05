import argparse
import csv
import hashlib
import io
import json
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

N_REL = 8
N_VAL = 16
EMB = 8
HIDDEN = 32
LATENT = 16
STEPS = 12
EPS = 1e-8

PRIMARY_SEEDS = (11, 37, 71)
CONDITIONS = ("J", "H0", "HT", "FF", "EF")
CHECKPOINT_EPOCHS = (0, 2, 10, 20, 40, 60, 100)
SIZES = {"train": 20000, "val": 2500, "test": 5000}
PAIR_N = 2048
BATCH = 256
LR = 1e-3
WD = 1e-4
EPOCHS = 100
CLIP = 1.0
BOOT_N = 5000
RIDGE_LAMBDA = 1e-3


def derive_seed(seed, name):
    h = hashlib.sha256(f"R8-M1|{seed}|{name}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_tensor(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def sha_state_dict(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def make_memories(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, N_VAL, (n, N_REL), generator=g)


def make_perms(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.stack([torch.randperm(N_REL, generator=g) for _ in range(n)])


def init_linear_embedding(m):
    if isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def init_gru(gru):
    for name, p in gru.named_parameters():
        if "weight_ih" in name:
            nn.init.xavier_uniform_(p)
        elif "weight_hh" in name:
            for block in p.chunk(3, 0):
                nn.init.orthogonal_(block)
        elif "bias" in name:
            nn.init.zeros_(p)


class Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.rel_emb = nn.Embedding(N_REL, EMB)
        self.val_emb = nn.Embedding(N_VAL, EMB)
        self.enc = nn.GRU(EMB * 2, HIDDEN, batch_first=True)
        self.to_h = nn.Linear(HIDDEN, LATENT)
        self.F = nn.Sequential(
            nn.Linear(LATENT, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, LATENT),
            nn.Tanh(),
        )
        self.head0 = nn.ModuleList([nn.Linear(LATENT, N_VAL) for _ in range(N_REL)])
        self.headT = nn.ModuleList([nn.Linear(LATENT, N_VAL) for _ in range(N_REL)])
        self.apply(init_linear_embedding)
        init_gru(self.enc)

    def encode(self, y, perms):
        b = len(y)
        rel = torch.arange(N_REL, device=y.device).expand(b, -1).gather(1, perms)
        val = y.gather(1, perms)
        x = torch.cat([self.rel_emb(rel), self.val_emb(val)], -1)
        _, s = self.enc(x)
        return torch.tanh(self.to_h(s[-1]))

    def trajectory(self, h0):
        hs = [h0]
        h = h0
        for _ in range(STEPS):
            h = self.F(h)
            hs.append(h)
        return torch.stack(hs, 1)

    def forward(self, y, perms):
        h0 = self.encode(y, perms)
        tr = self.trajectory(h0)
        return tr, [h(h0) for h in self.head0], [h(tr[:, -1]) for h in self.headT]


def clone_shared_model(shared_state):
    m = Core()
    m.load_state_dict({k: v.clone() for k, v in shared_state.items()})
    return m


def configure_condition(model, condition):
    for p in model.parameters():
        p.requires_grad = False

    def enable(module):
        for p in module.parameters():
            p.requires_grad = True

    if condition == "J":
        for p in model.parameters():
            p.requires_grad = True
    elif condition == "H0":
        for m in (model.rel_emb, model.val_emb, model.enc, model.to_h, model.head0):
            enable(m)
    elif condition == "HT":
        for m in (model.rel_emb, model.val_emb, model.enc, model.to_h, model.F, model.headT):
            enable(m)
    elif condition == "FF":
        for m in (model.rel_emb, model.val_emb, model.enc, model.to_h, model.head0, model.headT):
            enable(m)
    elif condition == "EF":
        for m in (model.F, model.head0, model.headT):
            enable(m)
    else:
        raise ValueError(condition)


def condition_loss(condition, l0, lT, y, ce):
    if condition == "H0":
        return sum(ce(l0[r], y[:, r]) for r in range(N_REL)) / N_REL
    if condition == "HT":
        return sum(ce(lT[r], y[:, r]) for r in range(N_REL)) / N_REL
    return sum(ce(l0[r], y[:, r]) + ce(lT[r], y[:, r]) for r in range(N_REL)) / N_REL


def eval_heads(model, y, perms):
    model.eval()
    c0 = np.zeros(N_REL, dtype=np.int64)
    cT = np.zeros(N_REL, dtype=np.int64)
    with torch.no_grad():
        for a in range(0, len(y), BATCH):
            yy = y[a:a+BATCH]
            pp = perms[a:a+BATCH]
            _, l0, lT = model(yy, pp)
            for r in range(N_REL):
                c0[r] += int((l0[r].argmax(1) == yy[:, r]).sum())
                cT[r] += int((lT[r].argmax(1) == yy[:, r]).sum())
    a0 = c0 / len(y)
    aT = cT / len(y)
    return {
        "h0_overall": float(a0.mean()),
        "h12_overall": float(aT.mean()),
        "combined": float((a0.mean() + aT.mean()) / 2.0),
        "h0_per_relation": a0.tolist(),
        "h12_per_relation": aT.tolist(),
    }


def save_checkpoint(model, path, epoch, condition, validation):
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save({"state_dict": state, "epoch": epoch, "condition": condition, "validation": validation}, path)


def train_condition(seed, condition, shared_state, train_y, val_y, val_perm, out, epochs=EPOCHS):
    model = clone_shared_model(shared_state)
    configure_condition(model, condition)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=WD)
    ce = nn.CrossEntropyLoss()
    history = []

    v0 = eval_heads(model, val_y, val_perm)
    history.append({"epoch": 0, **v0})
    save_checkpoint(model, out / f"{condition}_epoch_000.pt", 0, condition, v0)

    checkpoint_set = set(e for e in CHECKPOINT_EPOCHS if e <= epochs)
    for ep in range(1, epochs + 1):
        model.train()
        train_perm = make_perms(len(train_y), derive_seed(seed, f"perm_train_{ep}"))
        g = torch.Generator().manual_seed(derive_seed(seed, f"batch_{ep}"))
        order = torch.randperm(len(train_y), generator=g)
        total = 0.0
        seen = 0
        for a in range(0, len(train_y), BATCH):
            ix = order[a:a+BATCH]
            y = train_y[ix]
            p = train_perm[ix]
            opt.zero_grad(set_to_none=True)
            _, l0, lT = model(y, p)
            loss = condition_loss(condition, l0, lT, y, ce)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, CLIP)
            opt.step()
            total += float(loss.item()) * len(ix)
            seen += len(ix)
        val = eval_heads(model, val_y, val_perm)
        history.append({"epoch": ep, "train_loss": total / seen, **val})
        print(
            f"seed={seed} cond={condition} epoch={ep:03d} loss={total/seen:.6f} "
            f"h0={val['h0_overall']:.6f} h12={val['h12_overall']:.6f}", flush=True
        )
        if ep in checkpoint_set:
            save_checkpoint(model, out / f"{condition}_epoch_{ep:03d}.pt", ep, condition, val)
    save_json(out / f"{condition}_training_history.json", history)
    return history


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = Core()
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck


def make_pair_bank(seed, n=PAIR_N):
    bank = []
    for r in range(N_REL):
        base = make_memories(n, derive_seed(seed, f"pair_memory_relation_{r}"))
        alt = base.clone()
        g = torch.Generator().manual_seed(derive_seed(seed, f"pair_offset_relation_{r}"))
        offset = torch.randint(1, N_VAL, (n,), generator=g)
        alt[:, r] = (alt[:, r] + offset) % N_VAL
        perms = make_perms(n, derive_seed(seed, f"pair_perm_relation_{r}"))
        bank.append((base, alt, perms))
    return bank


def pair_survival(model, bank):
    all_rel = []
    with torch.no_grad():
        for base, alt, perms in bank:
            chunks = []
            for a in range(0, len(base), BATCH):
                p = perms[a:a+BATCH]
                ta = model.trajectory(model.encode(base[a:a+BATCH], p))
                tb = model.trajectory(model.encode(alt[a:a+BATCH], p))
                d = torch.linalg.vector_norm(ta - tb, dim=-1)
                chunks.append((d / (d[:, :1] + EPS)).cpu())
            all_rel.append(torch.cat(chunks, 0))
    return torch.stack(all_rel, 0).numpy().astype(np.float32)


def summarize_survival(s):
    med_by_time = np.median(s, axis=1)
    terminal_meds = med_by_time[:, -1]
    logs = np.log(np.clip(terminal_meds, 1e-12, None))
    return {
        "relation_median_survival_by_time": med_by_time.tolist(),
        "relation_median_survival_terminal": terminal_meds.tolist(),
        "G": float(np.std(logs, ddof=0)),
        "C": float(np.mean(np.log(np.clip(s[:, :, -1], 1e-12, None)))),
        "winner_relation": int(np.argmax(terminal_meds)),
    }


def bootstrap_G_difference(seed, name, a, b, nboot=BOOT_N):
    # a,b shape [relation,pair,time]
    at = a[:, :, -1]
    bt = b[:, :, -1]
    n = at.shape[1]
    rng = np.random.default_rng(derive_seed(seed, f"{name}|bootstrap"))
    vals = np.empty(nboot, dtype=np.float64)

    def G_from_terminal(x):
        meds = np.median(x, axis=1)
        return float(np.std(np.log(np.clip(meds, 1e-12, None)), ddof=0))

    obs = G_from_terminal(at) - G_from_terminal(bt)
    for i in range(nboot):
        ix = rng.integers(0, n, size=(N_REL, n))
        ra = np.take_along_axis(at, ix, axis=1)
        rb = np.take_along_axis(bt, ix, axis=1)
        vals[i] = G_from_terminal(ra) - G_from_terminal(rb)
    return {
        "observed": float(obs),
        "ci95_percentile": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))],
        "n_bootstrap": nboot,
    }


def one_hot_targets(y):
    yn = y.cpu().numpy()
    return np.eye(N_VAL, dtype=np.float64)[yn].reshape(len(yn), N_REL * N_VAL)


def trajectory(model, y, perms):
    chunks = []
    with torch.no_grad():
        for a in range(0, len(y), BATCH):
            h0 = model.encode(y[a:a+BATCH], perms[a:a+BATCH])
            chunks.append(model.trajectory(h0).cpu())
    return torch.cat(chunks, 0)


def ridge_score(train_x, train_y_hot, val_x, val_y, lam=RIDGE_LAMBDA):
    x = np.asarray(train_x, dtype=np.float64)
    z = np.asarray(val_x, dtype=np.float64)
    x = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    z = np.concatenate([z, np.ones((len(z), 1))], axis=1)
    reg = np.eye(x.shape[1]) * lam
    reg[-1, -1] = 0.0
    w = np.linalg.solve(x.T @ x + reg, x.T @ train_y_hot)
    scores = (z @ w).reshape(len(z), N_REL, N_VAL)
    pred = scores.argmax(-1)
    truth = val_y.cpu().numpy()
    corr = pred == truth
    return {"overall": float(corr.mean()), "per_relation": corr.mean(0).tolist()}


def ridge_accessibility(model, train_y, train_perm, val_y, val_perm):
    tr = trajectory(model, train_y, train_perm)
    va = trajectory(model, val_y, val_perm)
    hot = one_hot_targets(train_y)
    return {
        "h0": ridge_score(tr[:, 0].numpy(), hot, va[:, 0].numpy(), val_y),
        "h12": ridge_score(tr[:, -1].numpy(), hot, va[:, -1].numpy(), val_y),
    }


def native_geometry(model, val_y, val_perm):
    tr = trajectory(model, val_y, val_perm)
    dh = tr[:, 1:] - tr[:, :-1]
    speed = torch.linalg.vector_norm(dh, dim=-1)
    radius = torch.linalg.vector_norm(tr, dim=-1)
    direction = dh / (speed.unsqueeze(-1) + EPS)
    turn = torch.nn.functional.cosine_similarity(direction[:, :-1], direction[:, 1:], dim=-1)
    path = speed.sum(1)
    endpoint = torch.linalg.vector_norm(tr[:, -1] - tr[:, 0], dim=-1)
    eff = endpoint / (path + EPS)
    return {
        "path_length_mean": float(path.mean()),
        "endpoint_displacement_mean": float(endpoint.mean()),
        "endpoint_path_efficiency_mean": float(eff.mean()),
        "radius_mean_by_time": radius.mean(0).tolist(),
        "speed_mean_by_transition": speed.mean(0).tolist(),
        "turn_cosine_mean_by_turn": turn.mean(0).tolist(),
        "reversal_fraction": float((turn < 0).float().mean()),
    }


def run(seed, outdir, smoke=False):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    ntrain = 256 if smoke else SIZES["train"]
    nval = 128 if smoke else SIZES["val"]
    epochs = 2 if smoke else EPOCHS
    pair_n = 64 if smoke else PAIR_N

    train_y = make_memories(ntrain, derive_seed(seed, "memory_train"))
    val_y = make_memories(nval, derive_seed(seed, "memory_val"))
    val_perm = make_perms(nval, derive_seed(seed, "perm_val"))

    set_seed(derive_seed(seed, "shared_init"))
    shared = Core()
    shared_state = {k: v.detach().cpu().clone() for k, v in shared.state_dict().items()}
    shared_hash = sha_state_dict(shared_state)

    all_hist = {}
    for cond in CONDITIONS:
        all_hist[cond] = train_condition(seed, cond, shared_state, train_y, val_y, val_perm, out, epochs=epochs)

    # Verify epoch-0 state identity.
    init_hashes = {}
    for cond in CONDITIONS:
        ck = torch.load(out / f"{cond}_epoch_000.pt", map_location="cpu", weights_only=False)
        init_hashes[cond] = sha_state_dict(ck["state_dict"])
    if len(set(init_hashes.values())) != 1 or next(iter(init_hashes.values())) != shared_hash:
        raise RuntimeError("paired initialization hash mismatch")

    if smoke:
        save_json(out / "smoke_summary.json", {"seed": seed, "shared_init_sha256": shared_hash, "status": "ok"})
        return

    bank = make_pair_bank(seed, pair_n)
    checkpoint_analysis = {}
    epoch100_arrays = {}
    for cond in CONDITIONS:
        checkpoint_analysis[cond] = []
        for ep in CHECKPOINT_EPOCHS:
            model, ck = load_model(out / f"{cond}_epoch_{ep:03d}.pt")
            s = pair_survival(model, bank)
            sm = summarize_survival(s)
            checkpoint_analysis[cond].append({"epoch": ep, "validation": ck["validation"], **sm})
            if ep == 100:
                epoch100_arrays[cond] = s
        save_json(out / f"{cond}_checkpoint_analysis.json", checkpoint_analysis[cond])

    np.savez_compressed(out / "epoch100_pair_survival.npz", **epoch100_arrays)

    contrasts = {
        "H1_HT_minus_H0": bootstrap_G_difference(seed, "H1_HT_minus_H0", epoch100_arrays["HT"], epoch100_arrays["H0"]),
        "H2_J_minus_FF": bootstrap_G_difference(seed, "H2_J_minus_FF", epoch100_arrays["J"], epoch100_arrays["FF"]),
        "H3_J_minus_EF": bootstrap_G_difference(seed, "H3_J_minus_EF", epoch100_arrays["J"], epoch100_arrays["EF"]),
        "secondary_H0_minus_epoch0": bootstrap_G_difference(seed, "secondary_H0_minus_epoch0", epoch100_arrays["H0"], pair_survival(load_model(out / "J_epoch_000.pt")[0], bank)),
        "secondary_HT_minus_J": bootstrap_G_difference(seed, "secondary_HT_minus_J", epoch100_arrays["HT"], epoch100_arrays["J"]),
    }

    # Ridge secondary. Training relation-order permutation is fixed independently for this diagnostic.
    ridge_train_perm = make_perms(len(train_y), derive_seed(seed, "ridge_perm_train"))
    ridge = {}
    for cond in CONDITIONS:
        model, _ = load_model(out / f"{cond}_epoch_100.pt")
        ridge[cond] = ridge_accessibility(model, train_y, ridge_train_perm, val_y, val_perm)

    j_model, _ = load_model(out / "J_epoch_100.pt")
    geom = native_geometry(j_model, val_y, val_perm)

    final = {c: checkpoint_analysis[c][-1] for c in CONDITIONS}
    jv = final["J"]["validation"]
    validity = bool(jv["combined"] >= 0.38 and jv["h0_overall"] >= 0.55)

    seed_summary = {
        "experiment": "R8-M1",
        "seed": seed,
        "fresh_primary_seed": seed in PRIMARY_SEEDS,
        "shared_init_sha256": shared_hash,
        "epoch0_condition_hashes": init_hashes,
        "baseline_validity": validity,
        "final_conditions": final,
        "primary_contrasts": contrasts,
        "ridge_accessibility": ridge,
        "joint_native_geometry": geom,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    save_json(out / "seed_summary.json", seed_summary)

    with (out / "checkpoint_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "epoch", "h0", "h12", "combined", "G", "C", "winner_relation"])
        for cond in CONDITIONS:
            for row in checkpoint_analysis[cond]:
                v = row["validation"]
                w.writerow([cond, row["epoch"], v["h0_overall"], v["h12_overall"], v["combined"], row["G"], row["C"], row["winner_relation"]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run(args.seed, args.outdir, args.smoke)
