import argparse
import csv
import hashlib
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

FAMILY_SEEDS = (3, 9, 21, 28, 44, 62, 86, 101)
CONDITIONS = ("B", "P", "D", "O")
CHECKPOINT_EPOCHS = (0, 1, 2, 5, 10, 20, 40, 60, 100)
PI = np.array([3, 4, 5, 6, 7, 0, 1, 2], dtype=np.int64)
SIZES = {"train": 20000, "val": 2500, "test": 5000}
PAIR_N = 2048
BATCH = 256
LR = 1e-3
WD = 1e-4
EPOCHS = 100
CLIP = 1.0
GRAD_DIAG_N = 2048


def derive_seed(seed, name):
    h = hashlib.sha256(f"R8-M2|{seed}|{name}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def sha_tensor(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def sha_state_dict(sd, keys=None):
    h = hashlib.sha256()
    use = sorted(sd.keys()) if keys is None else sorted(keys)
    for k in use:
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


def clone_state(sd):
    return {k: v.detach().cpu().clone() for k, v in sd.items()}


def relation_specific_keys(sd):
    keys = ["rel_emb.weight"]
    keys += [k for k in sd if k.startswith("head0.") or k.startswith("headT.")]
    return sorted(keys)


def shared_keys(sd):
    rel = set(relation_specific_keys(sd))
    return sorted([k for k in sd if k not in rel])


def permute_relation_bundle(sd):
    out = clone_state(sd)
    p = torch.tensor(PI, dtype=torch.long)
    out["rel_emb.weight"] = sd["rel_emb.weight"][p].clone()
    for r in range(N_REL):
        src = int(PI[r])
        for prefix in ("head0", "headT"):
            out[f"{prefix}.{r}.weight"] = sd[f"{prefix}.{src}.weight"].clone()
            out[f"{prefix}.{r}.bias"] = sd[f"{prefix}.{src}.bias"].clone()
    return out


def map_data(y_source, condition):
    if condition == "D":
        return y_source[:, torch.tensor(PI, dtype=torch.long)]
    return y_source


def load_from_state(sd):
    m = Core()
    m.load_state_dict(clone_state(sd))
    return m


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
    state = clone_state(model.state_dict())
    torch.save({"state_dict": state, "epoch": epoch, "condition": condition, "validation": validation}, path)


def train_condition(seed, condition, init_state, train_source, val_source, val_perm, out, epochs=EPOCHS):
    model = load_from_state(init_state)
    train_y = map_data(train_source, condition)
    val_y = map_data(val_source, condition)
    trainable = list(model.parameters())
    opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=WD)
    ce = nn.CrossEntropyLoss()
    history = []

    v0 = eval_heads(model, val_y, val_perm)
    history.append({"epoch": 0, **v0})
    save_checkpoint(model, out / f"{condition}_epoch_000.pt", 0, condition, v0)
    checkpoint_set = set(e for e in CHECKPOINT_EPOCHS if e <= epochs)

    for ep in range(1, epochs + 1):
        model.train()
        train_perm = make_perms(len(train_y), derive_seed(seed, f"presentation_perm_{ep}"))
        order_name = f"batch_order_alt_{ep}" if condition == "O" else f"batch_order_common_{ep}"
        g = torch.Generator().manual_seed(derive_seed(seed, order_name))
        order = torch.randperm(len(train_y), generator=g)
        total = 0.0
        seen = 0
        for a in range(0, len(train_y), BATCH):
            ix = order[a:a+BATCH]
            y = train_y[ix]
            p = train_perm[ix]
            opt.zero_grad(set_to_none=True)
            _, l0, lT = model(y, p)
            loss = sum(
                ce(l0[r], y[:, r]) + ce(lT[r], y[:, r])
                for r in range(N_REL)
            ) / N_REL
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, CLIP)
            opt.step()
            total += float(loss.item()) * len(ix)
            seen += len(ix)
        val = eval_heads(model, val_y, val_perm)
        history.append({"epoch": ep, "train_loss": total / seen, **val})
        print(
            f"seed={seed} cond={condition} epoch={ep:03d} loss={total/seen:.6f} "
            f"h0={val['h0_overall']:.6f} h12={val['h12_overall']:.6f}",
            flush=True,
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


def make_pair_bank(seed, condition, n=PAIR_N):
    bank = []
    for model_r in range(N_REL):
        source_r = int(PI[model_r]) if condition == "D" else model_r
        base_source = make_memories(n, derive_seed(seed, f"pair_source_memory_{source_r}"))
        alt_source = base_source.clone()
        g = torch.Generator().manual_seed(derive_seed(seed, f"pair_source_offset_{source_r}"))
        offset = torch.randint(1, N_VAL, (n,), generator=g)
        alt_source[:, source_r] = (alt_source[:, source_r] + offset) % N_VAL
        base = map_data(base_source, condition)
        alt = map_data(alt_source, condition)
        perms = make_perms(n, derive_seed(seed, f"pair_presentation_source_{source_r}"))
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
        "relation_log_median_survival_terminal": logs.tolist(),
        "G": float(np.std(logs, ddof=0)),
        "C": float(np.mean(np.log(np.clip(s[:, :, -1], 1e-12, None)))),
        "winner_relation": int(np.argmax(terminal_meds)),
    }


def initial_shared_gradient_predictor(seed, init_state, train_source):
    model = load_from_state(init_state)
    model.train()
    n = min(GRAD_DIAG_N, len(train_source))
    y = train_source[:n]
    perms = make_perms(n, derive_seed(seed, "gradient_diag_presentation"))
    ce = nn.CrossEntropyLoss()
    shared_modules = (model.enc, model.to_h, model.F)
    shared_params = []
    for module in shared_modules:
        shared_params.extend(list(module.parameters()))

    grad_norms = []
    losses = []
    for r in range(N_REL):
        model.zero_grad(set_to_none=True)
        _, l0, lT = model(y, perms)
        loss = ce(l0[r], y[:, r]) + ce(lT[r], y[:, r])
        loss.backward()
        sq = 0.0
        for p in shared_params:
            if p.grad is not None:
                sq += float(torch.sum(p.grad.detach() ** 2))
        grad_norms.append(float(np.sqrt(sq)))
        losses.append(float(loss.item()))

    sd = init_state
    emb_norms = torch.linalg.vector_norm(sd["rel_emb.weight"], dim=1).numpy().tolist()
    h0_head_norms = []
    hT_head_norms = []
    for r in range(N_REL):
        h0_head_norms.append(float(torch.linalg.vector_norm(sd[f"head0.{r}.weight"])))
        hT_head_norms.append(float(torch.linalg.vector_norm(sd[f"headT.{r}.weight"])))

    return {
        "diagnostic_n": n,
        "shared_gradient_norm_by_relation": grad_norms,
        "joint_relation_loss_by_relation": losses,
        "relation_embedding_norm": emb_norms,
        "h0_head_weight_norm": h0_head_norms,
        "h12_head_weight_norm": hT_head_norms,
    }


def verify_initialization(base_state, perm_state):
    sk = shared_keys(base_state)
    if sha_state_dict(base_state, sk) != sha_state_dict(perm_state, sk):
        raise RuntimeError("P shared parameters differ from baseline")
    expected = permute_relation_bundle(base_state)
    if sha_state_dict(expected) != sha_state_dict(perm_state):
        raise RuntimeError("P relation-specific permutation mismatch")
    return {
        "baseline_full_sha256": sha_state_dict(base_state),
        "baseline_shared_sha256": sha_state_dict(base_state, sk),
        "permuted_full_sha256": sha_state_dict(perm_state),
        "permuted_shared_sha256": sha_state_dict(perm_state, sk),
    }


def run(seed, outdir, smoke=False):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    ntrain = 256 if smoke else SIZES["train"]
    nval = 128 if smoke else SIZES["val"]
    epochs = 2 if smoke else EPOCHS
    pair_n = 64 if smoke else PAIR_N

    train_source = make_memories(ntrain, derive_seed(seed, "source_memory_train"))
    val_source = make_memories(nval, derive_seed(seed, "source_memory_val"))
    val_perm = make_perms(nval, derive_seed(seed, "validation_presentation"))

    set_seed(derive_seed(seed, "baseline_shared_init"))
    base_model = Core()
    base_state = clone_state(base_model.state_dict())
    p_state = permute_relation_bundle(base_state)
    init_verification = verify_initialization(base_state, p_state)

    condition_states = {
        "B": base_state,
        "P": p_state,
        "D": base_state,
        "O": base_state,
    }

    gradient_predictor = initial_shared_gradient_predictor(seed, base_state, train_source)

    histories = {}
    for cond in CONDITIONS:
        histories[cond] = train_condition(
            seed, cond, condition_states[cond], train_source, val_source, val_perm, out, epochs=epochs
        )

    init_hashes = {}
    for cond in CONDITIONS:
        ck = torch.load(out / f"{cond}_epoch_000.pt", map_location="cpu", weights_only=False)
        init_hashes[cond] = sha_state_dict(ck["state_dict"])

    if not (init_hashes["B"] == init_hashes["D"] == init_hashes["O"]):
        raise RuntimeError("B/D/O epoch-0 hashes are not identical")
    if init_hashes["P"] != init_verification["permuted_full_sha256"]:
        raise RuntimeError("P epoch-0 hash does not match verified bundle permutation")

    if smoke:
        save_json(
            out / "smoke_summary.json",
            {
                "seed": seed,
                "status": "ok",
                "pi": PI.tolist(),
                "init_verification": init_verification,
                "epoch0_hashes": init_hashes,
            },
        )
        return

    checkpoint_analysis = {}
    epoch100_arrays = {}
    for cond in CONDITIONS:
        bank = make_pair_bank(seed, cond, pair_n)
        rows = []
        for ep in CHECKPOINT_EPOCHS:
            model, ck = load_model(out / f"{cond}_epoch_{ep:03d}.pt")
            s = pair_survival(model, bank)
            sm = summarize_survival(s)
            rows.append({"epoch": ep, "validation": ck["validation"], **sm})
            if ep == 100:
                epoch100_arrays[cond] = s
        checkpoint_analysis[cond] = rows
        save_json(out / f"{cond}_checkpoint_analysis.json", rows)

    np.savez_compressed(out / "epoch100_pair_survival.npz", **epoch100_arrays)

    final = {c: checkpoint_analysis[c][-1] for c in CONDITIONS}
    validity_by_condition = {
        c: bool(final[c]["validation"]["combined"] >= 0.38 and final[c]["validation"]["h0_overall"] >= 0.55)
        for c in CONDITIONS
    }

    seed_summary = {
        "experiment": "R8-M2",
        "family_seed": seed,
        "fresh_family_seed": seed in FAMILY_SEEDS,
        "pi": PI.tolist(),
        "source_train_sha256": sha_tensor(train_source),
        "source_val_sha256": sha_tensor(val_source),
        "init_verification": init_verification,
        "epoch0_condition_hashes": init_hashes,
        "validity_by_condition": validity_by_condition,
        "all_conditions_valid": bool(all(validity_by_condition.values())),
        "initial_shared_gradient_predictor": gradient_predictor,
        "checkpoint_analysis": checkpoint_analysis,
        "final_conditions": final,
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
                w.writerow([
                    cond,
                    row["epoch"],
                    v["h0_overall"],
                    v["h12_overall"],
                    v["combined"],
                    row["G"],
                    row["C"],
                    row["winner_relation"],
                ])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run(args.seed, args.outdir, args.smoke)
