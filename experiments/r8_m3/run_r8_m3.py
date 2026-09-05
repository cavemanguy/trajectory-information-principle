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
FAMILY_SEEDS = (14, 24, 34, 47, 58, 73, 89, 107, 116, 127, 139, 151)
CONDITIONS = ("B", "O")
CHECKPOINT_EPOCHS = (0, 1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 80, 100)
SIZES = {"train": 20000, "val": 2500, "test": 5000}
PAIR_N = 2048
GRAD_DIAG_N = 1024
BATCH = 256
LR = 1e-3
WD = 1e-4
EPOCHS = 100
CLIP = 1.0


def derive_seed(seed, name):
    h = hashlib.sha256(f"R8-M3|{seed}|{name}".encode()).digest()
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
        self.F = nn.Sequential(nn.Linear(LATENT, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, LATENT), nn.Tanh())
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
    return {"h0_overall": float(a0.mean()), "h12_overall": float(aT.mean()), "combined": float((a0.mean() + aT.mean()) / 2.0), "h0_per_relation": a0.tolist(), "h12_per_relation": aT.tolist()}


def save_checkpoint(model, path, epoch, condition, validation):
    torch.save({"state_dict": clone_state(model.state_dict()), "epoch": epoch, "condition": condition, "validation": validation}, path)


def train_condition(seed, condition, init_state, train_y, val_y, val_perm, out, epochs=EPOCHS):
    model = load_from_state(init_state)
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
            loss = sum(nn.functional.cross_entropy(l0[r], y[:, r]) + nn.functional.cross_entropy(lT[r], y[:, r]) for r in range(N_REL)) / N_REL
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, CLIP)
            opt.step()
            total += float(loss.item()) * len(ix)
            seen += len(ix)
        val = eval_heads(model, val_y, val_perm)
        history.append({"epoch": ep, "train_loss": total / seen, **val})
        print(f"seed={seed} cond={condition} epoch={ep:03d} loss={total/seen:.6f} h0={val['h0_overall']:.6f} h12={val['h12_overall']:.6f}", flush=True)
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
        base = make_memories(n, derive_seed(seed, f"pair_memory_{r}"))
        alt = base.clone()
        g = torch.Generator().manual_seed(derive_seed(seed, f"pair_offset_{r}"))
        offset = torch.randint(1, N_VAL, (n,), generator=g)
        alt[:, r] = (alt[:, r] + offset) % N_VAL
        perms = make_perms(n, derive_seed(seed, f"pair_presentation_{r}"))
        bank.append((base, alt, perms))
    return bank


def pair_survival_components(encoder_model, recurrent_model, bank):
    all_rel = []
    encoder_model.eval()
    recurrent_model.eval()
    with torch.no_grad():
        for base, alt, perms in bank:
            chunks = []
            for a in range(0, len(base), BATCH):
                p = perms[a:a+BATCH]
                h0a = encoder_model.encode(base[a:a+BATCH], p)
                h0b = encoder_model.encode(alt[a:a+BATCH], p)
                ta = recurrent_model.trajectory(h0a)
                tb = recurrent_model.trajectory(h0b)
                d = torch.linalg.vector_norm(ta - tb, dim=-1)
                chunks.append((d / (d[:, :1] + EPS)).cpu())
            all_rel.append(torch.cat(chunks, 0))
    return torch.stack(all_rel, 0).numpy().astype(np.float32)


def summarize_survival(s):
    med_by_time = np.median(s, axis=1)
    terminal_meds = med_by_time[:, -1]
    logs = np.log(np.clip(terminal_meds, 1e-12, None))
    return {"relation_median_survival_by_time": med_by_time.tolist(), "relation_median_survival_terminal": terminal_meds.tolist(), "relation_log_median_survival_terminal": logs.tolist(), "G": float(np.std(logs, ddof=0)), "winner_relation": int(np.argmax(terminal_meds))}


def shared_gradient_alignment(model, y, perms):
    model.train()
    shared_params = list(model.enc.parameters()) + list(model.to_h.parameters()) + list(model.F.parameters())
    grad_vectors = []
    losses = []
    for r in range(N_REL):
        model.zero_grad(set_to_none=True)
        _, l0, lT = model(y, perms)
        loss = nn.functional.cross_entropy(l0[r], y[:, r]) + nn.functional.cross_entropy(lT[r], y[:, r])
        loss.backward()
        pieces = [torch.zeros_like(p).reshape(-1) if p.grad is None else p.grad.detach().reshape(-1).clone() for p in shared_params]
        grad_vectors.append(torch.cat(pieces))
        losses.append(float(loss.item()))
    G = torch.stack(grad_vectors, 0)
    g_all = G.mean(0)
    g_all_norm = torch.linalg.vector_norm(g_all)
    norms = torch.linalg.vector_norm(G, dim=1)
    align = (G @ g_all) / (norms * g_all_norm + EPS)
    cos = (G @ G.T) / (norms[:, None] * norms[None, :] + EPS)
    support = (cos.sum(1) - 1.0) / (N_REL - 1)
    model.eval()
    return {"alignment_to_total": align.cpu().numpy().tolist(), "gradient_norm": norms.cpu().numpy().tolist(), "joint_relation_loss": losses, "mean_pairwise_cosine_support": support.cpu().numpy().tolist()}


def analyze_condition(seed, condition, out, bank, init_model, train_y):
    rows = []
    grad_n = min(GRAD_DIAG_N, len(train_y))
    grad_y = train_y[:grad_n]
    grad_perms = make_perms(grad_n, derive_seed(seed, "gradient_diag_presentation"))
    for ep in CHECKPOINT_EPOCHS:
        current, ck = load_model(out / f"{condition}_epoch_{ep:03d}.pt")
        matched = summarize_survival(pair_survival_components(current, current, bank))
        enc_only = summarize_survival(pair_survival_components(current, init_model, bank))
        rec_only = summarize_survival(pair_survival_components(init_model, current, bank))
        lm = np.asarray(matched["relation_log_median_survival_terminal"], dtype=np.float64)
        le = np.asarray(enc_only["relation_log_median_survival_terminal"], dtype=np.float64)
        lf = np.asarray(rec_only["relation_log_median_survival_terminal"], dtype=np.float64)
        synergy = lm - 0.5 * (le + lf)
        grad = shared_gradient_alignment(current, grad_y, grad_perms)
        rows.append({"epoch": ep, "validation": ck["validation"], "matched": matched, "encoder_current_F0": enc_only, "encoder0_F_current": rec_only, "coadaptation_synergy": synergy.tolist(), "synergy_winner_relation": int(np.argmax(synergy)), "shared_gradient": grad, "gradient_alignment_winner_relation": int(np.argmax(np.asarray(grad["alignment_to_total"])))})
    return rows


def run(seed, outdir, smoke=False):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    ntrain = 512 if smoke else SIZES["train"]
    nval = 128 if smoke else SIZES["val"]
    epochs = 2 if smoke else EPOCHS
    pair_n = 64 if smoke else PAIR_N
    train_y = make_memories(ntrain, derive_seed(seed, "memory_train"))
    val_y = make_memories(nval, derive_seed(seed, "memory_val"))
    val_perm = make_perms(nval, derive_seed(seed, "validation_presentation"))
    set_seed(derive_seed(seed, "shared_init"))
    base_model = Core()
    init_state = clone_state(base_model.state_dict())
    init_sha = sha_state_dict(init_state)
    for cond in CONDITIONS:
        train_condition(seed, cond, init_state, train_y, val_y, val_perm, out, epochs=epochs)
    init_hashes = {}
    for cond in CONDITIONS:
        ck = torch.load(out / f"{cond}_epoch_000.pt", map_location="cpu", weights_only=False)
        init_hashes[cond] = sha_state_dict(ck["state_dict"])
    if not (init_hashes["B"] == init_hashes["O"] == init_sha):
        raise RuntimeError("B/O epoch-0 initialization mismatch")
    if smoke:
        bank = make_pair_bank(seed, pair_n)
        init_model = load_from_state(init_state)
        s = pair_survival_components(init_model, init_model, bank)
        grad_y = train_y[:min(64, len(train_y))]
        grad_p = make_perms(len(grad_y), derive_seed(seed, "gradient_diag_presentation"))
        grad = shared_gradient_alignment(init_model, grad_y, grad_p)
        save_json(out / "smoke_summary.json", {"seed": seed, "status": "ok", "init_sha256": init_sha, "epoch0_hashes": init_hashes, "survival_shape": list(s.shape), "gradient_alignment_len": len(grad["alignment_to_total"])})
        return
    bank = make_pair_bank(seed, pair_n)
    init_model = load_from_state(init_state)
    checkpoint_analysis = {}
    for cond in CONDITIONS:
        rows = analyze_condition(seed, cond, out, bank, init_model, train_y)
        checkpoint_analysis[cond] = rows
        save_json(out / f"{cond}_checkpoint_analysis.json", rows)
    final = {c: checkpoint_analysis[c][-1] for c in CONDITIONS}
    validity_by_condition = {c: bool(final[c]["validation"]["combined"] >= 0.38 and final[c]["validation"]["h0_overall"] >= 0.55) for c in CONDITIONS}
    seed_summary = {"experiment": "R8-M3", "family_seed": seed, "fresh_family_seed": seed in FAMILY_SEEDS, "source_train_sha256": sha_tensor(train_y), "source_val_sha256": sha_tensor(val_y), "init_sha256": init_sha, "epoch0_condition_hashes": init_hashes, "validity_by_condition": validity_by_condition, "all_conditions_valid": bool(all(validity_by_condition.values())), "checkpoint_analysis": checkpoint_analysis, "final_conditions": final, "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__}}
    save_json(out / "seed_summary.json", seed_summary)
    with (out / "checkpoint_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "epoch", "h0", "h12", "combined", "G", "winner", "synergy_winner", "gradient_winner"])
        for cond in CONDITIONS:
            for row in checkpoint_analysis[cond]:
                v = row["validation"]
                w.writerow([cond, row["epoch"], v["h0_overall"], v["h12_overall"], v["combined"], row["matched"]["G"], row["matched"]["winner_relation"], row["synergy_winner_relation"], row["gradient_alignment_winner_relation"]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run(args.seed, args.outdir, args.smoke)
