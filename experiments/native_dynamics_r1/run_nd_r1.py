import argparse
import csv
import hashlib
import json
import math
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Recovered Observer-R2 lineage constants.
N_REL = 8
N_VAL = 16
EMB = 8
HIDDEN = 32
LATENT = 16
STEPS = 12
EPS = 1e-8

PRIMARY_SEEDS = (13, 29, 53)
SIZES = {"train": 20000, "val": 2500, "test": 5000}
BATCH = 256
LR = 1e-3
WD = 1e-4
EPOCHS = 100
CLIP = 1.0
CHECKPOINT_EPOCHS = (0, 1, 2, 5, 10, 20, 40, 60, 80, 100)
PAIR_N = 2048
BOOT_N = 5000
RIDGE_LAMBDA = 1e-3


def derive_seed(seed, name):
    h = hashlib.sha256(f"ND-R1|{seed}|{name}".encode()).digest()
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


def sha_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def sha_tensor(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


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


def make_memories(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, N_VAL, (n, N_REL), generator=g)


def make_perms(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.stack([torch.randperm(N_REL, generator=g) for _ in range(n)])


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
        hT = tr[:, -1]
        return tr, [h(h0) for h in self.head0], [h(hT) for h in self.headT]


def evaluate_heads(model, y, perms):
    model.eval()
    c0 = np.zeros(N_REL, dtype=np.int64)
    cT = np.zeros(N_REL, dtype=np.int64)
    with torch.no_grad():
        for a in range(0, len(y), BATCH):
            yy = y[a : a + BATCH]
            pp = perms[a : a + BATCH]
            _, l0, lT = model(yy, pp)
            for r in range(N_REL):
                c0[r] += int((l0[r].argmax(1) == yy[:, r]).sum())
                cT[r] += int((lT[r].argmax(1) == yy[:, r]).sum())
    a0 = c0 / len(y)
    aT = cT / len(y)
    return {
        "h0_overall": float(a0.mean()),
        "h12_overall": float(aT.mean()),
        "h0_per_relation": a0.tolist(),
        "h12_per_relation": aT.tolist(),
    }


def save_checkpoint(model, out, epoch, val_metrics):
    path = out / f"core_epoch_{epoch:03d}.pt"
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save({"state_dict": state, "epoch": epoch, "validation": val_metrics}, path)
    return {"path": path.name, "sha256": sha_file(path), "validation": val_metrics}


def train_core(seed, out):
    train_y = make_memories(SIZES["train"], derive_seed(seed, "memory_train"))
    val_y = make_memories(SIZES["val"], derive_seed(seed, "memory_val"))
    val_perm = make_perms(len(val_y), derive_seed(seed, "perm_val"))

    set_seed(derive_seed(seed, "core_init"))
    model = Core()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    ce = nn.CrossEntropyLoss()

    checkpoints = {}
    history = []

    val0 = evaluate_heads(model, val_y, val_perm)
    checkpoints[0] = save_checkpoint(model, out, 0, val0)
    history.append({"epoch": 0, **val0})

    for ep in range(1, EPOCHS + 1):
        model.train()
        train_perm = make_perms(len(train_y), derive_seed(seed, f"perm_train_{ep}"))
        g = torch.Generator().manual_seed(derive_seed(seed, f"core_batch_{ep}"))
        order = torch.randperm(len(train_y), generator=g)
        total = 0.0
        seen = 0
        for a in range(0, len(train_y), BATCH):
            ix = order[a : a + BATCH]
            y = train_y[ix]
            p = train_perm[ix]
            opt.zero_grad(set_to_none=True)
            _, l0, lT = model(y, p)
            loss = sum(ce(l0[r], y[:, r]) + ce(lT[r], y[:, r]) for r in range(N_REL)) / N_REL
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            total += float(loss.item()) * len(ix)
            seen += len(ix)

        val = evaluate_heads(model, val_y, val_perm)
        row = {"epoch": ep, "train_loss": total / seen, **val}
        history.append(row)
        print(
            f"seed={seed} epoch={ep:03d} loss={row['train_loss']:.6f} "
            f"val_h0={val['h0_overall']:.6f} val_h12={val['h12_overall']:.6f}",
            flush=True,
        )
        if ep in CHECKPOINT_EPOCHS:
            checkpoints[ep] = save_checkpoint(model, out, ep, val)

    save_json(out / "training_history.json", history)
    manifest = {
        "seed": seed,
        "fresh_primary_seed": seed in PRIMARY_SEEDS,
        "training_fixed_epochs": EPOCHS,
        "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
        "train_size": len(train_y),
        "val_size": len(val_y),
        "train_memory_sha256": sha_tensor(train_y),
        "val_memory_sha256": sha_tensor(val_y),
        "val_permutation_sha256": sha_tensor(val_perm),
        "checkpoints": checkpoints,
        "epoch100_competence_gate_h12_validation_ge_0.50": bool(history[-1]["h12_overall"] >= 0.50),
    }
    save_json(out / "training_manifest.json", manifest)
    return train_y, val_y, manifest


def load_checkpoint(path):
    model = Core()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, ck


def generate_trajectory(model, y, perms):
    chunks = []
    with torch.no_grad():
        for a in range(0, len(y), BATCH):
            h0 = model.encode(y[a : a + BATCH], perms[a : a + BATCH])
            chunks.append(model.trajectory(h0).cpu())
    return torch.cat(chunks, 0)


def make_pair_bank(seed):
    bank = []
    hashes = {}
    for r in range(N_REL):
        base = make_memories(PAIR_N, derive_seed(seed, f"pair_memory_relation_{r}"))
        alt = base.clone()
        g = torch.Generator().manual_seed(derive_seed(seed, f"pair_offset_relation_{r}"))
        offset = torch.randint(1, N_VAL, (PAIR_N,), generator=g)
        alt[:, r] = (alt[:, r] + offset) % N_VAL
        perms = make_perms(PAIR_N, derive_seed(seed, f"pair_perm_relation_{r}"))
        bank.append((base, alt, perms))
        hashes[str(r)] = {
            "base_sha256": sha_tensor(base),
            "alt_sha256": sha_tensor(alt),
            "perm_sha256": sha_tensor(perms),
        }
    return bank, hashes


def pair_survival_for_model(model, bank):
    all_rel = []
    with torch.no_grad():
        for base, alt, perms in bank:
            sa = []
            for a in range(0, PAIR_N, BATCH):
                p = perms[a : a + BATCH]
                ta = model.trajectory(model.encode(base[a : a + BATCH], p))
                tb = model.trajectory(model.encode(alt[a : a + BATCH], p))
                d = torch.linalg.vector_norm(ta - tb, dim=-1)
                s = d / (d[:, :1] + EPS)
                sa.append(s.cpu())
            all_rel.append(torch.cat(sa, 0))
    return torch.stack(all_rel, 0).numpy().astype(np.float32)


def geometry(tr):
    dh = tr[:, 1:] - tr[:, :-1]
    speed = torch.linalg.vector_norm(dh, dim=-1)
    direction = dh / (speed.unsqueeze(-1) + EPS)
    radius = torch.linalg.vector_norm(tr, dim=-1)
    turn_cos = torch.nn.functional.cosine_similarity(dh[:, :-1], dh[:, 1:], dim=-1)
    path_len = speed.sum(1)
    endpoint = torch.linalg.vector_norm(tr[:, -1] - tr[:, 0], dim=-1)
    efficiency = endpoint / (path_len + EPS)
    return {
        "dh": dh,
        "speed": speed,
        "direction": direction,
        "radius": radius,
        "turn_cos": turn_cos,
        "path_len": path_len,
        "endpoint": endpoint,
        "efficiency": efficiency,
    }


def summarize_vector_by_time(x):
    arr = x.detach().cpu().numpy()
    return {
        "mean": np.mean(arr, axis=0).tolist(),
        "median": np.median(arr, axis=0).tolist(),
    }


def summarize_geometry(tr):
    g = geometry(tr)
    turn = g["turn_cos"].detach().cpu().numpy()
    return {
        "radius": summarize_vector_by_time(g["radius"]),
        "speed": summarize_vector_by_time(g["speed"]),
        "consecutive_direction_cosine": summarize_vector_by_time(g["turn_cos"]),
        "reversal_fraction_by_turn": np.mean(turn < 0.0, axis=0).tolist(),
        "reversal_fraction_overall": float(np.mean(turn < 0.0)),
        "path_length_mean": float(g["path_len"].mean()),
        "path_length_median": float(g["path_len"].median()),
        "endpoint_displacement_mean": float(g["endpoint"].mean()),
        "endpoint_displacement_median": float(g["endpoint"].median()),
        "endpoint_path_efficiency_mean": float(g["efficiency"].mean()),
        "endpoint_path_efficiency_median": float(g["efficiency"].median()),
    }


def one_hot_targets(y):
    yn = y.detach().cpu().numpy()
    eye = np.eye(N_VAL, dtype=np.float64)
    return eye[yn].reshape(len(yn), N_REL * N_VAL)


def ridge_fit_predict(train_x, train_y_hot, test_x, test_y, lam=RIDGE_LAMBDA):
    x = np.asarray(train_x, dtype=np.float64)
    z = np.asarray(test_x, dtype=np.float64)
    x = np.concatenate([x, np.ones((len(x), 1), dtype=np.float64)], axis=1)
    z = np.concatenate([z, np.ones((len(z), 1), dtype=np.float64)], axis=1)
    reg = np.eye(x.shape[1], dtype=np.float64) * lam
    reg[-1, -1] = 0.0
    w = np.linalg.solve(x.T @ x + reg, x.T @ train_y_hot)
    scores = (z @ w).reshape(len(z), N_REL, N_VAL)
    pred = scores.argmax(-1)
    truth = test_y.detach().cpu().numpy()
    corr = pred == truth
    return {
        "overall": float(corr.mean()),
        "per_relation": corr.mean(0).tolist(),
    }


def ridge_accessibility(train_tr, test_tr, train_y, test_y):
    yhot = one_hot_targets(train_y)
    result = {"state_by_time": []}
    trn = train_tr.detach().cpu().numpy()
    ten = test_tr.detach().cpu().numpy()
    for t in range(STEPS + 1):
        result["state_by_time"].append(ridge_fit_predict(trn[:, t], yhot, ten[:, t], test_y))

    gt = geometry(train_tr)
    ge = geometry(test_tr)
    summaries = {
        "first_direction": (gt["direction"][:, 0], ge["direction"][:, 0]),
        "final_direction": (gt["direction"][:, -1], ge["direction"][:, -1]),
        "integrated_direction": (gt["direction"].sum(1), ge["direction"].sum(1)),
    }
    result["direction_summaries"] = {}
    for name, (a, b) in summaries.items():
        result["direction_summaries"][name] = ridge_fit_predict(
            a.detach().cpu().numpy(), yhot, b.detach().cpu().numpy(), test_y
        )
    return result


def relation_medians(survival, time_index=STEPS):
    # survival shape: [relation, pair, time]
    return np.median(survival[:, :, time_index], axis=1)


def selectivity_G(survival, time_index=STEPS):
    med = relation_medians(survival, time_index)
    return float(np.std(np.log(med + EPS), ddof=0))


def contraction_C0(survival):
    med = relation_medians(survival, STEPS)
    return float(np.mean(np.log(med + EPS)))


def rankdata_average(x):
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        avg = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg
        i = j
    return ranks


def spearman(x, y):
    rx = rankdata_average(x)
    ry = rankdata_average(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def bootstrap_delta_G(seed, s0, s100):
    rng = np.random.default_rng(derive_seed(seed, "bootstrap_delta_G"))
    vals = np.empty(BOOT_N, dtype=np.float64)
    n = s0.shape[1]
    for b in range(BOOT_N):
        med0 = np.empty(N_REL, dtype=np.float64)
        med1 = np.empty(N_REL, dtype=np.float64)
        for r in range(N_REL):
            idx = rng.integers(0, n, n)
            med0[r] = np.median(s0[r, idx, STEPS])
            med1[r] = np.median(s100[r, idx, STEPS])
        g0 = np.std(np.log(med0 + EPS), ddof=0)
        g1 = np.std(np.log(med1 + EPS), ddof=0)
        vals[b] = g1 - g0
    return {
        "n_bootstrap": BOOT_N,
        "ci95_percentile": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))],
        "bootstrap_mean": float(vals.mean()),
    }


def analyze_seed(seed, out, train_y, training_manifest):
    test_y = make_memories(SIZES["test"], derive_seed(seed, "memory_test"))
    probe_train_perm = make_perms(len(train_y), derive_seed(seed, "perm_probe_train"))
    test_perm = make_perms(len(test_y), derive_seed(seed, "perm_test"))
    pair_bank, pair_hashes = make_pair_bank(seed)

    pair_all = np.empty(
        (len(CHECKPOINT_EPOCHS), N_REL, PAIR_N, STEPS + 1), dtype=np.float32
    )
    checkpoint_summaries = []

    for ei, ep in enumerate(CHECKPOINT_EPOCHS):
        model, ck = load_checkpoint(out / f"core_epoch_{ep:03d}.pt")
        pair_surv = pair_survival_for_model(model, pair_bank)
        pair_all[ei] = pair_surv

        train_tr = generate_trajectory(model, train_y, probe_train_perm)
        test_tr = generate_trajectory(model, test_y, test_perm)
        geo_summary = summarize_geometry(test_tr)
        accessibility = ridge_accessibility(train_tr, test_tr, train_y, test_y)

        rel_med_final = relation_medians(pair_surv, STEPS)
        rel_med_by_time = np.median(pair_surv, axis=1)
        row = {
            "epoch": ep,
            "checkpoint_sha256": sha_file(out / f"core_epoch_{ep:03d}.pt"),
            "validation": ck["validation"],
            "C_terminal_mean_log_survival": contraction_C0(pair_surv),
            "G_terminal_relation_selectivity": selectivity_G(pair_surv, STEPS),
            "relation_median_survival_terminal": rel_med_final.tolist(),
            "relation_median_survival_by_time": rel_med_by_time.tolist(),
            "native_geometry": geo_summary,
            "linear_accessibility": accessibility,
        }
        checkpoint_summaries.append(row)
        print(
            f"seed={seed} analyzed epoch={ep:03d} "
            f"G={row['G_terminal_relation_selectivity']:.6f} "
            f"C={row['C_terminal_mean_log_survival']:.6f}",
            flush=True,
        )
        del train_tr, test_tr

    np.savez_compressed(
        out / "natural_pair_survival.npz",
        epochs=np.array(CHECKPOINT_EPOCHS, dtype=np.int64),
        survival=pair_all,
    )

    s0 = pair_all[0]
    s100 = pair_all[-1]
    g0 = selectivity_G(s0)
    g100 = selectivity_G(s100)
    delta_g = g100 - g0
    c0 = contraction_C0(s0)
    boot = bootstrap_delta_G(seed, s0, s100)
    med_t2 = relation_medians(s100, 2)
    med_t12 = relation_medians(s100, 12)
    early_rho = spearman(np.log(med_t2 + EPS), np.log(med_t12 + EPS))

    competence = bool(training_manifest["epoch100_competence_gate_h12_validation_ge_0.50"])
    selective = bool(c0 < 0 and delta_g > 0 and boot["ci95_percentile"][0] > 0)

    primary = {
        "experiment": "ND-R1",
        "seed": seed,
        "fresh_primary_seed": seed in PRIMARY_SEEDS,
        "competence_gate_pass": competence,
        "epoch100_h12_validation_accuracy": training_manifest["checkpoints"][100]["validation"]["h12_overall"],
        "C0_initial_terminal_mean_log_survival": c0,
        "G0_initial_relation_selectivity": g0,
        "G100_final_relation_selectivity": g100,
        "delta_G": delta_g,
        "delta_G_bootstrap": boot,
        "selective_preservation_seed_criterion_pass": selective,
        "early_establishment_spearman_t2_vs_t12": early_rho,
        "population_sd_ddof_for_G": 0,
        "natural_pair_count_per_relation": PAIR_N,
    }

    analysis_manifest = {
        "seed": seed,
        "test_memory_sha256": sha_tensor(test_y),
        "probe_train_permutation_sha256": sha_tensor(probe_train_perm),
        "test_permutation_sha256": sha_tensor(test_perm),
        "pair_bank_hashes": pair_hashes,
        "pair_survival_npz_sha256": sha_file(out / "natural_pair_survival.npz"),
        "ridge_lambda": RIDGE_LAMBDA,
        "bootstrap_n": BOOT_N,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "python": sys.version,
        "platform": platform.platform(),
    }

    save_json(out / "checkpoint_analysis.json", checkpoint_summaries)
    save_json(out / "primary_result.json", primary)
    save_json(out / "analysis_manifest.json", analysis_manifest)

    with (out / "checkpoint_summary.csv").open("w", newline="") as f:
        fields = [
            "epoch",
            "val_h0",
            "val_h12",
            "C_terminal_mean_log_survival",
            "G_terminal_relation_selectivity",
            "reversal_fraction_overall",
            "path_efficiency_mean",
            "ridge_h0",
            "ridge_h12",
            "ridge_first_direction",
            "ridge_final_direction",
            "ridge_integrated_direction",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in checkpoint_summaries:
            la = row["linear_accessibility"]
            ng = row["native_geometry"]
            w.writerow(
                {
                    "epoch": row["epoch"],
                    "val_h0": row["validation"]["h0_overall"],
                    "val_h12": row["validation"]["h12_overall"],
                    "C_terminal_mean_log_survival": row["C_terminal_mean_log_survival"],
                    "G_terminal_relation_selectivity": row["G_terminal_relation_selectivity"],
                    "reversal_fraction_overall": ng["reversal_fraction_overall"],
                    "path_efficiency_mean": ng["endpoint_path_efficiency_mean"],
                    "ridge_h0": la["state_by_time"][0]["overall"],
                    "ridge_h12": la["state_by_time"][-1]["overall"],
                    "ridge_first_direction": la["direction_summaries"]["first_direction"]["overall"],
                    "ridge_final_direction": la["direction_summaries"]["final_direction"]["overall"],
                    "ridge_integrated_direction": la["direction_summaries"]["integrated_direction"]["overall"],
                }
            )

    print(json.dumps(primary, indent=2), flush=True)


def run(seed, outdir):
    if seed not in PRIMARY_SEEDS:
        raise ValueError(f"seed must be one of frozen primary seeds {PRIMARY_SEEDS}")
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    train_y, _, manifest = train_core(seed, out)
    analyze_seed(seed, out, train_y, manifest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    run(args.seed, args.outdir)


if __name__ == "__main__":
    main()
