import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

N_REL = 8
N_VAL = 16
EMB = 8
HIDDEN = 32
LATENT = 16


def derive_seed(seed: int, name: str) -> int:
    h = hashlib.sha256(f"ALI-N8-R1|{seed}|{name}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def init_module(m: nn.Module) -> None:
    if isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def init_gru(gru: nn.GRU) -> None:
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
        self.gru = nn.GRU(EMB * 2, HIDDEN, batch_first=True)
        self.to_m = nn.Linear(HIDDEN, LATENT)
        self.F = nn.Sequential(
            nn.Linear(LATENT, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, LATENT), nn.Tanh()
        )
        self.head_m = nn.ModuleList([nn.Linear(LATENT, N_VAL) for _ in range(N_REL)])
        self.head_z = nn.ModuleList([nn.Linear(LATENT, N_VAL) for _ in range(N_REL)])
        self.apply(init_module)
        init_gru(self.gru)

    def encode(self, values: torch.Tensor, perms: torch.Tensor) -> torch.Tensor:
        b = values.size(0)
        rel = torch.arange(N_REL, device=values.device).expand(b, -1).gather(1, perms)
        val = values.gather(1, perms)
        x = torch.cat([self.rel_emb(rel), self.val_emb(val)], dim=-1)
        _, h = self.gru(x)
        return torch.tanh(self.to_m(h[-1]))

    def forward(self, values: torch.Tensor, perms: torch.Tensor):
        m = self.encode(values, perms)
        z = self.F(m)
        return m, z, [h(m) for h in self.head_m], [h(z) for h in self.head_z]


def make_memories(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, N_VAL, (n, N_REL), generator=g)


def make_perms(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.stack([torch.randperm(N_REL, generator=g) for _ in range(n)])


def evaluate(model, values, perms, batch_size):
    model.eval()
    correct = np.zeros(16, dtype=np.int64)
    total = 0
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            y = values[start:start + batch_size]
            p = perms[start:start + batch_size]
            _, _, lm, lz = model(y, p)
            total += len(y)
            for i in range(N_REL):
                correct[2 * i] += (lm[i].argmax(1) == y[:, i]).sum().item()
                correct[2 * i + 1] += (lz[i].argmax(1) == y[:, i]).sum().item()
    acc = correct / total
    return float(acc.mean()), acc.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--outdir", default="results/reproducible/ali_n8_r1/seed_5/core")
    args = ap.parse_args()

    if args.seed not in (5, 17, 31):
        raise SystemExit("Primary R1 core runner only accepts seeds 5, 17, or 31")

    cfg = {
        "experiment": "ALI-N8-R1",
        "phase": "core_train_validation_only",
        "seed": args.seed,
        "train_memories": 20000,
        "validation_memories": 2500,
        "test_memories": 5000,
        "batch_size": 256,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "max_epochs": 100,
        "patience": 12,
        "min_improvement": 1e-4,
        "gradient_clip": 1.0,
        "test_policy": "not_generated_or_evaluated_in_core_phase",
    }

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({
        **cfg,
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
    }, indent=2))

    # Clean replication rule: core training creates only train and validation splits.
    # The test split is generated for the first time by run_r1.py --phase final,
    # after every downstream checkpoint and diagnostic decoder is frozen.
    train = make_memories(cfg["train_memories"], derive_seed(args.seed, "memory_train"))
    val = make_memories(cfg["validation_memories"], derive_seed(args.seed, "memory_val"))
    val_perm = make_perms(len(val), derive_seed(args.seed, "perm_val"))

    set_seed(derive_seed(args.seed, "core_init"))
    model = Core()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()

    best = -1.0
    best_epoch = 0
    best_state = None
    stale = 0
    history = []

    for epoch in range(1, 101):
        model.train()
        train_perm = make_perms(len(train), derive_seed(args.seed, f"train_perm_{epoch}"))
        g = torch.Generator().manual_seed(derive_seed(args.seed, f"core_batch_{epoch}"))
        loader = DataLoader(TensorDataset(train, train_perm), batch_size=256, shuffle=True, generator=g)

        for y, p in loader:
            opt.zero_grad(set_to_none=True)
            _, _, lm, lz = model(y, p)
            loss = sum(ce(lm[i], y[:, i]) + ce(lz[i], y[:, i]) for i in range(N_REL)) / N_REL
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        val_metric, val_heads = evaluate(model, val, val_perm, 256)
        history.append({"epoch": epoch, "validation_metric": val_metric, "head_accuracies": val_heads})
        (out / "training_history.json").write_text(json.dumps(history, indent=2))
        print(f"epoch={epoch:03d} validation_metric={val_metric:.6f}", flush=True)

        if val_metric > best + 1e-4:
            best = val_metric
            best_epoch = epoch
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"state_dict": best_state, "epoch": best_epoch, "validation_metric": best}, out / "core_best.pt")
        else:
            stale += 1
            if stale >= 12:
                break

    if best_state is None:
        raise RuntimeError("No core checkpoint selected")
    model.load_state_dict(best_state)

    # Alpha is calibrated from training latent states only after core selection.
    model.eval()
    train_perm_final = make_perms(len(train), derive_seed(args.seed, "perm_train_alpha"))
    latent_chunks = []
    with torch.no_grad():
        for start in range(0, len(train), 256):
            latent_chunks.append(model.encode(train[start:start + 256], train_perm_final[start:start + 256]))
    m_train = torch.cat(latent_chunks)
    median_norm = m_train.norm(dim=1).median().item()
    alpha = 0.1 * median_norm

    ckpt_hash = hashlib.sha256((out / "core_best.pt").read_bytes()).hexdigest()
    summary = {
        "best_epoch": best_epoch,
        "best_validation_metric": best,
        "median_training_latent_norm": median_norm,
        "alpha": alpha,
        "checkpoint_sha256": ckpt_hash,
        "status": "core_complete_frozen_test_unseen",
        "test_generated": False,
        "test_evaluated": False,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
