import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BASE_PATH = HERE.parent / "r8_m10" / "m7r_base.py"
AD2_RESULT_PATH = HERE.parents[1] / "results" / "r8_ad2" / "aggregate" / "FINAL_RESULT.json"
spec = importlib.util.spec_from_file_location("r8_ad3_base", BASE_PATH)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

LINEAGE_SEEDS = (1782, 1801, 1819, 1838, 1856, 1875, 1893, 1912, 1930, 1949, 1967, 1986)
PROBE_TRAIN = 12000
PROBE_VAL = 4000
PROBE_TEST = 8000
INTERNAL_TIMES = tuple(range(1, 12))
SECONDARY_TIME = 12
LAMBDAS = (1e-6, 1e-4, 1e-2, 1.0, 100.0)
REPRESENTATIONS = (
    "STATE",
    "VELOCITY",
    "STATE_VELOCITY",
    "STATE_SHUFFLED_VELOCITY",
    "STATE_QUADRATIC",
)


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def train_to_maturity(seed, smoke=False):
    ntrain = 512 if smoke else base.TRAIN_N
    nval = 256 if smoke else base.VAL_N
    pair_n = 96 if smoke else base.PAIR_N
    max_epoch = 3 if smoke else base.MAX_BASELINE_EPOCH

    train_y = base.make_memories(ntrain, base.derive_seed(seed, "train"))
    val_y = base.make_memories(nval, base.derive_seed(seed, "val"))
    val_perm = base.make_perms(nval, base.derive_seed(seed, "val_perm"))
    bank = base.make_pair_bank(seed, pair_n)

    base.set_seed(base.derive_seed(seed, "init"))
    model = base.Core()
    opt = torch.optim.AdamW(model.parameters(), lr=base.LR, weight_decay=base.WD)

    if smoke:
        for ep in range(1, max_epoch + 1):
            base.train_one_epoch(model, opt, seed, train_y, ep)
        return model, max_epoch, []

    history = []
    maturity = None
    for ep in range(1, max_epoch + 1):
        base.train_one_epoch(model, opt, seed, train_y, ep)
        if ep >= base.FIRST_RECORDED_CHECK and ep % base.CHECK_EVERY == 0:
            rec = base.checkpoint_record(model, val_y, val_perm, bank, ep)
            history.append(rec)
            print(
                f"seed={seed} baseline ep={ep} combined={rec['validation']['combined']:.4f} "
                f"h0={rec['validation']['h0_overall']:.4f} competent={rec['competent']} "
                f"winner={rec['survival']['winner_relation']} loser={rec['survival']['loser_relation']}",
                flush=True,
            )
            maturity = base.maturity_from_history(history)
            if maturity is not None:
                break
    return model, maturity, history


def make_probe_split(seed, split, n):
    y = base.make_memories(n, base.derive_seed(seed, f"ad3_probe_{split}_y"))
    p = base.make_perms(n, base.derive_seed(seed, f"ad3_probe_{split}_perm"))
    return y, p


def collect_trajectory(model, y, perms):
    chunks = []
    model.eval()
    with torch.no_grad():
        for a in range(0, len(y), base.BATCH):
            yy = y[a:a + base.BATCH]
            pp = perms[a:a + base.BATCH]
            h0 = model.encode(yy, pp)
            tr = model.trajectory(h0)
            chunks.append(tr.cpu())
    return torch.cat(chunks, dim=0).numpy().astype(np.float64)


def quadratic_features(h):
    ii, jj = np.triu_indices(h.shape[1])
    q = h[:, ii] * h[:, jj]
    return np.concatenate([h, q], axis=1)


def feature_matrix(tr, t, rep, seed, split):
    h = tr[:, t, :]
    prev = tr[:, t - 1, :]
    d = h - prev
    if rep == "STATE":
        return h
    if rep == "VELOCITY":
        return d
    if rep == "STATE_VELOCITY":
        return np.concatenate([h, d], axis=1)
    if rep == "STATE_SHUFFLED_VELOCITY":
        rng = np.random.default_rng(base.derive_seed(seed, f"ad3_shuffle_{split}_{t}"))
        order = rng.permutation(len(d))
        return np.concatenate([h, d[order]], axis=1)
    if rep == "STATE_QUADRATIC":
        return quadratic_features(h)
    raise ValueError(rep)


def one_hot_targets(y):
    y = np.asarray(y, dtype=np.int64)
    out = np.zeros((len(y), base.N_REL * base.N_VAL), dtype=np.float64)
    rows = np.arange(len(y))[:, None]
    rel = np.arange(base.N_REL)[None, :]
    cols = rel * base.N_VAL + y
    out[rows, cols] = 1.0
    return out


def standardize_fit(x):
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.where(sd < 1e-10, 1.0, sd)
    return mu, sd


def standardize_apply(x, mu, sd):
    z = (x - mu) / sd
    return np.concatenate([z, np.ones((len(z), 1), dtype=z.dtype)], axis=1)


def fit_ridge(x, y, lam):
    xtx = x.T @ x
    xty = x.T @ y
    reg = np.eye(x.shape[1], dtype=np.float64) * float(lam)
    reg[-1, -1] = 0.0  # do not regularize bias
    return np.linalg.solve(xtx + reg, xty)


def accuracy_per_relation(logits, y):
    logits = logits.reshape(len(logits), base.N_REL, base.N_VAL)
    pred = logits.argmax(axis=2)
    y = np.asarray(y, dtype=np.int64)
    return (pred == y).mean(axis=0).astype(np.float64)


def fit_select_evaluate(xtr, ytr_oh, ytr, xv, yv, xt, yt):
    mu, sd = standardize_fit(xtr)
    a = standardize_apply(xtr, mu, sd)
    b = standardize_apply(xv, mu, sd)
    c = standardize_apply(xt, mu, sd)

    best = None
    for lam in LAMBDAS:
        w = fit_ridge(a, ytr_oh, lam)
        acc = accuracy_per_relation(b @ w, yv)
        score = float(acc.mean())
        rec = (score, -math.log10(lam), lam, w, acc)
        if best is None or rec[:2] > best[:2]:
            best = rec

    _, _, lam, w, val_acc = best
    test_acc = accuracy_per_relation(c @ w, yt)
    return {
        "lambda": float(lam),
        "val_mean_accuracy": float(val_acc.mean()),
        "val_per_relation": val_acc.tolist(),
        "test_mean_accuracy": float(test_acc.mean()),
        "test_per_relation": test_acc.tolist(),
        "n_features_without_bias": int(xtr.shape[1]),
    }


def ad2_regime_for_seed(seed):
    try:
        d = json.loads(AD2_RESULT_PATH.read_text())
        for row in d.get("rows", []):
            if int(row.get("seed")) == int(seed):
                return row.get("family_class")
    except Exception:
        pass
    return None


def probe_family(model, seed, smoke=False):
    sizes = {
        "train": 384 if smoke else PROBE_TRAIN,
        "val": 128 if smoke else PROBE_VAL,
        "test": 256 if smoke else PROBE_TEST,
    }
    ys = {}
    trs = {}
    for split, n in sizes.items():
        y, p = make_probe_split(seed, split, n)
        ys[split] = y.numpy().astype(np.int64)
        trs[split] = collect_trajectory(model, y, p)

    ytr_oh = one_hot_targets(ys["train"])
    by_time = {}
    all_times = INTERNAL_TIMES + (SECONDARY_TIME,)
    for t in all_times:
        rec = {}
        for rep in REPRESENTATIONS:
            xtr = feature_matrix(trs["train"], t, rep, seed, "train")
            xv = feature_matrix(trs["val"], t, rep, seed, "val")
            xt = feature_matrix(trs["test"], t, rep, seed, "test")
            rec[rep] = fit_select_evaluate(
                xtr,
                ytr_oh,
                ys["train"],
                xv,
                ys["val"],
                xt,
                ys["test"],
            )
        by_time[str(t)] = rec
        print(
            f"seed={seed} t={t} state={rec['STATE']['test_mean_accuracy']:.4f} "
            f"joint={rec['STATE_VELOCITY']['test_mean_accuracy']:.4f} "
            f"shuffle={rec['STATE_SHUFFLED_VELOCITY']['test_mean_accuracy']:.4f}",
            flush=True,
        )

    primary = {}
    for rep in REPRESENTATIONS:
        vals = [by_time[str(t)][rep]["test_mean_accuracy"] for t in INTERNAL_TIMES]
        per_rel = np.asarray(
            [by_time[str(t)][rep]["test_per_relation"] for t in INTERNAL_TIMES],
            dtype=np.float64,
        ).mean(axis=0)
        primary[rep] = {
            "mean_accuracy_internal": float(np.mean(vals)),
            "per_relation_accuracy_internal": per_rel.tolist(),
        }

    delta_state = (
        primary["STATE_VELOCITY"]["mean_accuracy_internal"]
        - primary["STATE"]["mean_accuracy_internal"]
    )
    delta_shuffle = (
        primary["STATE_VELOCITY"]["mean_accuracy_internal"]
        - primary["STATE_SHUFFLED_VELOCITY"]["mean_accuracy_internal"]
    )
    delta_quadratic = (
        primary["STATE_VELOCITY"]["mean_accuracy_internal"]
        - primary["STATE_QUADRATIC"]["mean_accuracy_internal"]
    )

    finite = all(
        math.isfinite(by_time[str(t)][rep]["test_mean_accuracy"])
        for t in all_times
        for rep in REPRESENTATIONS
    )
    return {
        "probe_sizes": sizes,
        "internal_times": list(INTERNAL_TIMES),
        "secondary_time": SECONDARY_TIME,
        "ridge_lambdas": list(LAMBDAS),
        "by_time": by_time,
        "primary": primary,
        "delta_state": float(delta_state),
        "delta_shuffle": float(delta_shuffle),
        "delta_quadratic": float(delta_quadratic),
        "ad2_regime_secondary": ad2_regime_for_seed(seed),
        "finite": bool(finite),
    }


def run(seed, outdir, smoke=False):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    model, maturity, history = train_to_maturity(seed, smoke=smoke)
    valid_maturity = bool(smoke or maturity is not None)
    diag = probe_family(model, seed, smoke=smoke)
    summary = {
        "experiment": "R8-AD3",
        "seed": int(seed),
        "expected_lineage": bool(seed in LINEAGE_SEEDS),
        "maturity_reached": valid_maturity,
        "maturity_epoch": int(maturity) if maturity is not None else (3 if smoke else None),
        "baseline_history": [] if smoke else history,
        "diagnostic": diag,
        "validity": {
            "maturity": valid_maturity,
            "finite": bool(diag["finite"]),
            "complete": True,
        },
        "all_valid": bool(valid_maturity and diag["finite"]),
        "environment": {
            "python": __import__("sys").version,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    save_json(out / ("smoke_summary.json" if smoke else "seed_summary.json"), summary)
    print(
        f"seed={seed} delta_state={diag['delta_state']:+.6f} "
        f"delta_shuffle={diag['delta_shuffle']:+.6f} "
        f"delta_quadratic={diag['delta_quadratic']:+.6f}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run(args.seed, args.outdir, smoke=args.smoke)


if __name__ == "__main__":
    main()
