import argparse
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BASE_PATH = HERE.parent / "r8_m10" / "m7r_base.py"
spec = importlib.util.spec_from_file_location("r8_ad2_base", BASE_PATH)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

FRESH_SEEDS = (1782, 1801, 1819, 1838, 1856, 1875, 1893, 1912, 1930, 1949, 1967, 1986)
N_NATIVE = 256
LONG_STEPS = 4096
TAIL_LEN = 1024
PERIOD_MAX = 512
FTLE_BURNIN = 512
FTLE_STEPS = 512
FTLE_EPS = 1e-4
FIX_TOL = 1e-5
PERIOD_TOL = 1e-4
SENSITIVE_THR = 0.01
QUASI_FTLE_THR = 0.01
QUASI_TOP8_THR = 0.90
FAMILY_DOMINANCE = 0.80
STATE_CLASSES = ("FIXED", "PERIODIC", "SENSITIVE", "QUASIPERIODIC_LIKE", "REGULAR_NONPERIODIC")


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def summarize(x):
    x = np.asarray(x, dtype=np.float64)
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "p05": float(np.quantile(x, 0.05)),
        "p95": float(np.quantile(x, 0.95)),
    }


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


def spectral_metrics(tail):
    # tail: [B, T, D]. Sum coordinate power so the spectrum is invariant to
    # orthogonal rotations of latent coordinates.
    x = tail - tail.mean(dim=1, keepdim=True)
    z = torch.fft.rfft(x, dim=1)
    power = (z.real.square() + z.imag.square()).sum(dim=2)
    power = power[:, 1:]  # remove DC
    total = power.sum(dim=1, keepdim=True).clamp_min(1e-30)
    p = power / total
    nfreq = p.shape[1]
    ent = -(p * p.clamp_min(1e-30).log()).sum(dim=1) / math.log(max(2, nfreq))
    k = min(8, nfreq)
    top8 = torch.topk(p, k=k, dim=1).values.sum(dim=1)
    return ent, top8


def periodic_metrics(tail, fixed, smoke=False):
    # tail: [B,T,D]. Find the best lag by final-state recurrence, then validate
    # that lag over the final comparable window.
    B, T, _ = tail.shape
    maxp = min(16 if smoke else PERIOD_MAX, T - 65)
    periods = torch.arange(2, maxp + 1, device=tail.device)
    idx = (T - 1) - periods
    past = tail[:, idx, :]
    d_final = torch.linalg.vector_norm(tail[:, -1:, :] - past, dim=2)
    best_ix = d_final.argmin(dim=1)
    best_p = periods[best_ix]
    best_final = d_final.gather(1, best_ix[:, None]).squeeze(1)

    one_step = torch.linalg.vector_norm(tail[:, 1:] - tail[:, :-1], dim=2)
    one_step_med = one_step[:, -64:].median(dim=1).values
    rec_med = torch.full((B,), float("inf"), dtype=tail.dtype, device=tail.device)

    for p in torch.unique(best_p).tolist():
        mask = best_p == int(p)
        now = tail[mask, -64:, :]
        prev = tail[mask, -64-int(p):-int(p), :]
        dp = torch.linalg.vector_norm(now - prev, dim=2)
        rec_med[mask] = dp.median(dim=1).values

    periodic = (~fixed) & (rec_med <= PERIOD_TOL) & (one_step_med > FIX_TOL)
    return best_p, best_final, rec_med, one_step_med, periodic


def ftle_metrics(model, h_burn, directions, steps):
    eps = FTLE_EPS
    h = h_burn.clone()
    v = directions / torch.linalg.vector_norm(directions, dim=1, keepdim=True).clamp_min(1e-12)
    hp = h + eps * v
    acc = torch.zeros(len(h), dtype=h.dtype, device=h.device)

    with torch.no_grad():
        for _ in range(steps):
            hn = model.F(h)
            hpn = model.F(hp)
            delta = hpn - hn
            dist = torch.linalg.vector_norm(delta, dim=1).clamp_min(1e-12)
            acc += torch.log(dist / eps)
            v = delta / dist[:, None]
            h = hn
            hp = hn + eps * v
    return acc / float(steps)


def regime_diagnostic(model, seed, smoke=False):
    n = 64 if smoke else N_NATIVE
    steps = 128 if smoke else LONG_STEPS
    tail_len = min(128 if smoke else TAIL_LEN, steps + 1)
    burnin = min(32 if smoke else FTLE_BURNIN, steps)
    ftle_steps = 64 if smoke else FTLE_STEPS

    y = base.make_memories(n, base.derive_seed(seed, "ad2_native_y"))
    perms = base.make_perms(n, base.derive_seed(seed, "ad2_native_perm"))
    g = torch.Generator().manual_seed(base.derive_seed(seed, "ad2_ftle_dir"))
    dirs_all = torch.randn((n, base.LATENT), generator=g)

    all_fixed = []
    all_periodic = []
    all_best_p = []
    all_rec_med = []
    all_end_resid = []
    all_ftle = []
    all_entropy = []
    all_top8 = []

    model.eval()
    with torch.no_grad():
        for a in range(0, n, base.BATCH):
            yy = y[a:a + base.BATCH]
            pp = perms[a:a + base.BATCH]
            h = model.encode(yy, pp)
            h_burn = None
            tail_states = []
            start_tail = steps - tail_len + 1
            if start_tail <= 0:
                tail_states.append(h.clone())

            for t in range(1, steps + 1):
                h = model.F(h)
                if t == burnin:
                    h_burn = h.clone()
                if t >= start_tail:
                    tail_states.append(h.clone())

            if h_burn is None:
                h_burn = h.clone()
            tail = torch.stack(tail_states, dim=1)
            if tail.shape[1] < 66:
                raise RuntimeError("tail too short for frozen diagnostics")

            one = torch.linalg.vector_norm(tail[:, 1:] - tail[:, :-1], dim=2)
            final64 = one[:, -64:]
            end_resid = torch.linalg.vector_norm(model.F(tail[:, -1]) - tail[:, -1], dim=1)
            fixed = (final64.max(dim=1).values <= FIX_TOL) & (end_resid <= FIX_TOL)
            best_p, _, rec_med, _, periodic = periodic_metrics(tail, fixed, smoke=smoke)
            ent, top8 = spectral_metrics(tail)

            dirs = dirs_all[a:a + len(yy)].to(h_burn.device, dtype=h_burn.dtype)
            ftle = ftle_metrics(model, h_burn, dirs, ftle_steps)

            all_fixed.append(fixed.cpu())
            all_periodic.append(periodic.cpu())
            all_best_p.append(best_p.cpu())
            all_rec_med.append(rec_med.cpu())
            all_end_resid.append(end_resid.cpu())
            all_ftle.append(ftle.cpu())
            all_entropy.append(ent.cpu())
            all_top8.append(top8.cpu())

    fixed = torch.cat(all_fixed).numpy().astype(bool)
    periodic = torch.cat(all_periodic).numpy().astype(bool)
    best_p = torch.cat(all_best_p).numpy().astype(int)
    rec_med = torch.cat(all_rec_med).numpy().astype(np.float64)
    end_resid = torch.cat(all_end_resid).numpy().astype(np.float64)
    ftle = torch.cat(all_ftle).numpy().astype(np.float64)
    entropy = torch.cat(all_entropy).numpy().astype(np.float64)
    top8 = torch.cat(all_top8).numpy().astype(np.float64)

    classes = np.full(n, "REGULAR_NONPERIODIC", dtype=object)
    classes[fixed] = "FIXED"
    classes[(~fixed) & periodic] = "PERIODIC"
    remaining = (~fixed) & (~periodic)
    sensitive = remaining & (ftle > SENSITIVE_THR)
    classes[sensitive] = "SENSITIVE"
    quasi = remaining & (~sensitive) & (np.abs(ftle) <= QUASI_FTLE_THR) & (top8 >= QUASI_TOP8_THR)
    classes[quasi] = "QUASIPERIODIC_LIKE"

    counts = Counter(classes.tolist())
    fractions = {c: float(counts.get(c, 0) / n) for c in STATE_CLASSES}
    dominant_class, dominant_fraction = max(fractions.items(), key=lambda kv: kv[1])
    family_class = dominant_class if dominant_fraction >= FAMILY_DOMINANCE else "MIXED"

    period_values = best_p[periodic]
    period_counts = {str(k): int(v) for k, v in sorted(Counter(period_values.tolist()).items())}

    finite = bool(
        np.isfinite(ftle).all()
        and np.isfinite(entropy).all()
        and np.isfinite(top8).all()
        and np.isfinite(end_resid).all()
        and np.isfinite(rec_med).all()
    )

    return {
        "n_native": int(n),
        "long_steps": int(steps),
        "tail_len": int(tail_len),
        "family_class": family_class,
        "dominant_state_fraction": float(dominant_fraction),
        "state_class_counts": {c: int(counts.get(c, 0)) for c in STATE_CLASSES},
        "state_class_fractions": fractions,
        "period_counts_among_periodic": period_counts,
        "ftle": summarize(ftle),
        "spectral_entropy": summarize(entropy),
        "top8_power_fraction": summarize(top8),
        "final_one_step_residual": summarize(end_resid),
        "validated_recurrence_distance": summarize(rec_med),
        "per_state": {
            "class": classes.tolist(),
            "best_period": best_p.tolist(),
            "recurrence_median": rec_med.tolist(),
            "ftle": ftle.tolist(),
            "spectral_entropy": entropy.tolist(),
            "top8_power_fraction": top8.tolist(),
            "end_residual": end_resid.tolist(),
        },
        "thresholds": {
            "fix_tol": FIX_TOL,
            "period_tol": PERIOD_TOL,
            "period_max": 16 if smoke else PERIOD_MAX,
            "ftle_epsilon": FTLE_EPS,
            "sensitive_threshold": SENSITIVE_THR,
            "quasi_ftle_abs_threshold": QUASI_FTLE_THR,
            "quasi_top8_threshold": QUASI_TOP8_THR,
            "family_dominance": FAMILY_DOMINANCE,
        },
        "finite": finite,
    }


def run(seed, outdir, smoke=False):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    model, maturity, history = train_to_maturity(seed, smoke=smoke)
    valid_maturity = bool(smoke or maturity is not None)
    diag = regime_diagnostic(model, seed, smoke=smoke)
    summary = {
        "experiment": "R8-AD2",
        "seed": int(seed),
        "fresh_seed": bool(seed in FRESH_SEEDS),
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
        f"seed={seed} family={diag['family_class']} fractions={diag['state_class_fractions']} "
        f"median_ftle={diag['ftle']['median']:.6g} median_top8={diag['top8_power_fraction']['median']:.6g}",
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
