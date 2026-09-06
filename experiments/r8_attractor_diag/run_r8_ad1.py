import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BASE_PATH = HERE.parent / "r8_m10" / "m7r_base.py"
spec = importlib.util.spec_from_file_location("r8_ad1_base", BASE_PATH)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

FRESH_SEEDS = (1561, 1579, 1597, 1616, 1634, 1652, 1671, 1689, 1708, 1726, 1745, 1763)
N_NATIVE = 2048
LONG_STEPS = 512
TAIL_START = 448
FIX_TOL = 1e-5
FAMILY_FRAC = 0.95
PERIODS = tuple(range(2, 17))
RECORD_STEPS = (0, 12, 64, 128, 256, 384, 448, 512)


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


def summarize_array(x):
    x = np.asarray(x, dtype=np.float64)
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "p95": float(np.quantile(x, 0.95)),
    }


def long_run_diagnostic(model, seed, smoke=False):
    n = 128 if smoke else N_NATIVE
    steps = 64 if smoke else LONG_STEPS
    tail_start = max(0, steps - 64)

    y = base.make_memories(n, base.derive_seed(seed, "ad1_native_y"))
    perms = base.make_perms(n, base.derive_seed(seed, "ad1_native_perm"))

    fixed_flags = []
    period_flags = {p: [] for p in PERIODS}
    d12_512 = []
    extra_resid = []
    residual_at = {s: [] for s in RECORD_STEPS if s > 0 and s <= steps}
    finals = []

    model.eval()
    with torch.no_grad():
        for a in range(0, n, base.BATCH):
            yy = y[a:a + base.BATCH]
            pp = perms[a:a + base.BATCH]
            h = model.encode(yy, pp)
            h12 = None
            tail_states = []
            residuals = []
            if tail_start == 0:
                tail_states.append(h.clone())

            for t in range(1, steps + 1):
                h_next = model.F(h)
                r = torch.linalg.vector_norm(h_next - h, dim=1)
                residuals.append(r)
                if t in residual_at:
                    residual_at[t].append(r.cpu())
                h = h_next
                if t == 12:
                    h12 = h.clone()
                if t >= tail_start:
                    tail_states.append(h.clone())

            if h12 is None:
                h12 = h.clone()
            h_end = h
            r_extra = torch.linalg.vector_norm(model.F(h_end) - h_end, dim=1)
            extra_resid.append(r_extra.cpu())
            finals.append(h_end.cpu())
            d12_512.append(torch.linalg.vector_norm(h12 - h_end, dim=1).cpu())

            rmat = torch.stack(residuals, dim=1)
            final32 = rmat[:, -min(32, steps):]
            fixed = (final32.max(dim=1).values <= FIX_TOL) & (r_extra <= FIX_TOL)
            fixed_flags.append(fixed.cpu())

            tail = torch.stack(tail_states, dim=1)
            one_step_tail = torch.linalg.vector_norm(tail[:, 1:] - tail[:, :-1], dim=2)
            one_step_med = one_step_tail[:, -min(32, one_step_tail.shape[1]):].median(dim=1).values
            for p in PERIODS:
                if tail.shape[1] <= p:
                    cyc = torch.zeros(len(tail), dtype=torch.bool, device=tail.device)
                else:
                    dp = torch.linalg.vector_norm(tail[:, p:] - tail[:, :-p], dim=2)
                    dp = dp[:, -min(32, dp.shape[1]):]
                    cyc = (dp.median(dim=1).values <= FIX_TOL) & (one_step_med > FIX_TOL) & (~fixed)
                period_flags[p].append(cyc.cpu())

    fixed = torch.cat(fixed_flags).numpy().astype(bool)
    pf = {p: torch.cat(v).numpy().astype(bool) for p, v in period_flags.items()}
    fixed_fraction = float(fixed.mean())
    period_fraction = {str(p): float(v.mean()) for p, v in pf.items()}

    family_class = "UNRESOLVED"
    family_period = None
    if fixed_fraction >= FAMILY_FRAC:
        family_class = "FIXED"
    else:
        for p in PERIODS:
            if period_fraction[str(p)] >= FAMILY_FRAC:
                family_class = "SHORT_CYCLE"
                family_period = int(p)
                break

    finals_np = torch.cat(finals).numpy().astype(np.float64)
    centroid = finals_np.mean(axis=0, keepdims=True)
    spread = np.linalg.norm(finals_np - centroid, axis=1)

    out = {
        "n_native": int(n),
        "long_steps": int(steps),
        "fix_tol": FIX_TOL,
        "family_fraction_threshold": FAMILY_FRAC,
        "family_class": family_class,
        "family_period": family_period,
        "fixed_fraction": fixed_fraction,
        "period_fraction": period_fraction,
        "d12_end": summarize_array(torch.cat(d12_512).numpy()),
        "extra_fixed_residual": summarize_array(torch.cat(extra_resid).numpy()),
        "final_state_spread_from_centroid": summarize_array(spread),
        "residual_at_steps": {
            str(s): summarize_array(torch.cat(chunks).numpy())
            for s, chunks in residual_at.items() if chunks
        },
        "finite": bool(
            np.isfinite(finals_np).all()
            and math.isfinite(fixed_fraction)
            and all(math.isfinite(x) for x in period_fraction.values())
        ),
    }
    return out


def run(seed, outdir, smoke=False):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    model, maturity, history = train_to_maturity(seed, smoke=smoke)
    diag = long_run_diagnostic(model, seed, smoke=smoke)
    valid_maturity = bool(smoke or maturity is not None)
    summary = {
        "experiment": "R8-AD1",
        "seed": int(seed),
        "fresh_seed": bool(seed in FRESH_SEEDS),
        "maturity_reached": valid_maturity,
        "maturity_epoch": int(maturity) if maturity is not None else (3 if smoke else None),
        "baseline_history": history if not smoke else [],
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
        f"seed={seed} class={diag['family_class']} fixed_fraction={diag['fixed_fraction']:.4f} "
        f"d12_end_median={diag['d12_end']['median']:.6g}",
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
