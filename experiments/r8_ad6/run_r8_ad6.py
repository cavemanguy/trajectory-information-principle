import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
AD5_PATH = HERE.parent / "r8_ad5" / "run_r8_ad5.py"
spec = importlib.util.spec_from_file_location("r8_ad6_ad5", AD5_PATH)
ad5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ad5)
ad3 = ad5.ad3
base = ad5.base

LINEAGE_SEEDS = ad5.LINEAGE_SEEDS
INTERVENTION_N = 4096
TIMES = (2, 4, 6, 8)
RHOS = (0.05, 0.10, 0.20)
N_TRANSVERSE = 4
RECOVERY_HORIZON = 4
PRIMARY_HORIZON = 3
EPS = 1e-8
GEOM_TOL = 1e-5


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def make_intervention_split(seed, n):
    y = base.make_memories(n, base.derive_seed(seed, "ad6_intervention_y"))
    p = base.make_perms(n, base.derive_seed(seed, "ad6_intervention_perm"))
    return y, p


def normalize_rows(x, eps=EPS):
    n = torch.linalg.vector_norm(x, dim=1, keepdim=True)
    return x / torch.clamp(n, min=eps), n


def collect_extended_trajectory(model, y, perms):
    tr = ad5.collect_trajectory(model, y, perms)
    with torch.no_grad():
        h13 = model.F(tr[:, -1]).cpu().unsqueeze(1)
    return torch.cat([tr, h13], dim=1)


def make_arms(native_t, native_next, seed, t, rho):
    d = native_next - native_t
    u, m = normalize_rows(d)
    valid = m[:, 0] >= 1e-7
    r = float(rho) * m

    arms = {
        "TPLUS": native_t + r * u,
        "TMINUS": native_t - r * u,
    }
    max_err = torch.tensor(0.0, dtype=native_t.dtype)

    for j in range(N_TRANSVERSE):
        q = ad5.orthogonal_random(u, base.derive_seed(seed, f"ad6_trans_{t}_{rho:.3f}_{j}"))
        arms[f"P{j}"] = native_t + r * q
        if bool(valid.any()):
            vv = valid
            dot = torch.abs((q[vv] * u[vv]).sum(1))
            max_err = torch.maximum(max_err, dot.max())

    if bool((~valid).any()):
        for z in arms.values():
            z[~valid] = native_t[~valid]

    if bool(valid.any()):
        vv = valid
        target = r[vv, 0]
        for z in arms.values():
            actual = torch.linalg.vector_norm((z - native_t)[vv], dim=1)
            max_err = torch.maximum(max_err, torch.max(torch.abs(actual - target)))

    return arms, {
        "valid_fraction": float(valid.float().mean()),
        "max_geometry_error": float(max_err),
    }, valid, r[:, 0]


def arm_recovery_metrics(state, native_ref, native_next_ref, r, valid):
    if not bool(valid.any()):
        return {
            "total_retention": 0.0,
            "parallel_retention": 0.0,
            "transverse_retention": 0.0,
            "parallel_fraction": 0.0,
            "signed_progress": 0.0,
        }
    vv = valid
    e = state[vv] - native_ref[vv]
    u, _ = normalize_rows(native_next_ref[vv] - native_ref[vv])
    rv = torch.clamp(r[vv], min=EPS)
    dot = (e * u).sum(1)
    total = torch.linalg.vector_norm(e, dim=1)
    perp = e - dot[:, None] * u
    perp_n = torch.linalg.vector_norm(perp, dim=1)
    frac = torch.abs(dot) / torch.clamp(total, min=EPS)
    return {
        "total_retention": float(torch.median(total / rv)),
        "parallel_retention": float(torch.median(torch.abs(dot) / rv)),
        "transverse_retention": float(torch.median(perp_n / rv)),
        "parallel_fraction": float(torch.median(frac)),
        "signed_progress": float(torch.median(dot / rv)),
    }


def terminal_from_state(model, state, current_index, y):
    final = ad5.rollout_to_12(model, state, current_index)
    return ad5.terminal_metrics(model, final, y)


def analyze_condition(model, tr, y, seed, t, rho, native_terminal):
    native_t = tr[:, t]
    native_next = tr[:, t + 1]
    arms, geom, valid, r = make_arms(native_t, native_next, seed, t, rho)
    arm_names = ("TPLUS", "TMINUS", "P0", "P1", "P2", "P3")
    states = {k: v.clone() for k, v in arms.items()}
    recovery = {name: [] for name in arm_names}

    for k in range(RECOVERY_HORIZON + 1):
        native_ref = tr[:, t + k]
        native_next_ref = tr[:, t + k + 1]
        for name in arm_names:
            recovery[name].append(
                arm_recovery_metrics(states[name], native_ref, native_next_ref, r, valid)
            )
        if k < RECOVERY_HORIZON:
            with torch.no_grad():
                cat = torch.cat([states[name] for name in arm_names], dim=0)
                nxt = model.F(cat)
            n = len(y)
            for i, name in enumerate(arm_names):
                states[name] = nxt[i*n:(i+1)*n]

    term = {}
    for name in arm_names:
        term[name] = terminal_from_state(model, states[name], t + RECOVERY_HORIZON, y)

    k = PRIMARY_HORIZON
    ret_tan = 0.5 * (
        recovery["TPLUS"][k]["total_retention"] + recovery["TMINUS"][k]["total_retention"]
    )
    ret_trans = float(np.mean([
        recovery[f"P{j}"][k]["total_retention"] for j in range(N_TRANSVERSE)
    ]))
    phase_ret = 0.5 * (
        recovery["TPLUS"][k]["signed_progress"] - recovery["TMINUS"][k]["signed_progress"]
    )

    tan_acc = 0.5 * (term["TPLUS"]["accuracy"] + term["TMINUS"]["accuracy"])
    trans_acc = float(np.mean([term[f"P{j}"]["accuracy"] for j in range(N_TRANSVERSE)]))
    tan_ce = 0.5 * (term["TPLUS"]["cross_entropy"] + term["TMINUS"]["cross_entropy"])
    trans_ce = float(np.mean([term[f"P{j}"]["cross_entropy"] for j in range(N_TRANSVERSE)]))

    drop_tan = native_terminal["accuracy"] - tan_acc
    drop_trans = native_terminal["accuracy"] - trans_acc
    ce_tan = tan_ce - native_terminal["cross_entropy"]
    ce_trans = trans_ce - native_terminal["cross_entropy"]

    return {
        "t": int(t),
        "rho": float(rho),
        "valid_fraction": geom["valid_fraction"],
        "max_geometry_error": geom["max_geometry_error"],
        "ret_tan_k3": float(ret_tan),
        "ret_trans_k3": float(ret_trans),
        "d_rec": float(ret_tan - ret_trans),
        "phase_ret_k3": float(phase_ret),
        "drop_tan": float(drop_tan),
        "drop_trans": float(drop_trans),
        "d_func": float(drop_tan - drop_trans),
        "ce_increase_tan": float(ce_tan),
        "ce_increase_trans": float(ce_trans),
        "d_func_ce": float(ce_tan - ce_trans),
        "recovery": recovery,
        "terminal": term,
    }


def analyze_family(model, seed, smoke=False):
    n = 256 if smoke else INTERVENTION_N
    y, perms = make_intervention_split(seed, n)
    tr = collect_extended_trajectory(model, y, perms)
    native_terminal = ad5.terminal_metrics(model, tr[:, base.STEPS], y)

    conditions = []
    max_geom = 0.0
    for t in TIMES:
        for rho in RHOS:
            c = analyze_condition(model, tr, y, seed, t, rho, native_terminal)
            conditions.append(c)
            max_geom = max(max_geom, c["max_geometry_error"])

    def mean_key(key):
        return float(np.mean([c[key] for c in conditions]))

    primary = {
        "ret_tan": mean_key("ret_tan_k3"),
        "ret_trans": mean_key("ret_trans_k3"),
        "d_rec": mean_key("d_rec"),
        "phase_ret": mean_key("phase_ret_k3"),
        "drop_tan": mean_key("drop_tan"),
        "drop_trans": mean_key("drop_trans"),
        "d_func": mean_key("d_func"),
        "d_func_ce": mean_key("d_func_ce"),
    }
    finite = all(math.isfinite(v) for v in primary.values())
    valid_fraction_min = float(min(c["valid_fraction"] for c in conditions))

    return {
        "n_examples": int(n),
        "times": list(TIMES),
        "rhos": list(RHOS),
        "n_transverse": N_TRANSVERSE,
        "recovery_horizon": RECOVERY_HORIZON,
        "primary_horizon": PRIMARY_HORIZON,
        "native": native_terminal,
        "conditions": conditions,
        "primary": primary,
        "ad2_regime_secondary": ad5.ad2_regime_for_seed(seed),
        "max_geometry_error": float(max_geom),
        "min_valid_fraction": valid_fraction_min,
        "geometry_valid": bool(max_geom <= GEOM_TOL),
        "finite": bool(finite),
    }


def run(seed, outdir, smoke=False):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    model, maturity, history = ad3.train_to_maturity(seed, smoke=smoke)
    valid_maturity = bool(smoke or maturity is not None)
    diag = analyze_family(model, seed, smoke=smoke)
    summary = {
        "experiment": "R8-AD6",
        "seed": int(seed),
        "expected_lineage": bool(seed in LINEAGE_SEEDS),
        "maturity_reached": valid_maturity,
        "maturity_epoch": int(maturity) if maturity is not None else (3 if smoke else None),
        "baseline_history": [] if smoke else history,
        "diagnostic": diag,
        "validity": {
            "maturity": valid_maturity,
            "finite": bool(diag["finite"]),
            "geometry": bool(diag["geometry_valid"]),
            "complete": True,
        },
        "all_valid": bool(valid_maturity and diag["finite"] and diag["geometry_valid"]),
        "environment": {
            "python": __import__("sys").version,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    save_json(out / ("smoke_summary.json" if smoke else "seed_summary.json"), summary)
    print(
        f"seed={seed} d_rec={diag['primary']['d_rec']:+.6f} "
        f"d_func={diag['primary']['d_func']:+.6f} "
        f"phase_ret={diag['primary']['phase_ret']:+.6f} "
        f"geom={diag['max_geometry_error']:.3e}",
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
