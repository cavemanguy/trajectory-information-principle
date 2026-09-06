import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fnn

HERE = Path(__file__).resolve().parent
AD3_PATH = HERE.parent / "r8_ad3" / "run_r8_ad3.py"
spec = importlib.util.spec_from_file_location("r8_ad5_ad3", AD3_PATH)
ad3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ad3)
base = ad3.base

LINEAGE_SEEDS = ad3.LINEAGE_SEEDS
INTERVENTION_N = 4096
TIMES = (2, 4, 6, 8, 10)
DIR_RHOS = (0.05, 0.10, 0.20)
CURV_ANGLES_DEG = (15.0, 30.0)
EPS = 1e-8
GEOM_TOL = 1e-5
AD2_RESULT_PATH = HERE.parents[1] / "results" / "r8_ad2" / "aggregate" / "FINAL_RESULT.json"


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def make_intervention_split(seed, n):
    y = base.make_memories(n, base.derive_seed(seed, "ad5_intervention_y"))
    p = base.make_perms(n, base.derive_seed(seed, "ad5_intervention_perm"))
    return y, p


def ad2_regime_for_seed(seed):
    try:
        d = json.loads(AD2_RESULT_PATH.read_text())
        for row in d.get("rows", []):
            if int(row.get("seed")) == int(seed):
                return row.get("family_class")
    except Exception:
        pass
    return None


def collect_trajectory(model, y, perms):
    chunks = []
    model.eval()
    with torch.no_grad():
        for a in range(0, len(y), base.BATCH):
            yy = y[a:a + base.BATCH]
            pp = perms[a:a + base.BATCH]
            h0 = model.encode(yy, pp)
            chunks.append(model.trajectory(h0).cpu())
    return torch.cat(chunks, dim=0)


def deterministic_random(shape, seed, dtype):
    g = torch.Generator().manual_seed(int(seed))
    return torch.randn(shape, generator=g, dtype=dtype)


def normalize_rows(x, eps=EPS):
    n = torch.linalg.vector_norm(x, dim=1, keepdim=True)
    return x / torch.clamp(n, min=eps), n


def orthogonal_random(u, seed):
    z = deterministic_random(u.shape, seed, u.dtype)
    z = z - (z * u).sum(1, keepdim=True) * u
    q, qn = normalize_rows(z)
    bad = qn[:, 0] < 1e-7
    if bool(bad.any()):
        idx = torch.argmin(torch.abs(u[bad]), dim=1)
        fb = torch.zeros((int(bad.sum()), u.shape[1]), dtype=u.dtype)
        fb[torch.arange(len(fb)), idx] = 1.0
        ub = u[bad]
        fb = fb - (fb * ub).sum(1, keepdim=True) * ub
        fb, _ = normalize_rows(fb)
        q[bad] = fb
    return q


def orthogonal_random_two(v, e, seed):
    z = deterministic_random(v.shape, seed, v.dtype)
    z = z - (z * v).sum(1, keepdim=True) * v
    z = z - (z * e).sum(1, keepdim=True) * e
    q, qn = normalize_rows(z)
    bad = qn[:, 0] < 1e-7
    if bool(bad.any()):
        vb = v[bad]
        eb = e[bad]
        idx = torch.argmin(torch.abs(vb) + torch.abs(eb), dim=1)
        fb = torch.zeros((int(bad.sum()), v.shape[1]), dtype=v.dtype)
        fb[torch.arange(len(fb)), idx] = 1.0
        fb = fb - (fb * vb).sum(1, keepdim=True) * vb
        fb = fb - (fb * eb).sum(1, keepdim=True) * eb
        fb, _ = normalize_rows(fb)
        q[bad] = fb
    return q


def rollout_to_12(model, h, current_index):
    z = h
    with torch.no_grad():
        for _ in range(int(current_index), base.STEPS):
            z = model.F(z)
    return z


def terminal_metrics(model, h12, y):
    acc_rel = []
    ce_rel = []
    with torch.no_grad():
        for r in range(base.N_REL):
            logits = model.headT[r](h12)
            acc_rel.append(float((logits.argmax(1) == y[:, r]).float().mean()))
            ce_rel.append(float(Fnn.cross_entropy(logits, y[:, r], reduction="mean")))
    return {
        "accuracy": float(np.mean(acc_rel)),
        "cross_entropy": float(np.mean(ce_rel)),
        "per_relation_accuracy": acc_rel,
        "per_relation_cross_entropy": ce_rel,
    }


def terminal_metrics_arms(model, states, current_index, y, arm_names):
    n = len(y)
    cat = torch.cat([states[name] for name in arm_names], dim=0)
    final = rollout_to_12(model, cat, current_index)
    out = {}
    for i, name in enumerate(arm_names):
        out[name] = terminal_metrics(model, final[i*n:(i+1)*n], y)
    return out


def phase_a_states(native_t, native_next, seed, t, rho):
    d = native_next - native_t
    u, m = normalize_rows(d)
    valid = m[:, 0] >= 1e-7
    rho = float(rho)
    r = rho * m

    q = orthogonal_random(u, base.derive_seed(seed, f"ad5_dir_q_{t}_{rho:.3f}"))
    theta = 2.0 * math.asin(rho / 2.0)
    uprime = math.cos(theta) * u + math.sin(theta) * q

    dir_state = native_t + m * uprime
    mag_plus = native_t + (1.0 + rho) * m * u
    mag_minus = native_t + (1.0 - rho) * m * u

    qr = deterministic_random(d.shape, base.derive_seed(seed, f"ad5_rand_q_{t}_{rho:.3f}"), d.dtype)
    qr, _ = normalize_rows(qr)
    rand_state = native_next + r * qr

    if bool((~valid).any()):
        for z in (dir_state, mag_plus, mag_minus, rand_state):
            z[~valid] = native_next[~valid]

    if bool(valid.any()):
        vv = valid
        native_m = m[vv, 0]
        target_r = r[vv, 0]
        dir_step = torch.linalg.vector_norm((dir_state - native_t)[vv], dim=1)
        dir_shift = torch.linalg.vector_norm((dir_state - native_next)[vv], dim=1)
        mp_shift = torch.linalg.vector_norm((mag_plus - native_next)[vv], dim=1)
        mm_shift = torch.linalg.vector_norm((mag_minus - native_next)[vv], dim=1)
        magp_u, _ = normalize_rows((mag_plus - native_t)[vv])
        magm_u, _ = normalize_rows((mag_minus - native_t)[vv])
        errs = [
            torch.max(torch.abs(dir_step - native_m)),
            torch.max(torch.abs(dir_shift - target_r)),
            torch.max(torch.abs(mp_shift - target_r)),
            torch.max(torch.abs(mm_shift - target_r)),
            torch.max(torch.linalg.vector_norm(magp_u - u[vv], dim=1)),
            torch.max(torch.linalg.vector_norm(magm_u - u[vv], dim=1)),
        ]
        geom_error = float(torch.stack(errs).max())
    else:
        geom_error = 0.0

    return {
        "DIR": dir_state,
        "MAG_PLUS": mag_plus,
        "MAG_MINUS": mag_minus,
        "RAND": rand_state,
    }, {
        "valid_fraction": float(valid.float().mean()),
        "max_geometry_error": geom_error,
    }


def phase_b_states(prev, cur, native_next, seed, t, angle_deg):
    dprev = cur - prev
    dnext = native_next - cur

    v, prev_m = normalize_rows(dprev)
    _, next_m = normalize_rows(dnext)
    a = (dnext * v).sum(1, keepdim=True)
    p = dnext - a * v
    e, p_m = normalize_rows(p)

    valid = (prev_m[:, 0] >= 1e-7) & (next_m[:, 0] >= 1e-7) & (p_m[:, 0] >= 1e-7)
    q = orthogonal_random_two(v, e, base.derive_seed(seed, f"ad5_curv_q_{t}_{angle_deg:.1f}"))

    alpha = math.radians(float(angle_deg))
    eprime = math.cos(alpha) * e + math.sin(alpha) * q
    d_az = a * v + p_m * eprime
    az_state = cur + d_az

    r = torch.linalg.vector_norm(d_az - dnext, dim=1, keepdim=True)
    ratio = torch.clamp(r / torch.clamp(2.0 * next_m, min=EPS), min=0.0, max=1.0)
    delta = 2.0 * torch.asin(ratio)

    phi = torch.atan2(p_m, a)
    plus_ok = (phi + delta) <= math.pi
    phi_new = torch.where(plus_ok, phi + delta, phi - delta)
    d_pol = next_m * (torch.cos(phi_new) * v + torch.sin(phi_new) * e)
    pol_state = cur + d_pol

    if bool((~valid).any()):
        az_state[~valid] = native_next[~valid]
        pol_state[~valid] = native_next[~valid]

    if bool(valid.any()):
        vv = valid
        az_m = torch.linalg.vector_norm((az_state - cur)[vv], dim=1)
        pol_m = torch.linalg.vector_norm((pol_state - cur)[vv], dim=1)
        native_m = next_m[vv, 0]
        az_shift = torch.linalg.vector_norm((az_state - native_next)[vv], dim=1)
        pol_shift = torch.linalg.vector_norm((pol_state - native_next)[vv], dim=1)

        native_cos = ((dprev[vv] * dnext[vv]).sum(1) /
                      (prev_m[vv, 0] * next_m[vv, 0]))
        az_d = (az_state - cur)[vv]
        az_cos = ((dprev[vv] * az_d).sum(1) /
                  (prev_m[vv, 0] * torch.linalg.vector_norm(az_d, dim=1)))

        errs = [
            torch.max(torch.abs(az_m - native_m)),
            torch.max(torch.abs(pol_m - native_m)),
            torch.max(torch.abs(az_shift - pol_shift)),
            torch.max(torch.abs(az_cos - native_cos)),
        ]
        geom_error = float(torch.stack(errs).max())
    else:
        geom_error = 0.0

    return {"AZ": az_state, "POL": pol_state}, {
        "valid_fraction": float(valid.float().mean()),
        "max_geometry_error": geom_error,
    }


def analyze_family(model, seed, smoke=False):
    n = 256 if smoke else INTERVENTION_N
    y, perms = make_intervention_split(seed, n)
    tr = collect_trajectory(model, y, perms)
    native = terminal_metrics(model, tr[:, -1], y)

    phase_a = []
    phase_b = []
    max_geom = 0.0

    for t in TIMES:
        ht = tr[:, t]
        hn = tr[:, t + 1]
        for rho in DIR_RHOS:
            states, geom = phase_a_states(ht, hn, seed, t, rho)
            max_geom = max(max_geom, geom["max_geometry_error"])
            metrics = terminal_metrics_arms(
                model, states, t + 1, y,
                ("DIR", "MAG_PLUS", "MAG_MINUS", "RAND"),
            )
            drop_dir = native["accuracy"] - metrics["DIR"]["accuracy"]
            drop_mp = native["accuracy"] - metrics["MAG_PLUS"]["accuracy"]
            drop_mm = native["accuracy"] - metrics["MAG_MINUS"]["accuracy"]
            drop_rand = native["accuracy"] - metrics["RAND"]["accuracy"]
            ce_dir = metrics["DIR"]["cross_entropy"] - native["cross_entropy"]
            ce_mp = metrics["MAG_PLUS"]["cross_entropy"] - native["cross_entropy"]
            ce_mm = metrics["MAG_MINUS"]["cross_entropy"] - native["cross_entropy"]
            ce_rand = metrics["RAND"]["cross_entropy"] - native["cross_entropy"]
            phase_a.append({
                "t": int(t),
                "rho": float(rho),
                "valid_fraction": geom["valid_fraction"],
                "max_geometry_error": geom["max_geometry_error"],
                "drop_dir": float(drop_dir),
                "drop_mag_plus": float(drop_mp),
                "drop_mag_minus": float(drop_mm),
                "drop_rand": float(drop_rand),
                "k_dir": float(drop_dir - 0.5 * (drop_mp + drop_mm)),
                "ce_increase_dir": float(ce_dir),
                "ce_increase_mag_plus": float(ce_mp),
                "ce_increase_mag_minus": float(ce_mm),
                "ce_increase_rand": float(ce_rand),
                "k_dir_ce": float(ce_dir - 0.5 * (ce_mp + ce_mm)),
                "metrics": metrics,
            })

        hp = tr[:, t - 1]
        for angle in CURV_ANGLES_DEG:
            states, geom = phase_b_states(hp, ht, hn, seed, t, angle)
            max_geom = max(max_geom, geom["max_geometry_error"])
            metrics = terminal_metrics_arms(model, states, t + 1, y, ("AZ", "POL"))
            drop_az = native["accuracy"] - metrics["AZ"]["accuracy"]
            drop_pol = native["accuracy"] - metrics["POL"]["accuracy"]
            ce_az = metrics["AZ"]["cross_entropy"] - native["cross_entropy"]
            ce_pol = metrics["POL"]["cross_entropy"] - native["cross_entropy"]
            phase_b.append({
                "t": int(t),
                "angle_deg": float(angle),
                "valid_fraction": geom["valid_fraction"],
                "max_geometry_error": geom["max_geometry_error"],
                "drop_az": float(drop_az),
                "drop_pol": float(drop_pol),
                "k_curv": float(drop_az - drop_pol),
                "ce_increase_az": float(ce_az),
                "ce_increase_pol": float(ce_pol),
                "k_curv_ce": float(ce_az - ce_pol),
                "metrics": metrics,
            })

    def mean_key(rows, key):
        return float(np.mean([r[key] for r in rows]))

    primary = {
        "k_dir": mean_key(phase_a, "k_dir"),
        "k_dir_ce": mean_key(phase_a, "k_dir_ce"),
        "drop_dir": mean_key(phase_a, "drop_dir"),
        "drop_mag": float(np.mean([
            0.5 * (r["drop_mag_plus"] + r["drop_mag_minus"]) for r in phase_a
        ])),
        "drop_rand": mean_key(phase_a, "drop_rand"),
        "k_curv": mean_key(phase_b, "k_curv"),
        "k_curv_ce": mean_key(phase_b, "k_curv_ce"),
        "drop_az": mean_key(phase_b, "drop_az"),
        "drop_pol": mean_key(phase_b, "drop_pol"),
    }

    finite = all(math.isfinite(v) for v in primary.values())
    return {
        "n_examples": int(n),
        "times": list(TIMES),
        "dir_rhos": list(DIR_RHOS),
        "curv_angles_deg": list(CURV_ANGLES_DEG),
        "native": native,
        "phase_a": phase_a,
        "phase_b": phase_b,
        "primary": primary,
        "ad2_regime_secondary": ad2_regime_for_seed(seed),
        "max_geometry_error": float(max_geom),
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
        "experiment": "R8-AD5",
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
        f"seed={seed} k_dir={diag['primary']['k_dir']:+.6f} "
        f"k_dir_ce={diag['primary']['k_dir_ce']:+.6f} "
        f"k_curv={diag['primary']['k_curv']:+.6f} "
        f"k_curv_ce={diag['primary']['k_curv_ce']:+.6f} "
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
