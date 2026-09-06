import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
AD3_PATH = HERE.parent / "r8_ad3" / "run_r8_ad3.py"
spec = importlib.util.spec_from_file_location("r8_ad4_ad3", AD3_PATH)
ad3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ad3)
base = ad3.base

LINEAGE_SEEDS = ad3.LINEAGE_SEEDS
PROBE_TRAIN = 12000
PROBE_VAL = 4000
PROBE_TEST = 8000
INTERNAL_TIMES = tuple(range(1, 12))
CURVATURE_TIMES = tuple(range(2, 12))
LAMBDAS = ad3.LAMBDAS
PRIMARY_REPS = ("STATE", "FULL", "SHUFFLED_FULL", "DIRECTION", "MAGNITUDE")
SECONDARY_REPS = ("ABS_PATTERN", "SIGN_PATTERN", "FULL_CURVATURE", "FULL_TURN")
EPS = 1e-8


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def make_probe_split(seed, split, n):
    y = base.make_memories(n, base.derive_seed(seed, f"ad4_probe_{split}_y"))
    p = base.make_perms(n, base.derive_seed(seed, f"ad4_probe_{split}_perm"))
    return y, p


def kinematics(tr, t):
    h = tr[:, t, :]
    prev = tr[:, t - 1, :]
    d = h - prev
    mag = np.linalg.norm(d, axis=1, keepdims=True)
    direction = d / np.maximum(mag, EPS)
    out = {"h": h, "d": d, "mag": mag, "direction": direction}
    if t >= 2:
        prevprev = tr[:, t - 2, :]
        dprev = prev - prevprev
        accel = d - dprev
        denom = np.maximum(np.linalg.norm(d, axis=1) * np.linalg.norm(dprev, axis=1), EPS)
        turn = ((d * dprev).sum(axis=1) / denom)[:, None]
        out.update({"dprev": dprev, "accel": accel, "turn": turn})
    return out


def feature_matrix(tr, t, rep, seed, split):
    z = kinematics(tr, t)
    h, d, mag, direction = z["h"], z["d"], z["mag"], z["direction"]
    if rep == "STATE":
        return h
    if rep == "FULL":
        return np.concatenate([h, d], axis=1)
    if rep == "SHUFFLED_FULL":
        rng = np.random.default_rng(base.derive_seed(seed, f"ad4_shuffle_{split}_{t}"))
        order = rng.permutation(len(d))
        return np.concatenate([h, d[order]], axis=1)
    if rep == "DIRECTION":
        return np.concatenate([h, direction], axis=1)
    if rep == "MAGNITUDE":
        return np.concatenate([h, np.log(mag + EPS)], axis=1)
    if rep == "ABS_PATTERN":
        return np.concatenate([h, np.abs(d)], axis=1)
    if rep == "SIGN_PATTERN":
        return np.concatenate([h, np.sign(d)], axis=1)
    if rep == "FULL_CURVATURE":
        if t < 2:
            raise ValueError("FULL_CURVATURE requires t>=2")
        return np.concatenate([h, d, z["accel"]], axis=1)
    if rep == "FULL_TURN":
        if t < 2:
            raise ValueError("FULL_TURN requires t>=2")
        return np.concatenate([h, d, z["turn"]], axis=1)
    raise ValueError(rep)


def fit_rep(trs, ys, ytr_oh, t, rep, seed):
    xtr = feature_matrix(trs["train"], t, rep, seed, "train")
    xv = feature_matrix(trs["val"], t, rep, seed, "val")
    xt = feature_matrix(trs["test"], t, rep, seed, "test")
    return ad3.fit_select_evaluate(xtr, ytr_oh, ys["train"], xv, ys["val"], xt, ys["test"])


def fit_coords(trs, ys, ytr_oh, t, seed):
    zs = {s: kinematics(trs[s], t) for s in ("train", "val", "test")}
    candidates = []
    for j in range(base.LATENT):
        mats = {
            s: np.concatenate([zs[s]["h"], zs[s]["d"][:, j:j+1]], axis=1)
            for s in zs
        }
        rec = ad3.fit_select_evaluate(
            mats["train"], ytr_oh, ys["train"], mats["val"], ys["val"], mats["test"], ys["test"]
        )
        candidates.append((rec["val_mean_accuracy"], -j, j, rec))
    candidates.sort(reverse=True, key=lambda x: (x[0], x[1]))
    best_j = int(candidates[0][2])
    best_rec = candidates[0][3]
    top4 = [int(x[2]) for x in candidates[:4]]
    mats4 = {
        s: np.concatenate([zs[s]["h"], zs[s]["d"][:, top4]], axis=1)
        for s in zs
    }
    top4_rec = ad3.fit_select_evaluate(
        mats4["train"], ytr_oh, ys["train"], mats4["val"], ys["val"], mats4["test"], ys["test"]
    )
    return {
        "best_coordinate": best_j,
        "best_single": best_rec,
        "top4_coordinates": top4,
        "top4": top4_rec,
        "candidate_validation_scores": [float(x[0]) for x in sorted(candidates, key=lambda q: q[2])],
    }


def mean_internal(by_time, rep, times):
    vals = [by_time[str(t)][rep]["test_mean_accuracy"] for t in times]
    per_rel = np.asarray([by_time[str(t)][rep]["test_per_relation"] for t in times], dtype=np.float64).mean(axis=0)
    return {"mean_accuracy": float(np.mean(vals)), "per_relation_accuracy": per_rel.tolist()}


def probe_family(model, seed, smoke=False):
    sizes = {
        "train": 384 if smoke else PROBE_TRAIN,
        "val": 128 if smoke else PROBE_VAL,
        "test": 256 if smoke else PROBE_TEST,
    }
    ys, trs = {}, {}
    for split, n in sizes.items():
        y, p = make_probe_split(seed, split, n)
        ys[split] = y.numpy().astype(np.int64)
        trs[split] = ad3.collect_trajectory(model, y, p)

    ytr_oh = ad3.one_hot_targets(ys["train"])
    by_time = {}
    coord_by_time = {}
    for t in INTERNAL_TIMES:
        rec = {}
        for rep in PRIMARY_REPS:
            rec[rep] = fit_rep(trs, ys, ytr_oh, t, rep, seed)
        rec["ABS_PATTERN"] = fit_rep(trs, ys, ytr_oh, t, "ABS_PATTERN", seed)
        rec["SIGN_PATTERN"] = fit_rep(trs, ys, ytr_oh, t, "SIGN_PATTERN", seed)
        if t >= 2:
            rec["FULL_CURVATURE"] = fit_rep(trs, ys, ytr_oh, t, "FULL_CURVATURE", seed)
            rec["FULL_TURN"] = fit_rep(trs, ys, ytr_oh, t, "FULL_TURN", seed)
        by_time[str(t)] = rec
        coord_by_time[str(t)] = fit_coords(trs, ys, ytr_oh, t, seed)
        print(
            f"seed={seed} t={t} state={rec['STATE']['test_mean_accuracy']:.4f} "
            f"full={rec['FULL']['test_mean_accuracy']:.4f} dir={rec['DIRECTION']['test_mean_accuracy']:.4f} "
            f"mag={rec['MAGNITUDE']['test_mean_accuracy']:.4f}",
            flush=True,
        )

    primary = {rep: mean_internal(by_time, rep, INTERNAL_TIMES) for rep in PRIMARY_REPS}
    secondary = {
        "ABS_PATTERN": mean_internal(by_time, "ABS_PATTERN", INTERNAL_TIMES),
        "SIGN_PATTERN": mean_internal(by_time, "SIGN_PATTERN", INTERNAL_TIMES),
        "FULL_CURVATURE": mean_internal(by_time, "FULL_CURVATURE", CURVATURE_TIMES),
        "FULL_TURN": mean_internal(by_time, "FULL_TURN", CURVATURE_TIMES),
        "FULL_CURV_WINDOW": mean_internal(by_time, "FULL", CURVATURE_TIMES),
        "STATE_CURV_WINDOW": mean_internal(by_time, "STATE", CURVATURE_TIMES),
        "BEST_SINGLE_COORD": {
            "mean_accuracy": float(np.mean([coord_by_time[str(t)]["best_single"]["test_mean_accuracy"] for t in INTERNAL_TIMES]))
        },
        "TOP4_COORD": {
            "mean_accuracy": float(np.mean([coord_by_time[str(t)]["top4"]["test_mean_accuracy"] for t in INTERNAL_TIMES]))
        },
    }

    delta_full = primary["FULL"]["mean_accuracy"] - primary["STATE"]["mean_accuracy"]
    delta_shuffle = primary["FULL"]["mean_accuracy"] - primary["SHUFFLED_FULL"]["mean_accuracy"]
    delta_dir = primary["DIRECTION"]["mean_accuracy"] - primary["STATE"]["mean_accuracy"]
    delta_mag = primary["MAGNITUDE"]["mean_accuracy"] - primary["STATE"]["mean_accuracy"]
    delta_curv = secondary["FULL_CURVATURE"]["mean_accuracy"] - secondary["FULL_CURV_WINDOW"]["mean_accuracy"]
    delta_turn = secondary["FULL_TURN"]["mean_accuracy"] - secondary["FULL_CURV_WINDOW"]["mean_accuracy"]
    ret_dir = float(delta_dir / delta_full) if delta_full > 0 else None
    ret_mag = float(delta_mag / delta_full) if delta_full > 0 else None

    finite = True
    for t in INTERNAL_TIMES:
        for rep, rr in by_time[str(t)].items():
            finite = finite and math.isfinite(rr["test_mean_accuracy"])
        finite = finite and math.isfinite(coord_by_time[str(t)]["best_single"]["test_mean_accuracy"])
        finite = finite and math.isfinite(coord_by_time[str(t)]["top4"]["test_mean_accuracy"])

    return {
        "probe_sizes": sizes,
        "internal_times": list(INTERNAL_TIMES),
        "curvature_times": list(CURVATURE_TIMES),
        "by_time": by_time,
        "coordinate_by_time": coord_by_time,
        "primary": primary,
        "secondary": secondary,
        "delta_full": float(delta_full),
        "delta_shuffle": float(delta_shuffle),
        "delta_direction": float(delta_dir),
        "delta_magnitude": float(delta_mag),
        "ret_direction": ret_dir,
        "ret_magnitude": ret_mag,
        "delta_curvature": float(delta_curv),
        "delta_turn": float(delta_turn),
        "finite": bool(finite),
    }


def run(seed, outdir, smoke=False):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    model, maturity, history = ad3.train_to_maturity(seed, smoke=smoke)
    valid_maturity = bool(smoke or maturity is not None)
    diag = probe_family(model, seed, smoke=smoke)
    summary = {
        "experiment": "R8-AD4",
        "seed": int(seed),
        "expected_lineage": bool(seed in LINEAGE_SEEDS),
        "maturity_reached": valid_maturity,
        "maturity_epoch": int(maturity) if maturity is not None else (3 if smoke else None),
        "baseline_history": [] if smoke else history,
        "diagnostic": diag,
        "validity": {"maturity": valid_maturity, "finite": bool(diag["finite"]), "complete": True},
        "all_valid": bool(valid_maturity and diag["finite"]),
        "environment": {"python": __import__("sys").version, "torch": torch.__version__, "numpy": np.__version__},
    }
    save_json(out / ("smoke_summary.json" if smoke else "seed_summary.json"), summary)
    print(
        f"seed={seed} full={diag['delta_full']:+.6f} dir={diag['delta_direction']:+.6f} "
        f"mag={diag['delta_magnitude']:+.6f} curv={diag['delta_curvature']:+.6f}",
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
