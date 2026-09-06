import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

EXPECTED = (1782, 1801, 1819, 1838, 1856, 1875, 1893, 1912, 1930, 1949, 1967, 1986)
FAMILY_CLASSES = ("FIXED", "PERIODIC", "SENSITIVE", "QUASIPERIODIC_LIKE", "REGULAR_NONPERIODIC", "MIXED")


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def classify_counts(counts, n_valid):
    if n_valid < 12:
        return "D6 — invalid/incomplete"
    if counts.get("SENSITIVE", 0) >= 8:
        return "D1 — sensitive nonconvergent dynamics supported"
    regular = counts.get("QUASIPERIODIC_LIKE", 0) + counts.get("REGULAR_NONPERIODIC", 0)
    if regular >= 8 and counts.get("SENSITIVE", 0) <= 2:
        return "D2 — regular nonperiodic dynamics supported"
    periodic = counts.get("FIXED", 0) + counts.get("PERIODIC", 0)
    if periodic >= 8:
        return "D3 — periodic/fixed dynamics supported"
    represented = sum(1 for c in FAMILY_CLASSES if counts.get(c, 0) > 0)
    if represented >= 3:
        return "D4 — heterogeneous learned regimes"
    return "D5 — unresolved/mixed"


def aggregate(root):
    files = sorted(Path(root).glob("**/seed_summary.json"))
    by_seed = {}
    for p in files:
        d = json.loads(p.read_text())
        if d.get("experiment") != "R8-AD2":
            continue
        by_seed[int(d["seed"])] = d

    rows = []
    valid = []
    for seed in EXPECTED:
        d = by_seed.get(seed)
        if d is None:
            rows.append({"seed": seed, "valid": False, "reason": "missing"})
            continue
        ok = bool(d.get("all_valid"))
        q = d.get("diagnostic", {})
        row = {
            "seed": seed,
            "valid": ok,
            "M": d.get("maturity_epoch"),
            "family_class": q.get("family_class"),
            "dominant_state_fraction": q.get("dominant_state_fraction"),
            "median_ftle": q.get("ftle", {}).get("median"),
            "median_top8": q.get("top8_power_fraction", {}).get("median"),
            "median_spectral_entropy": q.get("spectral_entropy", {}).get("median"),
            "state_class_fractions": q.get("state_class_fractions", {}),
        }
        rows.append(row)
        if ok:
            valid.append(row)

    counts = Counter(r["family_class"] for r in valid)
    classification = classify_counts(counts, len(valid))
    result = {
        "experiment": "R8-AD2",
        "classification": classification,
        "n_valid": len(valid),
        "expected_seeds": list(EXPECTED),
        "family_class_counts": {c: int(counts.get(c, 0)) for c in FAMILY_CLASSES},
        "rows": rows,
        "claim_boundary": (
            "R8-AD2 classifies finite-horizon native encoded-state dynamics. SENSITIVE denotes a positive "
            "finite-time Lyapunov-like estimate under the frozen procedure, not formal proof of chaos; "
            "QUASIPERIODIC_LIKE denotes low-sensitivity spectrally concentrated nonperiodic motion, not a "
            "formal quasiperiodic invariant set; recurrence above period 512 is not excluded."
        ),
    }
    if valid:
        result["median_ftle_across_families"] = stats([r["median_ftle"] for r in valid])
        result["median_top8_across_families"] = stats([r["median_top8"] for r in valid])
        result["median_spectral_entropy_across_families"] = stats([r["median_spectral_entropy"] for r in valid])
        for state_class in ("FIXED", "PERIODIC", "SENSITIVE", "QUASIPERIODIC_LIKE", "REGULAR_NONPERIODIC"):
            result[f"state_fraction_{state_class}"] = stats([
                r["state_class_fractions"].get(state_class, 0.0) for r in valid
            ])
    return result


def write_md(result, path):
    counts = result["family_class_counts"]
    lines = [
        "# R8-AD2 Final Result — Native Dynamical Regime Classification",
        "",
        f"**Primary classification:** {result['classification']}",
        "",
        "## Family classifications",
        "",
    ]
    for c in FAMILY_CLASSES:
        lines.append(f"- {c}: **{counts.get(c, 0)}/12**")
    if result.get("n_valid", 0) == 12:
        lines += [
            "",
            "## Cross-family diagnostics",
            "",
            f"- family median FTLE: mean {result['median_ftle_across_families']['mean']:+.6f}; median {result['median_ftle_across_families']['median']:+.6f}; range [{result['median_ftle_across_families']['min']:+.6f}, {result['median_ftle_across_families']['max']:+.6f}]",
            f"- family median top-8 spectral power: mean {result['median_top8_across_families']['mean']:.6f}; median {result['median_top8_across_families']['median']:.6f}",
            f"- family median spectral entropy: mean {result['median_spectral_entropy_across_families']['mean']:.6f}; median {result['median_spectral_entropy_across_families']['median']:.6f}",
        ]
    lines += [
        "",
        "## Per-family values",
        "",
        "| seed | M | class | dominant frac | median FTLE | median top8 | median entropy |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for r in result["rows"]:
        if not r.get("valid"):
            lines.append(f"| {r['seed']} | — | INVALID | — | — | — | — |")
        else:
            lines.append(
                f"| {r['seed']} | {r['M']} | {r['family_class']} | {r['dominant_state_fraction']:.4f} | "
                f"{r['median_ftle']:+.6f} | {r['median_top8']:.6f} | {r['median_spectral_entropy']:.6f} |"
            )
    lines += ["", "## Claim boundary", "", result["claim_boundary"], ""]
    Path(path).write_text("\n".join(lines))


def self_check():
    cases = [
        ({"SENSITIVE": 8, "MIXED": 4}, "D1"),
        ({"QUASIPERIODIC_LIKE": 5, "REGULAR_NONPERIODIC": 3, "SENSITIVE": 2, "MIXED": 2}, "D2"),
        ({"PERIODIC": 7, "FIXED": 1, "MIXED": 4}, "D3"),
        ({"SENSITIVE": 4, "PERIODIC": 4, "MIXED": 4}, "D4"),
        ({"SENSITIVE": 6, "MIXED": 6}, "D5"),
    ]
    for raw, prefix in cases:
        got = classify_counts(Counter(raw), 12)
        assert got.startswith(prefix), (raw, got)
    assert classify_counts(Counter(), 11).startswith("D6")
    print("R8-AD2 classifier self-check ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root")
    ap.add_argument("--outdir")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        self_check()
        return
    if not args.root or not args.outdir:
        raise SystemExit("--root and --outdir required")
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    result = aggregate(args.root)
    save_json(out / "FINAL_RESULT.json", result)
    write_md(result, out / "FINAL_RESULT.md")
    print(result["classification"])


if __name__ == "__main__":
    main()
