import argparse
import json
from pathlib import Path

import m7r_base as base

FRESH_SEEDS = base.FRESH_SEEDS
CONDITIONS = ("MIRROR", "FIXB", "H0MIRROR")
BASE_ENGINE_BLOB = "d72d4b53b12ffa6e63b167c6bfbf25555370a2d4"


def mirror_targets(condition, phase, A, B):
    if condition == "MIRROR":
        return None, (B if phase in (1, 3) else A)
    if condition == "FIXB":
        return None, B
    if condition == "H0MIRROR":
        return (B if phase in (1, 3) else A), None
    raise ValueError(condition)


def load_reference(seed):
    p = Path(__file__).with_name("M7R_BASELINE_REFERENCE.json")
    d = json.loads(p.read_text())
    return d["seeds"][str(seed)]


def validate_reference(summary, seed):
    ref = load_reference(seed)
    if not summary.get("maturity_reached"):
        return False, ref
    q = float(summary["baseline"]["Q"])
    ok = bool(
        int(summary["maturity_epoch"]) == int(ref["M"])
        and int(summary["A_baseline_winner"]) == int(ref["A"])
        and int(summary["B_baseline_loser"]) == int(ref["B"])
        and abs(q - float(ref["baseline_Q"])) <= 1e-5
    )
    return ok, ref


def run(seed, outdir, smoke=False):
    base.CONDITIONS = CONDITIONS
    base.post_targets = mirror_targets
    base.run(seed, outdir, smoke=smoke)

    if smoke:
        p = Path(outdir) / "smoke_summary.json"
        d = json.loads(p.read_text())
        d["experiment"] = "R8-M7I"
        d["parent_engine_blob"] = BASE_ENGINE_BLOB
        d["schedule"] = ["B", "A", "B"]
        p.write_text(json.dumps(d, indent=2, sort_keys=True))
        return

    p = Path(outdir) / "seed_summary.json"
    d = json.loads(p.read_text())
    match, ref = validate_reference(d, seed)
    d["experiment"] = "R8-M7I"
    d["paired_same_lineage"] = True
    d["parent_engine_blob"] = BASE_ENGINE_BLOB
    d["schedule"] = ["B", "A", "B"]
    d["baseline_reference"] = ref
    d.setdefault("validity", {})["baseline_reference"] = bool(match)
    d["all_valid"] = bool(all(bool(v) for v in d["validity"].values()))
    p.write_text(json.dumps(d, indent=2, sort_keys=True, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if not args.smoke and args.seed not in FRESH_SEEDS:
        raise SystemExit(f"seed {args.seed} is not in the paired M7R seed set")
    run(args.seed, args.outdir, smoke=args.smoke)


if __name__ == "__main__":
    main()
