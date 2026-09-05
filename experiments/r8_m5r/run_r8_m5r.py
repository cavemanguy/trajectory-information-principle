import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

FRESH_SEEDS = (20, 41, 56, 69, 84, 99, 118, 133, 146, 161, 181, 199)


def load_parent():
    p = Path(__file__).parents[1] / "r8_m5" / "run_r8_m5.py"
    spec = importlib.util.spec_from_file_location("r8_m5_parent", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def derive_seed(seed, name):
    h = hashlib.sha256(f"R8-M5R|{seed}|{name}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if not args.smoke and args.seed not in FRESH_SEEDS:
        raise SystemExit(f"seed {args.seed} is not preregistered")

    m = load_parent()
    m.FRESH_SEEDS = FRESH_SEEDS
    m.derive_seed = derive_seed
    m.run(args.seed, args.outdir, smoke=args.smoke)

    if not args.smoke:
        path = Path(args.outdir) / "seed_summary.json"
        d = json.loads(path.read_text())
        d["experiment"] = "R8-M5R"
        d["fresh_seed"] = bool(args.seed in FRESH_SEEDS)
        d["training_implementation"] = "exact R8-M5 run_r8_m5.py reused via wrapper"
        d["seed_namespace"] = "R8-M5R"
        path.write_text(json.dumps(d, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
