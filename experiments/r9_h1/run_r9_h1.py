from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from experiments.r9_t1 import run_r9_t1 as base
from experiments.r9_h1.models import build_arms, count_params, UpdateGateModel
from experiments.r9_h1 import diagnostics as dx
from experiments.r9_h1 import diagnostics_extra as ex

SEEDS = (2311, 2333, 2357, 2381, 2411, 2437, 2467, 2491)
TRAIN_STEPS = 360
BATCH = 32
EVAL_EPISODES = 256
PROBE_EPISODES = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
CHECKPOINT_EVERY = 40
PROTOCOL_VERSION = "R9-H1-v1"


def derive_seed(seed, name):
    h = hashlib.sha256(f"r9-h1|{seed}|{name}".encode()).digest()
    return int.from_bytes(h[:8], "little") % (2**31 - 1)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def masked_loss(logits, target):
    mask = target >= 0
    return F.cross_entropy(logits[mask], target[mask])


def _parameter_delta(before, params):
    total = 0.0
    with torch.no_grad():
        for a, p in zip(before, params):
            total += float(torch.sum((p.detach() - a) ** 2))
    return math.sqrt(total)


def _grad_norm(params):
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(torch.sum(p.grad.detach() ** 2))
    return math.sqrt(total)


def train_arm(model, seed, name, smoke=False):
    steps = 2 if smoke else TRAIN_STEPS
    batch_n = 4 if smoke else BATCH
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    trace = []
    model.train()
    for step in range(steps):
        b = base.make_batch(derive_seed(seed, f"train_{name}_{step}"), batch_n)
        opt.zero_grad(set_to_none=True)
        states = None
        if hasattr(model, "training_loss"):
            loss, states = model.training_loss(b["types"], b["payload"], b["target"])
        else:
            logits = model(b["types"], b["payload"])
            loss = masked_loss(logits, b["target"])
        checkpoint = (step % CHECKPOINT_EVERY == 0 or step == steps - 1)
        tracked_params = None
        before = None
        if checkpoint and isinstance(model, UpdateGateModel):
            tracked_params = list(model.transformer_parameters())
            before = [p.detach().clone() for p in tracked_params]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        grad_norm = None
        controller_mean = None
        if checkpoint and isinstance(model, UpdateGateModel):
            grad_norm = _grad_norm(tracked_params)
            controller_mean = float(states["UPDATE_WEIGHT"].detach().mean())
        opt.step()
        if checkpoint:
            row = {"step": int(step), "loss": float(loss.detach())}
            if isinstance(model, UpdateGateModel):
                row.update({
                    "controller_mean": controller_mean,
                    "transformer_grad_norm": grad_norm,
                    "transformer_update_norm": _parameter_delta(before, tracked_params),
                })
            trace.append(row)
    return trace


def learning_beta(trace):
    rows = [r for r in trace if "controller_mean" in r]
    if len(rows) < 4:
        return float("nan")
    gate = np.asarray([r["controller_mean"] for r in rows[:-1]], np.float64)
    loss = np.asarray([r["loss"] for r in rows[:-1]], np.float64)
    nxt = np.asarray([r["transformer_update_norm"] for r in rows[1:]], np.float64)

    def z(x):
        return (x - x.mean()) / (x.std() + 1e-8)

    X = np.column_stack([np.ones(len(gate)), z(loss), z(gate)])
    y = z(nxt)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[2])


def smoke_run(seed, outdir):
    set_seed(seed)
    arms = build_arms(derive_seed, seed)
    structure = {}
    all_finite = True
    b = base.make_batch(derive_seed(seed, "smoke_batch"), 4)
    for name, model in arms.items():
        trace = train_arm(model, seed, name, smoke=True)
        model.eval()
        with torch.no_grad():
            logits, states = model(b["types"], b["payload"], return_states=True)
        ok = tuple(logits.shape) == (4, base.SEQ_LEN, base.N_VAL)
        ok = ok and bool(torch.isfinite(logits).all())
        for s in states.values():
            ok = ok and s.shape[:2] == (4, base.SEQ_LEN) and bool(torch.isfinite(s).all())
        all_finite = all_finite and ok and all(math.isfinite(float(r["loss"])) for r in trace)
        structure[name] = {
            "params": count_params(model),
            "logit_shape": list(logits.shape),
            "state_keys": sorted(states.keys()),
            "probe_streams": list(model.probe_streams),
            "controller_signal": model.controller_signal,
            "finite": bool(ok),
        }
    summary = {
        "experiment": "R9-H1",
        "protocol_version": PROTOCOL_VERSION,
        "smoke_only": True,
        "seed": int(seed),
        "all_valid": bool(all_finite),
        "arms": structure,
        "checks": {
            "task_generator_reused_from_r9_t1": True,
            "model_forward_received_target": False,
            "causal_models_only": True,
            "scientific_metrics_emitted": False,
        },
    }
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "smoke_summary.json", summary)
    print("R9-H1 outcome-free smoke completed", flush=True)


def run_family(seed, outdir):
    if seed not in SEEDS:
        raise ValueError(f"seed {seed} is not in frozen R9-H1 seed set")
    set_seed(seed)
    arms = build_arms(derive_seed, seed)
    traces = {}
    for name, model in arms.items():
        traces[name] = train_arm(model, seed, name, smoke=False)
        print(f"trained {name}", flush=True)

    test_batch = base.make_batch(derive_seed(seed, "test_stream"), EVAL_EPISODES)
    probe_train = base.make_batch(derive_seed(seed, "probe_train"), PROBE_EPISODES)
    probe_test = base.make_batch(derive_seed(seed, "probe_test"), PROBE_EPISODES)

    architecture = {}
    anomaly_metrics = {}
    controls = {}
    audits = {}
    finite = True

    for name, model in arms.items():
        tm = dx.task_metrics(model, test_batch)
        architecture[name] = {
            "params": count_params(model),
            "test": tm,
            "train_trace": traces[name],
        }
        md = dx.mechanistic_diagnostics(
            model, probe_train, probe_test, derive_seed(seed, f"diag_shuffle_{name}")
        )
        md["answer_history"] = ex.answer_history(
            model, probe_train, probe_test, derive_seed(seed, f"answer_shuffle_{name}")
        )
        md["order_controls"] = ex.local_order_controls(
            model, probe_train, probe_test, derive_seed(seed, f"order_shuffle_{name}")
        )
        if name in ("TEACHER", "INTERROGATOR"):
            md["perturbation_response"] = ex.perturbation_response(
                model, probe_train, probe_test, name
            )
        else:
            md["perturbation_response"] = None
        anomaly_metrics[name] = md
        controls[name] = dx.control_effects(model, test_batch)
        audits[name] = dx.controller_audit(model, probe_train, probe_test)
        vals = [v for k, v in tm.items() if not k.endswith("_N")]
        finite = finite and all(math.isfinite(float(v)) for v in vals)

    teacher_off_task = dx.task_metrics(arms["TEACHER"], test_batch, "zero")
    never_task = architecture["CONTROL_RNN"]["test"]
    teacher_h = dx.teacher_off_history(
        arms["TEACHER"], arms["CONTROL_RNN"], probe_train, probe_test,
        derive_seed(seed, "teacher_off_history")
    )
    teacher_persistence = {
        "teacher_off_task": teacher_off_task,
        "never_taught_task": never_task,
        "persist_task": teacher_off_task["RETURN_EARLY_ACC"] - never_task["RETURN_EARLY_ACC"],
        "history": teacher_h,
        "persist_h": teacher_h["teacher_off"]["h1"] - teacher_h["never_taught"]["h1"],
    }

    rtr_h = anomaly_metrics["RTR"]["history"]["RNN2"]["h1"]
    trt_h = anomaly_metrics["TRT"]["history"]["T2"]["h1"]
    rtr_x = anomaly_metrics["RTR"]["cross_module"]["xform"]
    trt_x = anomaly_metrics["TRT"]["cross_module"]["xform"]
    serial_order = {
        "order_h": rtr_h - trt_h,
        "order_x": rtr_x - trt_x,
    }

    par = anomaly_metrics["PARALLEL"]
    parallel_specialization = {
        "par_h": par["history"]["RNN"]["h1"] - par["history"]["T"]["h1"],
        "complementarity": par["complementarity"],
        "linear_cka": par["pair_similarity"]["linear_cka"],
        "branch_ablation": ex.parallel_branch_ablation(arms["PARALLEL"], test_batch),
    }

    learn_beta = learning_beta(traces["UPDATE_GATE"])
    learning_coupling = {
        "learn_beta": learn_beta,
        "trace": traces["UPDATE_GATE"],
    }

    result = {
        "experiment": "R9-H1",
        "protocol_version": PROTOCOL_VERSION,
        "seed": int(seed),
        "frozen_seed": True,
        "architecture": architecture,
        "anomaly_metrics": anomaly_metrics,
        "controls": controls,
        "target_leak_audit": audits,
        "teacher_persistence": teacher_persistence,
        "serial_order": serial_order,
        "parallel_specialization": parallel_specialization,
        "learning_coupling": learning_coupling,
        "all_valid": bool(finite and math.isfinite(learn_beta)),
        "frozen": {
            "train_steps": TRAIN_STEPS,
            "batch": BATCH,
            "eval_episodes": EVAL_EPISODES,
            "probe_episodes": PROBE_EPISODES,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "seeds": list(SEEDS),
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "family_result.json", result)
    print(f"R9-H1 family seed {seed} complete", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke_run(args.seed, args.outdir)
    else:
        run_family(args.seed, args.outdir)


if __name__ == "__main__":
    main()
