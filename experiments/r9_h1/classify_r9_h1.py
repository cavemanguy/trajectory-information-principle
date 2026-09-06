from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SEEDS = (2311, 2333, 2357, 2381, 2411, 2437, 2467, 2491)
BOOT_N = 20000
BOOT_SEED = 91091
CONTROLLER_ARMS = ("TEACHER", "INTERROGATOR", "ATTN_CONTROLLER", "LOWRANK_MOD", "UPDATE_GATE")


def load_families(root):
    files = sorted(Path(root).glob("**/family_result.json"))
    rows = [json.loads(p.read_text()) for p in files]
    by_seed = {int(r["seed"]): r for r in rows}
    if set(by_seed) != set(SEEDS):
        raise RuntimeError(f"expected seeds {SEEDS}, found {sorted(by_seed)}")
    for s in SEEDS:
        r = by_seed[s]
        if r.get("experiment") != "R9-H1" or r.get("protocol_version") != "R9-H1-v1":
            raise RuntimeError(f"protocol mismatch for seed {s}")
        if not r.get("all_valid", False):
            raise RuntimeError(f"invalid family artifact for seed {s}")
    return [by_seed[s] for s in SEEDS]


def boot(values):
    x = np.asarray(values, np.float64)
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(x), size=(BOOT_N, len(x)))
    means = x[idx].mean(1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(x.mean()), float(lo), float(hi)


def classify_signed(values):
    mean, lo, hi = boot(values)
    pos = int(np.sum(np.asarray(values) > 0))
    neg = int(np.sum(np.asarray(values) < 0))
    if pos >= 6 and lo > 0:
        cls = "A+"
    elif neg >= 6 and hi < 0:
        cls = "A-"
    else:
        cls = "A0"
    return {
        "class": cls,
        "mean": mean,
        "ci95": [lo, hi],
        "positive": pos,
        "negative": neg,
        "values": list(map(float, values)),
    }


def mean_metric(rows, arm, metric):
    return float(np.mean([r["architecture"][arm]["test"][metric] for r in rows]))


def validity(rows):
    arms = rows[0]["architecture"].keys()
    out = {}
    for arm in arms:
        data = mean_metric(rows, arm, "DATA_ACC")
        ret = mean_metric(rows, arm, "RETURN_EARLY_ACC")
        out[arm] = {
            "class": "V+" if data >= 0.20 and ret >= 0.15 else "V-",
            "DATA_ACC": data,
            "RETURN_EARLY_ACC": ret,
        }
    return out


def qualify(cls, arm, valid):
    if cls in ("A+", "A-") and valid[arm]["class"] == "V-":
        return "LOW_COMPETENCE_" + cls
    return cls


def aggregate_audits(rows):
    out = {}
    for arm in rows[0]["target_leak_audit"]:
        audits = [r["target_leak_audit"][arm] for r in rows]
        if audits[0] is None:
            out[arm] = {"confounded": False, "reason": None}
            continue
        sig = float(np.mean([a["signal_target_acc"] for a in audits]))
        ctx = float(np.mean([a["payload_context_acc"] for a in audits]))
        arm_acc = mean_metric(rows, arm, "DATA_ACC")
        c1 = sig >= 0.50 and sig >= arm_acc - 0.10
        c2 = sig - ctx >= 0.35
        reasons = []
        if c1:
            reasons.append("controller signal >=0.50 target decode and within 0.10 of arm DATA accuracy")
        if c2:
            reasons.append("controller signal exceeds payload-only context by >=0.35")
        out[arm] = {
            "confounded": bool(c1 or c2),
            "mean_signal_target_acc": sig,
            "mean_payload_context_acc": ctx,
            "mean_arm_DATA_ACC": arm_acc,
            "reason": "; ".join(reasons) if reasons else None,
        }
    return out


def aggregate_history_field(rows, valid, field):
    out = {}
    for arm, md in rows[0]["anomaly_metrics"].items():
        out[arm] = {}
        for stream in md[field]:
            one = {}
            for metric in ("h1", "h2", "hshuf"):
                vals = [r["anomaly_metrics"][arm][field][stream][metric] for r in rows]
                c = classify_signed(vals)
                c["qualified_class"] = qualify(c["class"], arm, valid)
                one[metric] = c
            out[arm][stream] = one
    return out


def aggregate_complementarity(rows, valid, audits):
    out = {}
    for arm, md in rows[0]["anomaly_metrics"].items():
        if md["complementarity"] is None:
            continue
        comp = classify_signed([r["anomaly_metrics"][arm]["complementarity"]["comp"] for r in rows])
        shuf = classify_signed([r["anomaly_metrics"][arm]["complementarity"]["comp_shuf"] for r in rows])
        if audits.get(arm, {}).get("confounded", False):
            overall = "AX"
        elif comp["class"] == "A+" and shuf["class"] == "A+":
            overall = "A+"
        else:
            overall = "A0"
        if overall == "A+" and valid[arm]["class"] == "V-":
            overall = "LOW_COMPETENCE_A+"
        out[arm] = {"class": overall, "comp": comp, "comp_shuf": shuf}
    return out


def aggregate_cross(rows, valid):
    out = {}
    for arm in rows[0]["anomaly_metrics"]:
        vals = [r["anomaly_metrics"][arm]["cross_module"]["xform"] for r in rows]
        c = classify_signed(vals)
        c["qualified_class"] = qualify(c["class"], arm, valid)
        out[arm] = c
    return out


def aggregate_reactivation(rows, valid):
    out = {}
    for arm, md in rows[0]["anomaly_metrics"].items():
        out[arm] = {}
        for stream in md["reactivation"]:
            vals = [r["anomaly_metrics"][arm]["reactivation"][stream]["react"] for r in rows]
            c = classify_signed(vals)
            c["qualified_class"] = qualify(c["class"], arm, valid)
            out[arm][stream] = c
    return out


def aggregate_order_controls(rows, valid):
    out = {}
    for arm, md in rows[0]["anomaly_metrics"].items():
        out[arm] = {}
        for stream in md["order_controls"]:
            def vals(diff_to):
                return [
                    r["anomaly_metrics"][arm]["order_controls"][stream]["ordered_acc"]
                    - r["anomaly_metrics"][arm]["order_controls"][stream][diff_to]
                    for r in rows
                ]
            one = {}
            for label, key in [
                ("ordered_vs_shuffled", "shuffled_order_acc"),
                ("ordered_vs_reversed", "reversed_acc"),
                ("ordered_vs_integrated", "integrated_acc"),
                ("ordered_vs_first", "first_transient_acc"),
                ("ordered_vs_final", "final_transient_acc"),
            ]:
                c = classify_signed(vals(key))
                c["qualified_class"] = qualify(c["class"], arm, valid)
                one[label] = c
            out[arm][stream] = one
    return out


def aggregate_controls(rows, valid, audits):
    out = {}
    for arm in rows[0]["controls"]:
        if rows[0]["controls"][arm] is None:
            continue
        pieces = {}
        passes = 0
        for key in ("ctrl_shuf", "ctrl_zero", "ctrl_rand", "ctrl_wrong"):
            c = classify_signed([r["controls"][arm][key] for r in rows])
            pieces[key] = c
            passes += int(c["class"] == "A+")
        if audits.get(arm, {}).get("confounded", False):
            overall = "AX"
        elif pieces["ctrl_shuf"]["class"] == "A+" and passes >= 3:
            overall = "A+"
        else:
            overall = "A0"
        if overall == "A+" and valid[arm]["class"] == "V-":
            overall = "LOW_COMPETENCE_A+"
        out[arm] = {"class": overall, "contrasts": pieces}
    return out


def aggregate_perturbation_response(rows, valid):
    out = {}
    for arm in ("TEACHER", "INTERROGATOR"):
        vals = [r["anomaly_metrics"][arm]["perturbation_response"] for r in rows]
        key_gain = classify_signed([v["response_key_gain_vs_shuffled"] for v in vals])
        direction = classify_signed([v["direction_selectivity"] for v in vals])
        key_gain["qualified_class"] = qualify(key_gain["class"], arm, valid)
        direction["qualified_class"] = qualify(direction["class"], arm, valid)
        out[arm] = {
            "response_key_gain_vs_shuffled": key_gain,
            "direction_selectivity": direction,
            "mean_response_norm_native": float(np.mean([v["response_norm_native"] for v in vals])),
            "mean_response_norm_shuffled": float(np.mean([v["response_norm_shuffled"] for v in vals])),
            "mean_response_norm_random": float(np.mean([v["response_norm_random"] for v in vals])),
        }
    return out


def aggregate_teacher_persistence(rows, valid):
    task = classify_signed([r["teacher_persistence"]["persist_task"] for r in rows])
    hist = classify_signed([r["teacher_persistence"]["persist_h"] for r in rows])
    if (task["class"] == "A+" or hist["class"] == "A+") and task["class"] != "A-" and hist["class"] != "A-":
        overall = "A+"
    elif task["class"] == "A-" or hist["class"] == "A-":
        overall = "A-"
    else:
        overall = "A0"
    overall = qualify(overall, "TEACHER", valid)
    return {"class": overall, "persist_task": task, "persist_h": hist}


def aggregate_path_and_similarity(rows):
    path = {}
    similarity = {}
    for arm, md in rows[0]["anomaly_metrics"].items():
        pvals = [r["anomaly_metrics"][arm]["path_dependence"] for r in rows]
        path[arm] = {
            "mean_matched_groups": float(np.mean([v["matched_groups"] for v in pvals])),
            "mean_geometry": float(np.mean([v["geometry"] for v in pvals])),
            "functional_gap": classify_signed([v["functional_gap"] for v in pvals]),
        }
        svals = [r["anomaly_metrics"][arm]["pair_similarity"] for r in rows]
        if svals[0] is None:
            similarity[arm] = None
        else:
            similarity[arm] = {
                "stream_a": svals[0]["stream_a"],
                "stream_b": svals[0]["stream_b"],
                "mean_linear_cka": float(np.mean([v["linear_cka"] for v in svals])),
            }
    return path, similarity


def aggregate(rows):
    valid = validity(rows)
    audits = aggregate_audits(rows)
    history = aggregate_history_field(rows, valid, "history")
    answer_history = aggregate_history_field(rows, valid, "answer_history")
    comp = aggregate_complementarity(rows, valid, audits)
    cross = aggregate_cross(rows, valid)
    react = aggregate_reactivation(rows, valid)
    order_controls = aggregate_order_controls(rows, valid)
    controls = aggregate_controls(rows, valid, audits)
    perturb = aggregate_perturbation_response(rows, valid)
    teacher = aggregate_teacher_persistence(rows, valid)
    path, similarity = aggregate_path_and_similarity(rows)

    order_h = classify_signed([r["serial_order"]["order_h"] for r in rows])
    order_x = classify_signed([r["serial_order"]["order_x"] for r in rows])
    par_h = classify_signed([r["parallel_specialization"]["par_h"] for r in rows])
    par_r = classify_signed([r["parallel_specialization"]["branch_ablation"]["rnn_contribution_return"] for r in rows])
    par_t = classify_signed([r["parallel_specialization"]["branch_ablation"]["transformer_contribution_return"] for r in rows])
    learn_beta = classify_signed([r["learning_coupling"]["learn_beta"] for r in rows])

    arch_means = {}
    for arm in rows[0]["architecture"]:
        tests = rows[0]["architecture"][arm]["test"].keys()
        arch_means[arm] = {
            "params_mean": float(np.mean([r["architecture"][arm]["params"] for r in rows])),
            **{
                k: float(np.mean([r["architecture"][arm]["test"][k] for r in rows]))
                for k in tests
                if not k.endswith("_N")
            },
        }

    return {
        "experiment": "R9-H1",
        "title": "Hybrid Interaction Anomaly Screen",
        "seeds": list(SEEDS),
        "validity": valid,
        "answer_channel_audit": audits,
        "hidden_key_history_accessibility": history,
        "answer_history_accessibility": answer_history,
        "complementary_accessibility": comp,
        "cross_module_transformation": cross,
        "reactivation": react,
        "local_order_controls": order_controls,
        "controller_dependence": controls,
        "perturbation_response": perturb,
        "teacher_persistence": teacher,
        "serial_order": {"ORDER_H": order_h, "ORDER_X": order_x},
        "parallel_specialization": {
            "PAR_H": par_h,
            "complementarity": comp.get("PARALLEL"),
            "RNN_RETURN_CONTRIBUTION": par_r,
            "TRANSFORMER_RETURN_CONTRIBUTION": par_t,
        },
        "path_dependence": path,
        "pair_similarity": similarity,
        "learning_trajectory_coupling": {
            "class": learn_beta["class"],
            "LEARN_BETA": learn_beta,
            "status": "exploratory",
        },
        "architecture_means": arch_means,
        "claim_boundary": "R9-H1 is an anomaly screen. Results do not establish an independent trajectory-information substance, universal phase code, architecture novelty, or architecture superiority.",
    }


def markdown(result):
    lines = [
        "# R9-H1 Final Aggregate — Hybrid Interaction Anomaly Screen",
        "",
        "R9-H1 is an anomaly hunt, not a performance ranking. Accuracy is used only as the preregistered interpretation floor.",
        "",
        "## Validity",
        "",
        "| arm | validity | DATA | return-early |",
        "|---|---:|---:|---:|",
    ]
    for arm, v in result["validity"].items():
        lines.append(f"| {arm} | {v['class']} | {v['DATA_ACC']:.4f} | {v['RETURN_EARLY_ACC']:.4f} |")

    lines += ["", "## Controller / teaching classifications", ""]
    for arm, c in result["controller_dependence"].items():
        lines.append(f"- {arm}: **{c['class']}**")
    lines.append(f"- TEACHER persistence: **{result['teacher_persistence']['class']}**")

    lines += ["", "## Serial order", ""]
    for key, c in result["serial_order"].items():
        lines.append(
            f"- {key}: **{c['class']}**, mean {c['mean']:+.4f}, "
            f"95% CI [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]"
        )

    lines += ["", "## Parallel specialization", ""]
    p = result["parallel_specialization"]["PAR_H"]
    lines.append(
        f"- PAR_H: **{p['class']}**, mean {p['mean']:+.4f}, "
        f"95% CI [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]"
    )
    lines.append(
        f"- RNN branch return contribution: **{result['parallel_specialization']['RNN_RETURN_CONTRIBUTION']['class']}**"
    )
    lines.append(
        f"- Transformer branch return contribution: **{result['parallel_specialization']['TRANSFORMER_RETURN_CONTRIBUTION']['class']}**"
    )

    lines += ["", "## Perturbation response", ""]
    for arm, p in result["perturbation_response"].items():
        k = p["response_key_gain_vs_shuffled"]
        d = p["direction_selectivity"]
        lines.append(
            f"- {arm}: response-key gain **{k['qualified_class']}**, direction selectivity **{d['qualified_class']}**"
        )

    lines += ["", "## Learning-trajectory coupling", ""]
    lb = result["learning_trajectory_coupling"]["LEARN_BETA"]
    lines.append(
        f"- LEARN_BETA: **{lb['class']}** (exploratory), mean {lb['mean']:+.4f}, "
        f"95% CI [{lb['ci95'][0]:+.4f}, {lb['ci95'][1]:+.4f}]"
    )

    lines += ["", "## Claim boundary", "", result["claim_boundary"], ""]
    return "\n".join(lines)


def self_check():
    p = classify_signed([0.2] * 8)
    n = classify_signed([-0.2] * 8)
    z = classify_signed([0.2, -0.2] * 4)
    assert p["class"] == "A+"
    assert n["class"] == "A-"
    assert z["class"] == "A0"
    print("R9-H1 classifier self-check passed")


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
    rows = load_families(args.root)
    result = aggregate(rows)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "FINAL_RESULT.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    (out / "FINAL_RESULT.md").write_text(markdown(result))
    print("R9-H1 frozen aggregate classification complete")
    for arm, v in result["validity"].items():
        print(f"{arm}: {v['class']}")
    for arm, c in result["controller_dependence"].items():
        print(f"controller {arm}: {c['class']}")
    print(f"teacher persistence: {result['teacher_persistence']['class']}")
    print(f"ORDER_H: {result['serial_order']['ORDER_H']['class']}")
    print(f"ORDER_X: {result['serial_order']['ORDER_X']['class']}")


if __name__ == "__main__":
    main()
