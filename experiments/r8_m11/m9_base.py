import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import m7r_base as base

FRESH_SEEDS = (631, 648, 664, 681, 699, 716, 733, 751, 768, 784, 802, 821)
CHECK_EVERY = 10
FIRST_RECORDED_CHECK = 40
MAX_BASELINE_EPOCH = 400
PAIR_N = 2048
A_HISTORY = ((0.00, 60), (0.25, 30), (0.50, 30))
B_HISTORY = ((1.00, 60), (0.75, 30), (0.50, 30))
MIDPOINT_POST_EPOCH = 120
HOLD_EPOCHS = 120
HOLD_CHECKS = (30, 60, 90, 120)
ENC_PREFIXES = ("rel_emb.", "val_emb.", "enc.", "to_h.")
REC_PREFIXES = ("F.",)
READER_PREFIXES = ("head0.", "headT.")


def derive_seed(seed, name):
    h = hashlib.sha256(f"R8-M9|{seed}|{name}".encode()).digest()
    return int.from_bytes(h[:4], "big")


base.derive_seed = derive_seed


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def clone_state(sd):
    return {k: v.detach().cpu().clone() for k, v in sd.items()}


def key_block(k):
    if k.startswith(ENC_PREFIXES):
        return "E"
    if k.startswith(REC_PREFIXES):
        return "F"
    if k.startswith(READER_PREFIXES):
        return "R"
    raise RuntimeError(f"unassigned parameter key: {k}")


def subset_state(sd, block):
    return {k: v for k, v in sd.items() if key_block(k) == block}


def sha_subset(sd, block):
    return base.sha_state_dict(subset_state(sd, block))


def compose_state(enc_state, rec_state, reader_state):
    keys = set(enc_state) | set(rec_state) | set(reader_state)
    out = {}
    for k in sorted(keys):
        b = key_block(k)
        src = enc_state if b == "E" else rec_state if b == "F" else reader_state
        if k not in src:
            raise RuntimeError(f"missing {b} key {k}")
        out[k] = src[k].detach().cpu().clone()
    return out


def _hash_obj(h, obj):
    if torch.is_tensor(obj):
        t = obj.detach().cpu().contiguous()
        h.update(b"T")
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    elif isinstance(obj, dict):
        h.update(b"D")
        for k in sorted(obj.keys(), key=lambda x: str(x)):
            h.update(str(k).encode())
            _hash_obj(h, obj[k])
    elif isinstance(obj, (list, tuple)):
        h.update(b"L")
        for x in obj:
            _hash_obj(h, x)
    else:
        h.update(b"S")
        h.update(repr(obj).encode())


def sha_optimizer_state(state):
    h = hashlib.sha256()
    _hash_obj(h, state)
    return h.hexdigest()


def task_loss_lambda(model, y, perms, A, B, lam):
    ce = nn.CrossEntropyLoss()
    _, l0, lT = model(y, perms)
    z0 = torch.stack([ce(l0[r], y[:, r]) for r in range(base.N_REL)])
    zT = torch.stack([ce(lT[r], y[:, r]) for r in range(base.N_REL)])
    w = torch.ones(base.N_REL, dtype=zT.dtype, device=zT.device)
    w[int(A)] = 1.0 + 3.0 * (1.0 - float(lam))
    w[int(B)] = 1.0 + 3.0 * float(lam)
    return 0.5 * (z0.mean() + (zT * w).sum() / w.sum())


def train_one_epoch_lambda(model, opt, seed, train_y, ep, A, B, lam):
    model.train()
    train_perm = base.make_perms(len(train_y), derive_seed(seed, f"presentation_{ep}"))
    g = torch.Generator().manual_seed(derive_seed(seed, f"order_{ep}"))
    order = torch.randperm(len(train_y), generator=g)
    for a in range(0, len(train_y), base.BATCH):
        ix = order[a:a + base.BATCH]
        opt.zero_grad(set_to_none=True)
        loss = task_loss_lambda(model, train_y[ix], train_perm[ix], A, B, lam)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss seed={seed} epoch={ep} lambda={lam}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), base.CLIP)
        opt.step()


def record(model, val_y, val_perm, bank, epoch, A, B, lam=None, post=None):
    r = base.checkpoint_record(model, val_y, val_perm, bank, epoch, A, B)
    if lam is not None:
        r["lambda"] = float(lam)
    if post is not None:
        r["post_maturity_epoch"] = int(post)
    return r


def run_history(seed, name, schedule, start_state, start_opt, start_sha,
                M, A, B, train_y, val_y, val_perm, bank, expected_post=120):
    model = base.Core()
    model.load_state_dict(clone_state(start_state))
    opt = torch.optim.AdamW(model.parameters(), lr=base.LR, weight_decay=base.WD)
    opt.load_state_dict(copy.deepcopy(start_opt))
    if base.sha_state_dict(model.state_dict()) != start_sha:
        raise RuntimeError(f"history fork mismatch: {name}")
    if sha_optimizer_state(opt.state_dict()) != sha_optimizer_state(start_opt):
        raise RuntimeError(f"history optimizer fork mismatch: {name}")

    records = []
    post = 0
    for lam, duration in schedule:
        for _ in range(int(duration)):
            post += 1
            train_one_epoch_lambda(model, opt, seed, train_y, M + post, A, B, lam)
        r = record(model, val_y, val_perm, bank, M + post, A, B, lam, post)
        records.append(r)
        print(f"seed={seed} {name} post={post} lambda={lam:.2f} Q={r['Q']:.4f}", flush=True)

    if expected_post is not None and post != int(expected_post):
        raise RuntimeError(f"history length mismatch {name}: {post}")
    return {
        "records": records,
        "model_state": clone_state(model.state_dict()),
        "optimizer_state": copy.deepcopy(opt.state_dict()),
        "state_sha256": base.sha_state_dict(model.state_dict()),
        "optimizer_sha256": sha_optimizer_state(opt.state_dict()),
        "fork_identity": True,
    }


def build_transplant(enc_state, rec_state, reader_state, expected_hashes):
    sd = compose_state(enc_state, rec_state, reader_state)
    model = base.Core()
    model.load_state_dict(sd)
    got = {b: sha_subset(model.state_dict(), b) for b in ("E", "F", "R")}
    ok = all(got[b] == expected_hashes[b] for b in ("E", "F", "R"))
    if not ok:
        raise RuntimeError(f"transplant block-hash mismatch expected={expected_hashes} got={got}")
    return model, got


def midpoint_latent_distance(state_a, state_b, y, perms):
    ma = base.Core(); ma.load_state_dict(clone_state(state_a)); ma.eval()
    mb = base.Core(); mb.load_state_dict(clone_state(state_b)); mb.eval()
    h0v, h12v = [], []
    with torch.no_grad():
        for a in range(0, len(y), base.BATCH):
            yy = y[a:a + base.BATCH]
            pp = perms[a:a + base.BATCH]
            h0a = ma.encode(yy, pp); h0b = mb.encode(yy, pp)
            ta = ma.trajectory(h0a); tb = mb.trajectory(h0b)
            h0v.append(torch.linalg.vector_norm(h0a - h0b, dim=1).cpu())
            h12v.append(torch.linalg.vector_norm(ta[:, -1] - tb[:, -1], dim=1).cpu())
    return {
        "mean_h0_distance": float(torch.cat(h0v).mean()),
        "mean_h12_distance": float(torch.cat(h12v).mean()),
    }


def weighted_mid_accuracy(perf, A, B):
    arr = np.asarray(perf["h12_per_relation"], dtype=np.float64)
    w = np.ones(base.N_REL, dtype=np.float64)
    w[int(A)] = 2.5
    w[int(B)] = 2.5
    return float(np.sum(arr * w) / np.sum(w))


def reader_transfer_matrix(states, val_y, val_perm, A, B):
    out = {}
    combos = {
        "AA": (states["A"], states["A"]),
        "AB": (states["A"], states["B"]),
        "BA": (states["B"], states["A"]),
        "BB": (states["B"], states["B"]),
    }
    for dyn_name, (enc_src, rec_src) in combos.items():
        for reader_name, reader_src in (("A", states["A"]), ("B", states["B"])):
            sd = compose_state(enc_src, rec_src, reader_src)
            m = base.Core(); m.load_state_dict(sd)
            perf = base.eval_model(m, val_y, val_perm)
            out[f"{dyn_name}_R{reader_name}"] = {
                "h0_overall": perf["h0_overall"],
                "h12_overall": perf["h12_overall"],
                "h12_A": float(perf["h12_per_relation"][int(A)]),
                "h12_B": float(perf["h12_per_relation"][int(B)]),
                "h12_mid_weighted": weighted_mid_accuracy(perf, A, B),
                "h12_per_relation": perf["h12_per_relation"],
            }
    return out


def run_hold(seed, label, model_state, opt_state, mode, M, A, B,
             train_y, val_y, val_perm, bank):
    model = base.Core(); model.load_state_dict(clone_state(model_state))
    start_sha = base.sha_state_dict(model.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=base.LR, weight_decay=base.WD)
    if mode != "RESET":
        opt.load_state_dict(copy.deepcopy(opt_state))
        if sha_optimizer_state(opt.state_dict()) != sha_optimizer_state(opt_state):
            raise RuntimeError(f"optimizer load mismatch {label}")
    start_opt_sha = sha_optimizer_state(opt.state_dict())

    records = []
    for hp in range(1, HOLD_EPOCHS + 1):
        abs_post = MIDPOINT_POST_EPOCH + hp
        train_one_epoch_lambda(model, opt, seed, train_y, M + abs_post, A, B, 0.5)
        if hp in HOLD_CHECKS:
            r = record(model, val_y, val_perm, bank, M + abs_post, A, B, 0.5, abs_post)
            r["hold_epoch"] = int(hp)
            records.append(r)
            print(f"seed={seed} hold={label} ep={hp} Q={r['Q']:.4f}", flush=True)

    return {
        "records": records,
        "fork_state_sha256": start_sha,
        "start_optimizer_sha256": start_opt_sha,
        "final_state_sha256": base.sha_state_dict(model.state_dict()),
        "finite": all(base.finite_record(x) for x in records),
    }


def smoke_run(seed, outdir):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    train_y = base.make_memories(512, derive_seed(seed, "train"))
    val_y = base.make_memories(256, derive_seed(seed, "val"))
    val_perm = base.make_perms(256, derive_seed(seed, "val_perm"))
    bank = base.make_pair_bank(seed, 96)

    base.set_seed(derive_seed(seed, "init"))
    model = base.Core(); opt = torch.optim.AdamW(model.parameters(), lr=base.LR, weight_decay=base.WD)
    for ep in range(1, 3):
        base.train_one_epoch(model, opt, seed, train_y, ep)
    surv = base.survival_summary(model, bank)
    A = int(surv["winner_relation"]); B = int(surv["loser_relation"]); M = 2
    start = clone_state(model.state_dict()); opt0 = copy.deepcopy(opt.state_dict()); sha0 = base.sha_state_dict(start)

    ha = run_history(seed, "A_SMOKE", ((0.0, 1), (0.25, 1), (0.5, 1)), start, opt0, sha0,
                     M, A, B, train_y, val_y, val_perm, bank, expected_post=3)
    hb = run_history(seed, "B_SMOKE", ((1.0, 1), (0.75, 1), (0.5, 1)), start, opt0, sha0,
                     M, A, B, train_y, val_y, val_perm, bank, expected_post=3)
    states = {"A": ha["model_state"], "B": hb["model_state"]}
    hashes = {s: {b: sha_subset(states[s], b) for b in ("E", "F", "R")} for s in ("A", "B")}
    q = {}
    for name, es, fs in (("AA", "A", "A"), ("AB", "A", "B"), ("BA", "B", "A"), ("BB", "B", "B")):
        m, _ = build_transplant(states[es], states[fs], states["A"], {"E": hashes[es]["E"], "F": hashes[fs]["F"], "R": hashes["A"]["R"]})
        q[name] = base.checkpoint_record(m, val_y, val_perm, bank, M + 3, A, B)["Q"]
    rmat = reader_transfer_matrix(states, val_y, val_perm, A, B)
    save_json(out / "smoke_summary.json", {
        "status": "ok", "seed": int(seed), "A": A, "B": B, "Q": q,
        "reader_keys": sorted(rmat), "partition_hashes": hashes,
    })


def run(seed, outdir):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    train_y = base.make_memories(base.TRAIN_N, derive_seed(seed, "train"))
    val_y = base.make_memories(base.VAL_N, derive_seed(seed, "val"))
    test_y = base.make_memories(base.TEST_N, derive_seed(seed, "test"))
    val_perm = base.make_perms(base.VAL_N, derive_seed(seed, "val_perm"))
    test_perm = base.make_perms(base.TEST_N, derive_seed(seed, "test_perm"))
    bank = base.make_pair_bank(seed, PAIR_N)

    base.set_seed(derive_seed(seed, "init"))
    model = base.Core(); opt = torch.optim.AdamW(model.parameters(), lr=base.LR, weight_decay=base.WD)
    history = []
    M = None
    for ep in range(1, MAX_BASELINE_EPOCH + 1):
        base.train_one_epoch(model, opt, seed, train_y, ep)
        if ep >= FIRST_RECORDED_CHECK and ep % CHECK_EVERY == 0:
            r = base.checkpoint_record(model, val_y, val_perm, bank, ep)
            history.append(r)
            print(f"seed={seed} baseline ep={ep} combined={r['validation']['combined']:.4f} h0={r['validation']['h0_overall']:.4f} winner={r['survival']['winner_relation']} loser={r['survival']['loser_relation']}", flush=True)
            M = base.maturity_from_history(history)
            if M is not None:
                break

    if M is None:
        save_json(out / "seed_summary.json", {
            "experiment": "R8-M9", "seed": int(seed), "maturity_reached": False,
            "maturity_epoch": None, "baseline_history": history,
            "validity": {"maturity": False, "fork_identity": False, "transplants": False, "holds": False, "finite": True, "complete": False},
            "all_valid": False,
            "environment": {"python": __import__("sys").version, "torch": torch.__version__, "numpy": np.__version__},
        })
        return

    A = int(history[-1]["survival"]["winner_relation"])
    B = int(history[-1]["survival"]["loser_relation"])
    if A == B:
        raise RuntimeError("A and B identical")

    baseline = base.checkpoint_record(model, val_y, val_perm, bank, M, A, B)
    start_state = clone_state(model.state_dict())
    start_opt = copy.deepcopy(opt.state_dict())
    start_sha = base.sha_state_dict(start_state)

    ha = run_history(seed, "A_HISTORY", A_HISTORY, start_state, start_opt, start_sha,
                     M, A, B, train_y, val_y, val_perm, bank)
    hb = run_history(seed, "B_HISTORY", B_HISTORY, start_state, start_opt, start_sha,
                     M, A, B, train_y, val_y, val_perm, bank)
    states = {"A": ha["model_state"], "B": hb["model_state"]}
    opts = {"A": ha["optimizer_state"], "B": hb["optimizer_state"]}
    block_hashes = {s: {b: sha_subset(states[s], b) for b in ("E", "F", "R")} for s in ("A", "B")}

    transplants = {}
    all_transplant_ok = True
    for name, es, fs in (("AA", "A", "A"), ("AB", "A", "B"), ("BA", "B", "A"), ("BB", "B", "B")):
        expected = {"E": block_hashes[es]["E"], "F": block_hashes[fs]["F"], "R": block_hashes["A"]["R"]}
        tm, got = build_transplant(states[es], states[fs], states["A"], expected)
        rr = base.checkpoint_record(tm, val_y, val_perm, bank, M + MIDPOINT_POST_EPOCH, A, B)
        transplants[name] = {"record": rr, "block_hashes": got, "expected_hashes": expected}
        all_transplant_ok = all_transplant_ok and (got == expected)

    qa = float(transplants["AA"]["record"]["Q"])
    qab = float(transplants["AB"]["record"]["Q"])
    qba = float(transplants["BA"]["record"]["Q"])
    qb = float(transplants["BB"]["record"]["Q"])
    effects = {
        "H_parent": qb - qa,
        "E_effect": 0.5 * ((qba - qa) + (qb - qab)),
        "F_effect": 0.5 * ((qab - qa) + (qb - qba)),
        "I_EF": (qb - qab) - (qba - qa),
    }

    latent_distance = midpoint_latent_distance(states["A"], states["B"], val_y, val_perm)
    reader_val = reader_transfer_matrix(states, val_y, val_perm, A, B)

    holds = {
        "INHERITED": {
            "A": run_hold(seed, "INHERITED_A", states["A"], opts["A"], "INHERITED", M, A, B, train_y, val_y, val_perm, bank),
            "B": run_hold(seed, "INHERITED_B", states["B"], opts["B"], "INHERITED", M, A, B, train_y, val_y, val_perm, bank),
        },
        "RESET": {
            "A": run_hold(seed, "RESET_A", states["A"], None, "RESET", M, A, B, train_y, val_y, val_perm, bank),
            "B": run_hold(seed, "RESET_B", states["B"], None, "RESET", M, A, B, train_y, val_y, val_perm, bank),
        },
        "CROSSED": {
            "A": run_hold(seed, "CROSSED_A", states["A"], opts["B"], "CROSSED", M, A, B, train_y, val_y, val_perm, bank),
            "B": run_hold(seed, "CROSSED_B", states["B"], opts["A"], "CROSSED", M, A, B, train_y, val_y, val_perm, bank),
        },
    }

    hold_ok = all(v[s]["finite"] for v in holds.values() for s in ("A", "B"))
    fork_ok = bool(ha["fork_identity"] and hb["fork_identity"])
    finite_trans = all(base.finite_record(transplants[k]["record"]) for k in transplants)

    # Final test evaluation occurs only after every fixed training path is complete.
    reader_test = reader_transfer_matrix(states, test_y, test_perm, A, B)

    summary = {
        "experiment": "R8-M9",
        "seed": int(seed),
        "fresh_seed": bool(seed in FRESH_SEEDS),
        "maturity_reached": True,
        "maturity_epoch": int(M),
        "A_baseline_winner": int(A),
        "B_baseline_loser": int(B),
        "baseline": baseline,
        "baseline_state_sha256": start_sha,
        "baseline_history": history,
        "histories": {
            "A": {k: v for k, v in ha.items() if k not in ("model_state", "optimizer_state")},
            "B": {k: v for k, v in hb.items() if k not in ("model_state", "optimizer_state")},
        },
        "block_hashes": block_hashes,
        "transplants": transplants,
        "effects": effects,
        "midpoint_latent_distance": latent_distance,
        "reader_transfer_validation": reader_val,
        "reader_transfer_test": reader_test,
        "holds": holds,
        "validity": {
            "maturity": True,
            "fork_identity": fork_ok,
            "transplants": bool(all_transplant_ok),
            "holds": bool(hold_ok),
            "finite": bool(finite_trans and hold_ok),
            "complete": True,
        },
        "all_valid": bool(fork_ok and all_transplant_ok and finite_trans and hold_ok),
        "environment": {"python": __import__("sys").version, "torch": torch.__version__, "numpy": np.__version__},
    }
    save_json(out / "seed_summary.json", summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke_run(args.seed, args.outdir)
    else:
        run(args.seed, args.outdir)


if __name__ == "__main__":
    main()
