"""Frozen AFO-1A source qualification. Scientific modes only after smoke."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiments.afo_1.protocol import Action, canonical, digest, paired_probe
from experiments.afo_1a.source import (
    PROTOCOL, SEEDS, START, MODE, DATA, PAD, NVAL, SEQ_LEN, HIDDEN,
    STEPS, BATCH, EVAL_N, OP_NAMES, Source, SourceAdapter, apply_op,
    derive_seed, set_seed, make_batch, model_inputs, parameter_fingerprint,
    save_checkpoint, load_checkpoint,
)

PREREG_COMMIT = '9a3f8223e6566309b43f6ea9a4267712217bacbb'
PREREG_BLOB = '9d7ba7ad324a2bbaa6d932ef90ecad4c80c53d19'
SOURCE_FILES = ('experiments/afo_1/protocol.py', 'experiments/afo_1a/source.py',
                'experiments/afo_1a/run_afo_1a.py', 'experiments/afo_1a/classify_afo_1a.py',
                'experiments/afo_1a/test_afo_1a.py', 'experiments/afo_1a/requirements.txt',
                'experiments/afo_1a/IMPLEMENTATION_NOTES.md', 'experiments/afo_1a/AMENDMENT_0.md',
                'experiments/afo_1a/PREREGISTRATION.md', 'experiments/afo_1a/README.md',
                '.github/workflows/afo-1a.yml')


def json_write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(obj) + b'\n')


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_manifest(root=Path('.')):
    return {p: file_sha(root / p) for p in SOURCE_FILES}


def environment():
    return {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
            'numpy': np.__version__, 'cpu_threads': torch.get_num_threads(),
            'deterministic': torch.are_deterministic_algorithms_enabled()}


def accuracy_record(pred, target, mask):
    count = int(mask.sum())
    if count == 0:
        raise ValueError('empty evaluation subset')
    correct = int(((pred == target) & mask).sum())
    return {'correct': correct, 'count': count, 'accuracy': correct / count}


def evaluate(model, seed, out):
    """Preserve the complete, legally observed held-out native trace."""
    model.eval()
    batch = make_batch(derive_seed(seed, 'heldout-evaluation'), EVAL_N)
    types, payload = model_inputs(batch)
    logits_all = np.empty((EVAL_N, SEQ_LEN, NVAL), np.float32)
    states_all = np.empty((EVAL_N, SEQ_LEN, HIDDEN), np.float32)
    times_all = np.empty((EVAL_N, SEQ_LEN), np.int64)
    with torch.no_grad():
        for n in range(EVAL_N):
            h = torch.zeros(1, HIDDEN)
            for t in range(SEQ_LEN):
                h = model.transition(h, types[n:n+1, t], payload[n:n+1, t])
                logits = model.head(h)
                if not torch.isfinite(h).all() or not torch.isfinite(logits).all():
                    raise FloatingPointError('nonfinite held-out trace')
                states_all[n, t] = h[0].numpy()
                logits_all[n, t] = logits[0].numpy()
                times_all[n, t] = time.monotonic_ns()
    pred = logits_all.argmax(-1)
    mask = batch['types'] == DATA
    metrics = {'data': accuracy_record(pred, batch['target'], mask),
               'operations': {}, 'final_block': accuracy_record(
                   pred, batch['target'], mask & (batch['block'] == 2))}
    for k, name in enumerate(OP_NAMES):
        metrics['operations'][name] = accuracy_record(pred, batch['target'], mask & (batch['op'] == k))
    metrics['qualified'] = bool(metrics['data']['accuracy'] >= .65 and
        metrics['final_block']['accuracy'] >= .60 and
        all(v['accuracy'] >= .50 for v in metrics['operations'].values()))
    metrics['chance'] = .125
    metrics['operation_chance'] = .25
    metrics['examples'] = EVAL_N
    metrics['data_positions_per_episode'] = 24
    metrics['native_state_norm_mean'] = float(np.linalg.norm(states_all[:, :28], axis=-1).mean())
    metrics['native_displacement_norm_mean'] = float(np.linalg.norm(
        np.diff(states_all[:, :28], axis=1), axis=-1).mean())
    if not np.isfinite([metrics['native_state_norm_mean'], metrics['native_displacement_norm_mean']]).all():
        raise FloatingPointError('nonfinite native summary')
    np.savez_compressed(out / 'NATIVE_TRACE.npz', states=states_all, logits=logits_all,
                        observed_ns=times_all, step_index=np.broadcast_to(
                            np.arange(SEQ_LEN, dtype=np.int64), (EVAL_N, SEQ_LEN)).copy(),
                        types=batch['types'], payload=batch['payload'])
    np.savez_compressed(out / 'REFERENCE_TRACE.npz',
                        **{k: batch[k] for k in ('target', 'op', 'before', 'after', 'block')})
    return metrics, batch, states_all, logits_all


def replay_audit(model, seed, batch, states, logits):
    """Eight preselected held-out episodes; no label-conditioned selection."""
    checks = []
    fp = parameter_fingerprint(model)
    for n in range(8):
        events = [{'type': int(t), 'payload': int(p)} for t, p in
                  zip(batch['types'][n, :28], batch['payload'][n, :28])]
        source = SourceAdapter(model)
        native_states, native_logits = [], []
        for event in events:
            output = source.step(event)
            native_states.append(source.h.reshape(-1).numpy().copy())
            native_logits.append(np.asarray(output['logits'], dtype=np.float32))
        if not np.array_equal(np.asarray(native_states), states[n, :28]) or not np.array_equal(np.asarray(native_logits), logits[n, :28]):
            raise ValueError('canonical native replay mismatch')
        checkpoint = SourceAdapter(model)
        for event in events[:5]:
            checkpoint.step(event)
        before = digest(checkpoint.snapshot())
        continuation = events[5:9]
        factory = checkpoint.fork
        zero = paired_probe(checkpoint, factory, Action('zero'), continuation,
                            f'{seed}:{n}:zero', start_index=5)
        for a, b in zip(zero.native, zero.perturbed):
            if a.state != b.state or a.output_event != b.output_event or a.state_digest != b.state_digest:
                raise ValueError('zero-action replay identity failed')
        if not all(all(v == 0.0 for v in r) for r in zero.response):
            raise ValueError('zero-action response is not zero')
        action = Action('vector', values=(.01,) + (0.,)*(HIDDEN-1), epsilon=.01)
        perturbed = paired_probe(checkpoint, factory, action, continuation,
                                 f'{seed}:{n}:vector', start_index=5)
        if digest(checkpoint.snapshot()) != before or checkpoint.fingerprint != fp:
            raise ValueError('isolated probe changed native source')
        restored = checkpoint.fork()
        restored.restore(checkpoint.snapshot())
        for event in continuation:
            a = checkpoint.step(event)
            b = restored.step(event)
            if a != b or checkpoint.snapshot() != restored.snapshot():
                raise ValueError('restored continuation mismatch')
        checks.append({'episode': n, 'checkpoint_index': 5, 'state_digest': before,
                       'zero_identity': True, 'nonzero_source_unchanged': True,
                       'restore_identity': True, 'parameter_fingerprint': fp,
                       'native_response_norms': [float(np.linalg.norm(r)) for r in perturbed.response]})
    return {'passed': True, 'checks': checks}


def run_family(seed, out, source_commit, expected_manifest=None):
    if seed not in SEEDS or not source_commit or len(source_commit) != 40:
        raise ValueError('invalid frozen family identity')
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    set_seed(derive_seed(seed, 'initialization'))
    manifest = source_manifest()
    if expected_manifest is None or manifest != expected_manifest:
        raise ValueError('source differs from frozen smoke manifest')
    json_write(out / 'RUN_MANIFEST.json', {'protocol': PROTOCOL, 'seed': seed,
        'source_commit': source_commit, 'preregistration_commit': PREREG_COMMIT,
        'preregistration_blob': PREREG_BLOB, 'source_files': manifest,
        'environment': environment(), 'namespaces': {
            k: derive_seed(seed, k) for k in ('initialization', 'heldout-evaluation',
                                              'replay', 'observer-train', 'observer-val', 'observer-test')},
        'training_steps': STEPS, 'batch': BATCH, 'evaluation_episodes': EVAL_N})
    status = {'protocol': PROTOCOL, 'seed': seed, 'source_commit': source_commit,
              'status': 'started', 'completed_steps': 0}
    json_write(out / 'STATUS.json', status)
    model = Source()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    history = []
    try:
        model.train()
        for step in range(1, STEPS+1):
            batch = make_batch(derive_seed(seed, f'train:{step}'), BATCH)
            types, payload = model_inputs(batch)
            target = torch.as_tensor(batch['target'])
            optimizer.zero_grad(set_to_none=True)
            logits = model(types, payload)
            loss = F.cross_entropy(logits.reshape(-1, NVAL), target.reshape(-1), ignore_index=-100)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'nonfinite loss at step {step}')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            status['completed_steps'] = step
            if step % 100 == 0:
                history.append({'step': step, 'loss': float(loss.detach()),
                                'gradient_norm_preclip': float(grad)})
                json_write(out / 'TRAINING_HISTORY.json', history)
                json_write(out / 'STATUS.json', status)
        ckpt = out / 'SOURCE_FINAL.pt'
        checkpoint_sha = save_checkpoint(model, ckpt)
        frozen = load_checkpoint(ckpt)
        if parameter_fingerprint(model) != parameter_fingerprint(frozen):
            raise ValueError('final checkpoint fingerprint mismatch')
        metrics, batch, states, logits = evaluate(frozen, seed, out)
        replay = replay_audit(frozen, seed, batch, states, logits)
        json_write(out / 'REPLAY_AUDIT.json', replay)
        artifact_hashes = {name: file_sha(out / name) for name in
            ('SOURCE_FINAL.pt', 'NATIVE_TRACE.npz', 'REFERENCE_TRACE.npz',
             'TRAINING_HISTORY.json', 'REPLAY_AUDIT.json', 'RUN_MANIFEST.json')}
        record = {'protocol': PROTOCOL, 'seed': seed, 'source_commit': source_commit,
                  'preregistration_commit': PREREG_COMMIT, 'preregistration_blob': PREREG_BLOB,
                  'source_files': manifest, 'all_valid': True, 'training_steps': STEPS,
                  'checkpoint_sha256': checkpoint_sha, 'parameter_fingerprint': parameter_fingerprint(frozen),
                  'artifact_sha256': artifact_hashes,
                  'environment': environment(), 'metrics': metrics, 'replay_passed': replay['passed'],
                  'training_history': history}
        json_write(out / 'FAMILY_RESULT.json', record)
        status.update(status='complete', qualified=metrics['qualified'],
                      checkpoint_sha256=checkpoint_sha, replay_passed=True)
        json_write(out / 'STATUS.json', status)
        return record
    except BaseException as exc:
        status.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        try:
            failure_path = out / 'FAILURE_STATE.pt'
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'completed_steps': status['completed_steps']}, failure_path)
            status['failure_state_sha256'] = file_sha(failure_path)
        except Exception as preservation_error:
            status['preservation_error'] = repr(preservation_error)
        json_write(out / 'STATUS.json', status)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--manifest', required=True)
    a = parser.parse_args()
    expected = json.loads(Path(a.manifest).read_text())
    run_family(a.seed, a.out, a.source_commit, expected)


if __name__ == '__main__':
    main()
