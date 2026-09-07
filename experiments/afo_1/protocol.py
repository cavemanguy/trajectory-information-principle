"""AFO-1 preparation: explicit, replay-safe observation and intervention contracts.

No neural model, semantic decoder, or scientific result is implemented here.
A system adapter is responsible for complete mutable-state snapshots and honest
state-schema/fingerprint declarations. Never use this interface to mutate a live
external process or an uncontrolled physical system.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Protocol, TypeAlias

Json: TypeAlias = Any


def canonical(obj: Json) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest(obj: Json) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()


def finite_vector(values: tuple[float, ...]) -> None:
    if not values or not all(math.isfinite(float(x)) for x in values):
        raise ValueError("state/action vector must be nonempty and finite")


@dataclass(frozen=True)
class Action:
    kind: str  # zero, vector, or integer_xor
    values: tuple[float, ...] = ()
    epsilon: float = 0.0
    bit_index: int | None = None
    bit_width: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"zero", "vector", "integer_xor"}:
            raise ValueError("unsupported action")
        if self.kind == "zero":
            if self.values or self.epsilon != 0 or self.bit_index is not None:
                raise ValueError("zero action must have no payload")
        elif self.kind == "vector":
            finite_vector(self.values)
            if not math.isfinite(self.epsilon) or not 0 < self.epsilon <= 1:
                raise ValueError("epsilon must be finite and in (0,1]")
            if self.bit_index is not None:
                raise ValueError("vector action cannot contain a bit index")
            if math.sqrt(sum(float(x)**2 for x in self.values)) > self.epsilon * (1 + 1e-12):
                raise ValueError("vector exceeds its norm budget")
        else:
            if self.values or self.epsilon != 0 or self.bit_width not in {8, 16, 32, 64}:
                raise ValueError("invalid integer XOR specification")
            if type(self.bit_index) is not int or not 0 <= self.bit_index < self.bit_width:
                raise ValueError("bit index outside declared integer width")


def xor_integer(value: int, action: Action) -> int:
    if action.kind != "integer_xor" or type(value) is not int:
        raise TypeError("integer XOR requires an integer and integer_xor action")
    if not 0 <= value < (1 << action.bit_width):
        raise ValueError("value outside declared unsigned width")
    return value ^ (1 << action.bit_index)


class FrozenSystem(Protocol):
    @property
    def fingerprint(self) -> str: ...
    def snapshot(self) -> Json: ...  # ALL mutable continuation state
    def restore(self, snapshot: Json) -> None: ...
    def observe(self) -> tuple[float, ...]: ...
    def step(self, event: Json) -> Json: ...
    def perturb(self, action: Action) -> None: ...


@dataclass(frozen=True)
class Frame:
    run_id: str
    step_index: int
    observed_ns: int
    state: tuple[float, ...]
    state_digest: str
    input_event: Json
    output_event: Json

    def to_json(self) -> str:
        return canonical(asdict(self)).decode("utf-8")


@dataclass(frozen=True)
class ProbeResult:
    action: Action
    native: tuple[Frame, ...]
    perturbed: tuple[Frame, ...]
    response: tuple[tuple[float, ...], ...]
    source_digest_before: str
    source_digest_after: str


def _frame(system: FrozenSystem, run_id: str, index: int,
           event: Json, output: Json) -> Frame:
    state = tuple(float(x) for x in system.observe())
    finite_vector(state)
    return Frame(run_id, index, time.monotonic_ns(), state,
                 digest(system.snapshot()), copy.deepcopy(event), copy.deepcopy(output))


def capture(system: FrozenSystem, events: list[Json], run_id: str,
            start_index: int = 0) -> tuple[Frame, ...]:
    """Advance the explicitly supplied native system, without an observer or probe."""
    if start_index < 0 or not run_id:
        raise ValueError("invalid run identity")
    frames = []
    for i, event in enumerate(events, start_index):
        output = system.step(copy.deepcopy(event))
        frames.append(_frame(system, run_id, i, event, output))
    return tuple(frames)


def paired_probe(system: FrozenSystem, factory: Callable[[], FrozenSystem],
                 action: Action, events: list[Json], run_id: str,
                 start_index: int = 0) -> ProbeResult:
    """Fork twice from a checkpoint; never call step/restore/perturb on source."""
    source_state = copy.deepcopy(system.snapshot())
    before = digest(source_state)
    fingerprint = system.fingerprint
    forks = []
    for label in ("native", "perturbed"):
        fork = factory()
        if fork is system or fork.fingerprint != fingerprint:
            raise ValueError("fork must be separate and have identical frozen parameters")
        fork.restore(copy.deepcopy(source_state))
        if label == "perturbed":
            fork.perturb(action)
        frames = capture(fork, events, run_id + ":" + label, start_index)
        forks.append(frames)
    after = digest(system.snapshot())
    if before != after or fingerprint != system.fingerprint:
        raise RuntimeError("source was changed during probing")
    if len(forks[0]) != len(forks[1]):
        raise RuntimeError("unequal continuation lengths")
    responses = []
    for a, b in zip(*forks):
        if len(a.state) != len(b.state):
            raise ValueError("incompatible observed state dimensions")
        responses.append(tuple(y - x for x, y in zip(a.state, b.state)))
    return ProbeResult(action, forks[0], forks[1], tuple(responses), before, after)
