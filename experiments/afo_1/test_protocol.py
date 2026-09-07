"""Structural-only tests; no training, semantic labels, or scientific metrics."""
import copy
import unittest
from dataclasses import asdict

from protocol import Action, capture, digest, paired_probe, xor_integer


class Toy:
    fingerprint = "afo-structural-test-v1"
    def __init__(self): self.h = [0.0, 0.0]
    def snapshot(self): return copy.deepcopy(self.h)
    def restore(self, state): self.h = copy.deepcopy(state)
    def observe(self): return tuple(self.h)
    def step(self, event):
        self.h = [0.8 * self.h[0] + event["x"], self.h[0] - self.h[1]]
        return {"value": self.h[0]}
    def perturb(self, action):
        if action.kind == "vector":
            self.h = [x + y for x, y in zip(self.h, action.values)]
        elif action.kind != "zero": raise ValueError("toy supports vector only")


class ContractTests(unittest.TestCase):
    def test_source_unchanged_and_replay(self):
        source = Toy()
        capture(source, [{"x": 1.0}], "warmup")
        before = digest(source.snapshot())
        events = [{"x": 2.0}, {"x": -1.0}]
        result = paired_probe(source, Toy, Action("vector", (0.1, 0.0), 0.1), events, "p")
        self.assertEqual(before, digest(source.snapshot()))
        self.assertEqual(result.source_digest_before, result.source_digest_after)
        self.assertEqual(result.native[0].step_index, 0)
        self.assertEqual(result.native[0].state, (2.8, 1.0))
        self.assertAlmostEqual(result.response[0][0], 0.08)
        self.assertEqual(asdict(result.action)["kind"], "vector")
        self.assertEqual(len(result.native), 2)
        self.assertEqual(result.native[0].to_json().find('ground_truth'), -1)
        repeat = paired_probe(source, Toy, result.action, events, "q")
        self.assertEqual(result.response, repeat.response)
        self.assertEqual([f.state for f in result.native], [f.state for f in repeat.native])

    def test_zero_is_identity(self):
        r = paired_probe(Toy(), Toy, Action("zero"), [{"x": 1.0}], "zero")
        self.assertEqual(r.response, ((0.0, 0.0),))

    def test_invalid_actions(self):
        for action in [lambda: Action("vector", (float("nan"),), 0.1),
                       lambda: Action("vector", (1.0,), 0.1),
                       lambda: Action("integer_xor", bit_index=8, bit_width=8),
                       lambda: Action("integer_xor", bit_index=-1, bit_width=8)]:
            with self.assertRaises(ValueError): action()
        a = Action("integer_xor", bit_index=3, bit_width=8)
        self.assertEqual(xor_integer(0x48, a), 0x40)
        with self.assertRaises(ValueError): xor_integer(256, a)

    def test_fork_must_be_independent(self):
        s = Toy()
        with self.assertRaises(ValueError): paired_probe(s, lambda: s, Action("zero"), [], "bad")
        class Wrong(Toy): fingerprint = "other"
        with self.assertRaises(ValueError): paired_probe(s, Wrong, Action("zero"), [], "bad")


if __name__ == "__main__": unittest.main()
