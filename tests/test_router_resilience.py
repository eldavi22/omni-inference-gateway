"""Check routing choices, availability changes, and predictable fallback behavior."""

import itertools
import math
import unittest

import torch

from router_engine import IntentRoutingNetwork, RouterEngine


class RouterResilienceTests(unittest.TestCase):
    def setUp(self):
        self.router = RouterEngine()

    def test_all_backend_availability_combinations(self):
        names = self.router.labels
        for flags in itertools.product((False, True), repeat=len(names)):
            available = dict(zip(names, flags))
            with self.subTest(available=available):
                decision = self.router.decide(512, False, None, available)
                if any(flags):
                    self.assertTrue(available[decision.backend])
                else:
                    self.assertEqual(self.router.fallback_order(decision.backend, available), [])
                self.assertTrue(math.isfinite(decision.confidence))
                self.assertGreaterEqual(decision.confidence, 0.0)
                self.assertLessEqual(decision.confidence, 1.0)

    def test_preferred_backend_is_honored_when_online(self):
        available = dict.fromkeys(self.router.labels, True)
        for backend in self.router.labels:
            with self.subTest(backend=backend):
                decision = self.router.decide(300, True, backend, available)
                self.assertEqual(decision.backend, backend)
                self.assertEqual(decision.confidence, 1.0)

    def test_offline_preference_never_selects_unavailable_worker(self):
        available = {"vllm": False, "sglang": True, "llama_cpp": False}
        for preference in self.router.labels:
            with self.subTest(preference=preference):
                decision = self.router.decide(300, False, preference, available)
                self.assertEqual(decision.backend, "sglang")

    def test_structured_request_prefers_sglang_when_all_online(self):
        available = dict.fromkeys(self.router.labels, True)
        decision = self.router.decide(100, True, None, available)
        self.assertEqual(decision.backend, "sglang")

    def test_long_unstructured_request_prefers_vllm(self):
        available = dict.fromkeys(self.router.labels, True)
        decision = self.router.decide(2048, False, None, available)
        self.assertEqual(decision.backend, "vllm")

    def test_cpu_only_uses_llama_cpp_for_different_requests(self):
        available = {"vllm": False, "sglang": False, "llama_cpp": True}
        for tokens, structured in ((0, False), (256, True), (4096, False)):
            with self.subTest(tokens=tokens, structured=structured):
                self.assertEqual(self.router.decide(tokens, structured, None, available).backend, "llama_cpp")

    def test_fallback_never_duplicates_or_includes_offline_backends(self):
        for preferred in self.router.labels:
            for flags in itertools.product((False, True), repeat=3):
                available = dict(zip(self.router.labels, flags))
                order = self.router.fallback_order(preferred, available)
                with self.subTest(preferred=preferred, available=available):
                    self.assertEqual(len(order), len(set(order)))
                    self.assertEqual(set(order), {name for name, online in available.items() if online})
                    if available[preferred]:
                        self.assertEqual(order[0], preferred)

    def test_large_token_counts_produce_finite_confidence(self):
        available = dict.fromkeys(self.router.labels, True)
        for count in (0, 1, 2048, 100_000, 10_000_000):
            with self.subTest(count=count):
                decision = self.router.decide(count, False, None, available)
                self.assertTrue(math.isfinite(decision.confidence))

    def test_routing_is_deterministic_for_the_same_input(self):
        available = dict.fromkeys(self.router.labels, True)
        decisions = [self.router.decide(512, True, None, available) for _ in range(12)]
        self.assertEqual(decisions, [decisions[0]] * len(decisions))

    def test_model_outputs_finite_logits_for_batch(self):
        model = IntentRoutingNetwork().eval()
        with torch.inference_mode():
            logits = model(torch.zeros(32, 5))
        self.assertEqual(tuple(logits.shape), (32, 3))
        self.assertTrue(bool(torch.isfinite(logits).all()))

    def test_fallback_priority_is_stable(self):
        available = dict.fromkeys(self.router.labels, True)
        self.assertEqual(self.router.fallback_order("sglang", available), ["sglang", "vllm", "llama_cpp"])


if __name__ == "__main__":
    unittest.main()
