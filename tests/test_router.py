import unittest
import torch
from router_engine import IntentRoutingNetwork, RouterEngine

class RouterTests(unittest.TestCase):
    def test_model_returns_three_logits_for_five_features(self):
        model=IntentRoutingNetwork().eval()
        with torch.inference_mode():
            result=model(torch.zeros(1,5))
        self.assertEqual(tuple(result.shape),(1,3))
        self.assertTrue(bool(torch.isfinite(result).all()))

    def test_router_respects_explicit_available_backend(self):
        engine=RouterEngine()
        available={'vllm':True,'sglang':True,'llama_cpp':True}
        decision=engine.decide(100,False,'sglang',available)
        self.assertEqual(decision.backend,'sglang')
        self.assertEqual(decision.confidence,1.0)

    def test_router_selects_only_available_backend(self):
        engine=RouterEngine()
        available={'vllm':False,'sglang':False,'llama_cpp':True}
        self.assertEqual(engine.decide(100,False,None,available).backend,'llama_cpp')

    def test_fallback_excludes_offline_backends(self):
        engine=RouterEngine()
        self.assertEqual(engine.fallback_order('vllm',{'vllm':False,'sglang':True,'llama_cpp':True}),['sglang','llama_cpp'])

    def test_fallback_uses_priority_without_duplicates(self):
        engine=RouterEngine()
        self.assertEqual(engine.fallback_order('llama_cpp',{'vllm':True,'sglang':True,'llama_cpp':True}),['llama_cpp','vllm','sglang'])

    def test_no_available_backends_yields_empty_fallback(self):
        engine=RouterEngine()
        self.assertEqual(engine.fallback_order('vllm',{'vllm':False,'sglang':False,'llama_cpp':False}),[])

if __name__=='__main__': unittest.main()
