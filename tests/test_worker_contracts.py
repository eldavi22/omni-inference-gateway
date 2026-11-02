"""Exercise the inference adapters without downloading models or requiring a GPU."""

import asyncio
import sys
import types
import unittest
from unittest.mock import patch

from backends.llama_worker import LlamaCppWorker
from backends.sglang_worker import SGLangWorker
from backends.vllm_worker import VLLMWorker


class LlamaWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_model_path_is_reported(self):
        worker = LlamaCppWorker(None)
        await worker.start()
        self.assertFalse(worker.available)
        self.assertIn("not configured", worker.startup_error)
        with self.assertRaisesRegex(RuntimeError, "not configured"):
            await worker.generate("prompt", 10, 0.1)

    async def test_cpu_completion_returns_text(self):
        worker = LlamaCppWorker("model.gguf")
        class FakeModel:
            def create_completion(self, **kwargs):
                self.last_request = kwargs
                return {"choices": [{"text": "cpu response"}]}
        worker.model = FakeModel()
        worker.available = True
        result = await worker.generate("hello", 12, 0.2)
        self.assertEqual(result, "cpu response")
        self.assertEqual(worker.model.last_request["max_tokens"], 12)


class SGLangWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_endpoint_is_reported(self):
        worker = SGLangWorker(None)
        await worker.start()
        self.assertFalse(worker.available)
        with self.assertRaises(RuntimeError):
            await worker.generate("prompt", 10, 0.1)

    async def test_json_response_is_validated(self):
        import backends.sglang_worker as module
        worker = SGLangWorker("http://localhost:30000")
        worker.available = True
        worker.runtime = object()

        class FakeProgram:
            def __init__(self, answer):
                self.answer = answer
            def run(self, **kwargs):
                return {"answer": self.answer}

        with patch.object(module, "sgl", object()), patch.object(module, "structured_program", FakeProgram('{"ok": true}'), create=True):
            text = await worker.generate("prompt", 10, 0.1, {"type": "object"})
            self.assertEqual(text, '{"ok": true}')

        with patch.object(module, "sgl", object()), patch.object(module, "structured_program", FakeProgram("not json"), create=True):
            with self.assertRaises(ValueError):
                await worker.generate("prompt", 10, 0.1, {"type": "object"})

    async def test_structured_response_rejects_scalar(self):
        import backends.sglang_worker as module
        worker = SGLangWorker("http://localhost:30000")
        worker.available = True
        worker.runtime = object()
        fake = types.SimpleNamespace(run=lambda **kwargs: {"answer": "42"})
        with patch.object(module, "sgl", object()), patch.object(module, "structured_program", fake, create=True):
            with self.assertRaisesRegex(ValueError, "object or array"):
                await worker.generate("prompt", 10, 0.1, {"type": "object"})


class VLLMWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_unavailable_worker_fails_cleanly(self):
        worker = VLLMWorker(None)
        await worker.start()
        self.assertFalse(worker.available)
        with self.assertRaises(RuntimeError):
            await worker.generate("prompt", 10, 0.1)

    async def test_streamed_completion_returns_final_text(self):
        class SamplingParams:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
        class FakeEngine:
            async def generate(self, prompt, sampling, request_id):
                self.last_request = (prompt, sampling.kwargs, request_id)
                yield types.SimpleNamespace(outputs=[types.SimpleNamespace(text="partial")])
                yield types.SimpleNamespace(outputs=[types.SimpleNamespace(text="complete")])

        fake_vllm = types.SimpleNamespace(SamplingParams=SamplingParams)
        worker = VLLMWorker("example")
        worker.engine = FakeEngine()
        worker.available = True
        with patch.dict(sys.modules, {"vllm": fake_vllm}):
            result = await worker.generate("hello", 16, 0.3)
        self.assertEqual(result, "complete")
        self.assertEqual(worker.engine.last_request[1]["max_tokens"], 16)

    async def test_empty_engine_output_raises(self):
        class SamplingParams:
            def __init__(self, **kwargs):
                pass
        class FakeEngine:
            async def generate(self, prompt, sampling, request_id):
                if False:
                    yield None

        worker = VLLMWorker("example")
        worker.engine = FakeEngine()
        worker.available = True
        with patch.dict(sys.modules, {"vllm": types.SimpleNamespace(SamplingParams=SamplingParams)}):
            with self.assertRaisesRegex(RuntimeError, "no completion"):
                await worker.generate("hello", 16, 0.3)


if __name__ == "__main__":
    unittest.main()
