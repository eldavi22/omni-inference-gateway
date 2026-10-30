from __future__ import annotations

import asyncio
from typing import Any


class LlamaCppWorker:
    """GGUF/CPU fallback backed by llama-cpp-python."""

    def __init__(self, model_path: str | None) -> None:
        self.model_path = model_path
        self.model: Any | None = None
        self.available = False
        self.startup_error: str | None = None

    async def start(self) -> None:
        if not self.model_path:
            self.startup_error = "LLAMA_GGUF_MODEL is not configured."
            return

        from llama_cpp import Llama

        # n_gpu_layers=0 preserves the guaranteed low-resource CPU fallback.
        self.model = await asyncio.to_thread(
            Llama,
            model_path=self.model_path,
            n_ctx=4096,
            n_gpu_layers=0,
            verbose=False,
        )
        self.available = True

    def _generate_sync(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        if self.model is None:
            raise RuntimeError(self.startup_error or "llama.cpp worker is unavailable.")
        result = self.model.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return str(result["choices"][0]["text"])

    async def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        del json_schema
        if not self.available:
            raise RuntimeError(self.startup_error or "llama.cpp worker is unavailable.")
        return await asyncio.to_thread(
            self._generate_sync, prompt, max_tokens, temperature
        )
