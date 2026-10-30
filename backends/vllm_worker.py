from __future__ import annotations

from typing import Any


class VLLMWorker:
    """High-throughput vLLM adapter using AsyncLLMEngine and PagedAttention."""

    def __init__(self, model_name: str | None) -> None:
        self.model_name = model_name
        self.engine: Any | None = None
        self.available = False
        self.startup_error: str | None = None

    async def start(self) -> None:
        if not self.model_name:
            self.startup_error = "VLLM_MODEL is not configured."
            return

        from vllm import AsyncEngineArgs, AsyncLLMEngine

        # vLLM's scheduler and PagedAttention maximize batched throughput.
        args = AsyncEngineArgs(
            model=self.model_name,
            trust_remote_code=False,
            enable_prefix_caching=True,
        )
        self.engine = AsyncLLMEngine.from_engine_args(args)
        self.available = True

    async def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        if not self.available or self.engine is None:
            raise RuntimeError(self.startup_error or "vLLM worker is unavailable.")

        from vllm import SamplingParams

        sampling = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        request_id = f"omni-{id(prompt)}"
        final_output: Any | None = None

        # vLLM streams intermediate RequestOutput objects; retain the final one.
        async for output in self.engine.generate(prompt, sampling, request_id):
            final_output = output

        if final_output is None or not final_output.outputs:
            raise RuntimeError("vLLM produced no completion.")
        return str(final_output.outputs[0].text)
