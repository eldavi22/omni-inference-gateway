from __future__ import annotations

import asyncio
import json
from typing import Any


try:
    import sglang as sgl
except ImportError:  # Allows static inspection before optional installation.
    sgl = None


if sgl is not None:
    @sgl.function
    def structured_program(
        state: Any,
        prompt: str,
        max_tokens: int,
        structured: bool,
    ) -> None:
        """SGLang program: control flow is expressed in the prompt graph."""
        state += sgl.system(
            "You are a precise inference worker. "
            "Return valid JSON only when structured output is requested."
        )
        state += sgl.user(prompt)
        if structured:
            state += sgl.assistant(sgl.gen("answer", max_tokens=max_tokens))
        else:
            state += sgl.assistant(sgl.gen("answer", max_tokens=max_tokens))


class SGLangWorker:
    """Adapter for an SGLang runtime, optimized for structured workflows."""

    def __init__(self, endpoint: str | None) -> None:
        self.endpoint = endpoint
        self.runtime: Any | None = None
        self.available = False
        self.startup_error: str | None = None

    async def start(self) -> None:
        if not self.endpoint:
            self.startup_error = "SGLANG_URL is not configured."
            return
        if sgl is None:
            self.startup_error = "sglang package is not installed."
            return

        # The runtime can point to a separately scaled SGLang server.
        self.runtime = sgl.Runtime(self.endpoint)
        self.available = True

    def _run_sync(
        self,
        prompt: str,
        max_tokens: int,
        json_schema: dict[str, Any] | None,
    ) -> str:
        if self.runtime is None or sgl is None:
            raise RuntimeError(self.startup_error or "SGLang worker is unavailable.")

        state = structured_program.run(
            backend=self.runtime,
            prompt=prompt,
            max_tokens=max_tokens,
            structured=json_schema is not None,
        )
        answer = str(state["answer"])

        # SGLang controls generation flow; application-level validation makes
        # schema failures explicit to callers rather than silently accepting them.
        if json_schema is not None:
            parsed = json.loads(answer)
            if not isinstance(parsed, (dict, list)):
                raise ValueError("Structured SGLang response must be JSON object or array.")
            return json.dumps(parsed, ensure_ascii=False)
        return answer

    async def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        del temperature  # SGLang server owns sampling defaults in this adapter.
        if not self.available:
            raise RuntimeError(self.startup_error or "SGLang worker is unavailable.")
        return await asyncio.to_thread(
            self._run_sync, prompt, max_tokens, json_schema
        )
