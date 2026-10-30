import asyncio
import os
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backends.llama_worker import LlamaCppWorker
from backends.sglang_worker import SGLangWorker
from backends.vllm_worker import VLLMWorker
from metrics.jax_latency import latency_summary
from router_engine import BackendName, RouteDecision, RouterEngine
from utils.fast_tokenizer import FastTokenizer


class GenerateRequest(BaseModel):
    """Schema shared by all execution backends."""

    prompt: str = Field(min_length=1, max_length=100_000)
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    structured: bool = False
    json_schema: dict[str, Any] | None = None
    preferred_backend: BackendName | None = None


class GenerateResponse(BaseModel):
    text: str
    backend: BackendName
    token_count: int
    elapsed_ms: float
    route_reason: str
    attempted_backends: list[BackendName]


@dataclass
class Gateway:
    """Orchestrates tokenization, routing, execution and fallback.

    vLLM is preferred for throughput, SGLang is preferred for structured
    or multi-step work, and llama.cpp is the CPU/GGUF contingency path.
    All workers expose the same asynchronous ``generate`` contract.
    """

    tokenizer: FastTokenizer
    router: RouterEngine
    workers: dict[BackendName, Any]
    latency_samples_ms: list[float]

    @classmethod
    def from_environment(cls) -> "Gateway":
        return cls(
            tokenizer=FastTokenizer(os.getenv("TOKENIZER_JSON")),
            router=RouterEngine(),
            workers={
                "vllm": VLLMWorker(os.getenv("VLLM_MODEL")),
                "sglang": SGLangWorker(os.getenv("SGLANG_URL")),
                "llama_cpp": LlamaCppWorker(os.getenv("LLAMA_GGUF_MODEL")),
            },
            latency_samples_ms=[],
        )

    async def start(self) -> None:
        # Heavy model initialization is deliberately delayed until FastAPI starts.
        for worker in self.workers.values():
            try:
                await worker.start()
            except Exception as exc:
                worker.startup_error = str(exc)

    def availability(self) -> dict[BackendName, bool]:
        return {name: worker.available for name, worker in self.workers.items()}

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        encoding = self.tokenizer.encode(request.prompt)
        decision: RouteDecision = self.router.decide(
            token_count=len(encoding.ids),
            structured=request.structured or request.json_schema is not None,
            preferred_backend=request.preferred_backend,
            availability=self.availability(),
        )
        candidates = self.router.fallback_order(decision.backend, self.availability())
        attempted: list[BackendName] = []
        last_error = "No backend was configured."

        for backend_name in candidates:
            worker = self.workers[backend_name]
            attempted.append(backend_name)
            started = time.perf_counter()
            try:
                text = await worker.generate(
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    json_schema=request.json_schema,
                )
                elapsed_ms = (time.perf_counter() - started) * 1_000.0
                self.latency_samples_ms.append(elapsed_ms)
                self.latency_samples_ms = self.latency_samples_ms[-10_000:]
                return GenerateResponse(
                    text=text,
                    backend=backend_name,
                    token_count=len(encoding.ids),
                    elapsed_ms=round(elapsed_ms, 3),
                    route_reason=decision.reason,
                    attempted_backends=attempted,
                )
            except Exception as exc:
                # Failure of one engine is intentionally contained: the next
                # compatible engine may still serve the same normalized request.
                last_error = f"{backend_name}: {exc}"

        raise RuntimeError(f"All compatible backends failed. Last error: {last_error}")


gateway = Gateway.from_environment()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await gateway.start()
    yield


app = FastAPI(
    title="Omni-Inference Gateway",
    version="1.0.0",
    lifespan=lifespan,
    description="Unified FastAPI router for vLLM, SGLang and llama.cpp.",
)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "backends": {
            name: {
                "available": worker.available,
                "startup_error": worker.startup_error,
            }
            for name, worker in gateway.workers.items()
        },
    }


@app.post("/v1/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    try:
        return await gateway.generate(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/metrics/latency")
async def metrics_latency() -> dict[str, float]:
    if not gateway.latency_samples_ms:
        return {"count": 0.0}
    # JAX JIT compiles this summary when the service first observes metrics.
    return latency_summary(gateway.latency_samples_ms)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
