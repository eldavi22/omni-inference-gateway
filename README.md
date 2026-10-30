# Omni-Inference Gateway

A modular **FastAPI inference gateway** that routes prompts to configured inference backends. Source code reconstructed from the owner's original archived module listing.

## What is implemented

- `main.py`: async FastAPI endpoints `GET /health`, `POST /v1/generate`, `GET /metrics/latency`, with fallback attempts and normalized responses.
- `router_engine.py`: deterministic five-feature, three-output PyTorch routing network with **explicitly initialized weights** (not a model trained from a dataset).
- `backends/`: adapters for vLLM, SGLang and llama.cpp.
- `utils/fast_tokenizer.py`: Hugging Face tokenization.
- `metrics/jax_latency.py`: JAX/JIT latency summary.

## Setup

Requires Python 3.11+ and platform-specific backend/model installations. The included `requirements.txt` documents full dependencies, some of which require additional GPU/OS support. For a CPU-only routing test, install only `torch` and run `python -m unittest discover -s tests -v`.

Configure a compatible model/runtime first, then set environment variables from `.env.example` in your shell. Run `python main.py` after all necessary dependencies are installed.

## Quality controls

`python -m compileall -q main.py router_engine.py backends utils metrics tests` checks Python syntax. The included unit tests check router output shape, availability, fallback ordering, explicit backend preference and safe handling when no backend is configured. GitHub Actions installs CPU-only PyTorch and runs checks.

## Limits and ownership

- No multi-engine production load test is claimed. Real vLLM/SGLang/GGUF installations and licensed model files are needed for end-to-end generation.
- This router uses hand-set weights; it is **not** a trained ML checkpoint.
- The code is shared under MIT; the license does not cover third-party models, binaries or datasets.
- The original archived code used five-dimensional input and a three-class layer but omitted a `forward()` method. This package adds that method, without changing the intended weights.
