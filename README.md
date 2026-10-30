# Omni-Inference Gateway

Omni-Inference Gateway is a Python project I built to experiment with a single API that can work with different language model inference engines. Instead of tying an application to one runtime, the gateway checks which backends are available, chooses a compatible one, and can try another if a request fails.

The application uses FastAPI and includes adapters for vLLM, SGLang, and llama.cpp. A small PyTorch network scores routing options based on five input features. I initialized its weights manually, so this is a routing experiment rather than a model trained on a dataset.

Hugging Face Tokenizers handles tokenization, while JAX provides latency summaries. The API exposes three endpoints:

- `GET /health` shows backend availability.
- `POST /v1/generate` accepts generation requests and returns a consistent response format.
- `GET /metrics/latency` reports timing statistics from completed requests.

## Getting started

You'll need Python 3.11 or newer. Actual text generation also requires a compatible inference runtime and model files. The full set of dependencies is listed in `requirements.txt`, although some packages need specific operating systems or GPU environments.

Configure the variables described in `.env.example` for the backends you want to use. Once the required dependencies and models are installed, start the API with:

```bash
python main.py
```

The repository's `backends/` folder contains the runtime adapters. Routing logic lives in `router_engine.py`, tokenization utilities are under `utils/`, and JAX metrics are under `metrics/`.

## Testing

I included focused tests for the routing logic, backend availability, selection order, fallback behavior, and the case where no backend is configured.

```bash
python -m unittest discover -s tests -v
```

You can run the routing tests with CPU-only PyTorch without installing every inference engine. GitHub Actions runs these tests and checks the Python syntax on each push and pull request.

## Current scope

The gateway's routing and adapter code is available here, but I haven't included model weights or claimed that all three engines have been benchmarked together in production. End-to-end generation depends on the individual runtimes and models being installed and configured.

The published version was assembled from my archived source files, with a missing PyTorch `forward()` method added during cleanup. The project code is released under the MIT license; external model files and third-party libraries have their own terms.
