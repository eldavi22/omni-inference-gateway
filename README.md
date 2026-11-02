**Omni-Inference Gateway**

I built Omni-Inference Gateway to explore a simpler way to work with different AI model engines through one application. Instead of relying on a single engine, it can check what's available, route a request, and try another option when needed.
The project uses Python and FastAPI, with PyTorch for routing and JAX for timing measurements. It includes connections for vLLM, SGLang, and llama.cpp, along with automated tests for the routing logic.
The code and setup files are included here. Running full text generation requires compatible models and the relevant engines to be installed separately. The project is available under the MIT license.
