from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

BackendName = Literal["vllm", "sglang", "llama_cpp"]


@dataclass(frozen=True)
class RouteDecision:
    backend: BackendName
    confidence: float
    reason: str


class IntentRoutingNetwork(nn.Module):
    """Tiny deterministic PyTorch intent classifier.

    Features encode request length, structured-output intent, chaining hints
    and resource constraints. In a production installation its weights can
    be replaced by a trained checkpoint without changing the gateway API.
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(5, 3)
        with torch.no_grad():
            # Class order: vLLM, SGLang, llama.cpp.
            self.linear.weight.copy_(
                torch.tensor(
                    [
                        [1.40, -1.20, -0.70, -1.10, -1.10],
                        [-0.25, 2.30, 1.70, -0.80, 2.10],
                        [-0.60, -0.70, -0.40, 2.60, -0.50],
                    ],
                    dtype=torch.float32,
                )
            )
            self.linear.bias.copy_(torch.tensor([0.45, -0.10, -0.15]))


    def forward(self, features: Tensor) -> Tensor:
        """Return routing logits for the five input features."""
        return self.linear(features)


class RouterEngine:
    labels: tuple[BackendName, ...] = ("vllm", "sglang", "llama_cpp")

    def __init__(self) -> None:
        torch.set_grad_enabled(False)
        self.model = IntentRoutingNetwork().eval()

    def decide(
        self,
        token_count: int,
        structured: bool,
        preferred_backend: BackendName | None,
        availability: dict[BackendName, bool],
    ) -> RouteDecision:
        if preferred_backend is not None and availability.get(preferred_backend, False):
            return RouteDecision(
                backend=preferred_backend,
                confidence=1.0,
                reason="Backend explicitly requested by client.",
            )

        long_context = min(token_count / 2048.0, 1.0)
        structured_value = float(structured)
        # In a real system these can come from scheduler telemetry.
        chained_hint = structured_value
        cpu_only = float(
            not availability.get("vllm", False)
            and not availability.get("sglang", False)
        )
        constrained = float(availability.get("llama_cpp", False) and cpu_only)
        features = torch.tensor(
            [[long_context, structured_value, chained_hint, constrained, structured_value]],
            dtype=torch.float32,
        )

        with torch.inference_mode():
            probabilities: Tensor = torch.softmax(self.model(features), dim=1)[0]
        index = int(torch.argmax(probabilities).item())
        proposed = self.labels[index]

        if availability.get(proposed, False):
            selected = proposed
        else:
            selected = next(
                (name for name in self.labels if availability.get(name, False)),
                proposed,
            )

        return RouteDecision(
            backend=selected,
            confidence=float(probabilities[index].item()),
            reason=(
                f"PyTorch intent classifier selected {selected} "
                f"(structured={structured}, tokens={token_count})."
            ),
        )

    def fallback_order(
        self,
        preferred: BackendName,
        availability: dict[BackendName, bool],
    ) -> list[BackendName]:
        ordered = [preferred] + [name for name in self.labels if name != preferred]
        return [name for name in ordered if availability.get(name, False)]
