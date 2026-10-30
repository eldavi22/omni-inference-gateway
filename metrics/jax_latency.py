from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp


@jax.jit
def _latency_kernel(values: jax.Array) -> jax.Array:
    """AOT/JIT-compilable latency summary for accelerator execution."""
    ordered = jnp.sort(values)
    size = ordered.shape[0]
    p50_index = jnp.minimum(size - 1, jnp.int32(0.50 * (size - 1)))
    p95_index = jnp.minimum(size - 1, jnp.int32(0.95 * (size - 1)))
    return jnp.array(
        [
            jnp.mean(values),
            ordered[p50_index],
            ordered[p95_index],
            jnp.max(values),
        ],
        dtype=jnp.float32,
    )


def latency_summary(samples_ms: Sequence[float]) -> dict[str, float]:
    if not samples_ms:
        return {"count": 0.0}

    values = jnp.asarray(samples_ms, dtype=jnp.float32)
    mean_ms, p50_ms, p95_ms, max_ms = _latency_kernel(values)
    return {
        "count": float(values.size),
        "mean_ms": float(mean_ms),
        "p50_ms": float(p50_ms),
        "p95_ms": float(p95_ms),
        "max_ms": float(max_ms),
    }
