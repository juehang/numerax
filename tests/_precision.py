"""Helpers for running tests in both float32 and float64.

Tests take the ``dtype`` fixture from ``conftest.py`` (or use
``per_dtype`` for dtype-specific inputs), cast their inputs with
``cast``, and compare against float64 references evaluated at the exact
dtype-rounded input with ``check``.
"""

import jax.numpy as jnp
import numpy as np
import pytest


def per_dtype(f32, f64):
    """Parametrize ``(dtype, value)`` with dtype-specific values, for
    inputs that only make sense in one precision (e.g. 1e-300)."""
    return [pytest.param(jnp.float32, v, id=f"f32-{v}") for v in f32] + [
        pytest.param(jnp.float64, v, id=f"f64-{v}") for v in f64
    ]


def cast(value, dtype):
    return jnp.asarray(value, dtype)


def rtol_for(dtype, cond=0.0, f64=1e-12, f32=1e-6):
    """Base tolerance for the dtype, widened to ~8 ulps times the
    condition number where that is larger."""
    eps = float(jnp.finfo(dtype).eps)
    base = f64 if dtype == jnp.float64 else f32
    return max(base, 8 * eps * (1 + cond))


def check(actual, expected, dtype, rtol):
    """``actual`` has the requested dtype and matches ``expected`` to
    ``rtol``. An expected value below the dtype's normal range only
    requires a (near) zero result; one beyond its range requires the
    correspondingly signed infinity."""
    assert actual.dtype == dtype
    actual, expected = float(actual), float(expected)
    info = jnp.finfo(dtype)
    if np.isinf(expected) or expected == 0.0:
        assert actual == expected
    elif abs(expected) > float(info.max):
        assert actual == np.copysign(np.inf, expected)
    elif abs(expected) < float(info.tiny):
        assert abs(actual) < float(info.tiny)
    else:
        assert abs(actual - expected) / abs(expected) <= rtol, (
            actual,
            expected,
        )
