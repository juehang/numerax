"""
Test suite for error functions and Fresnel integrals.
"""

import jax
import jax.numpy as jnp
import pytest
from scipy.special import erfcinv as scipy_erfcinv

from numerax.special import erfcinv


@pytest.mark.parametrize(
    "x",
    [
        0.01,
        0.1,
        0.25,
        0.5,
        0.75,
        0.9,
        1.0,
        1.1,
        1.5,
        1.9,
        1.99,
    ],
)
def test_erfcinv_against_scipy(x):
    """Test erfcinv values against SciPy reference implementation."""
    scipy_result = scipy_erfcinv(x)
    numerax_result = erfcinv(x)
    assert jnp.allclose(numerax_result, scipy_result, rtol=1e-6)


def test_erfcinv_vectorized():
    """Test erfcinv with vectorized inputs."""
    x_vals = jnp.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 1.9])

    # Test vectorized computation
    scipy_results = jnp.array([scipy_erfcinv(x) for x in x_vals])
    numerax_results = erfcinv(x_vals)
    assert jnp.allclose(numerax_results, scipy_results, rtol=1e-6)


def test_erfcinv_gradient_exists():
    """Test that gradients can be computed without errors."""
    # Simple smoke test - verify gradient doesn't crash or return NaN
    grad_fn = jax.grad(erfcinv)

    test_vals = [0.1, 0.5, 1.0, 1.5, 1.9]
    for x in test_vals:
        grad_val = grad_fn(x)
        assert jnp.isfinite(grad_val), f"Gradient not finite at x={x}"


def test_erfcinv_jit_compatible():
    """Test that erfcinv works with JAX JIT compilation."""
    jit_erfcinv = jax.jit(erfcinv)

    x = 0.5
    result = jit_erfcinv(x)
    expected = scipy_erfcinv(x)
    assert jnp.allclose(result, expected, rtol=1e-6)
