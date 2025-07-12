"""
Test suite for gamma special functions.
"""

import jax
import jax.numpy as jnp
import pytest
from jax.scipy import special

from numerax.special import gammap_inverse


@pytest.mark.parametrize(
    "p,a",
    [
        (0.1, 0.5),
        (0.3, 1.0),
        (0.5, 1.5),
        (0.8, 2.0),
        (0.95, 3.0),
    ],
)
def test_gamma_inverse_correctness(p, a):
    """Test that gammap_inverse correctly inverts gammainc."""
    result = special.gammainc(a, gammap_inverse(p, a))
    assert jnp.abs(result - p) < 1e-6


@pytest.mark.parametrize(
    "p,a",
    [
        (0.2, 0.5),
        (0.4, 1.0),
        (0.6, 1.5),
        (0.8, 2.0),
        (0.9, 2.5),
    ],
)
def test_gamma_inverse_gradients(p, a):
    """Test that gradients of gammap_inverse are computed correctly."""
    # Compute x such that gammainc(a, x) = p
    x = gammap_inverse(p, a)

    # Manual gradient: 1 / d/dx gammainc(a, x)
    manual_grad = 1 / jax.grad(lambda x_val: special.gammainc(a, x_val))(x)

    # Autodiff gradient
    auto_grad = jax.grad(gammap_inverse)(p, a)

    assert jnp.abs(manual_grad - auto_grad) < 1e-6
