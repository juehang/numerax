"""
Test suite for gamma special functions.
"""

import jax
import jax.numpy as jnp
import pytest
from jax.scipy import special

from numerax.special import gammap_inverse

from ._precision import cast

# Known bug (float32 cases fail): TINY/EPS in _gamma.py are fixed at
# import time as float64 (x64 is on), so jnp.fmax(x, TINY) promotes the
# float32 Halley iterate and lax.while_loop raises a carry-dtype
# TypeError. chi2.ppf goes through gammap_inverse and fails the same way.


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
def test_gamma_inverse_correctness(p, a, dtype):
    """Test that gammap_inverse correctly inverts gammainc."""
    p, a = cast(p, dtype), cast(a, dtype)
    x = gammap_inverse(p, a)
    assert x.dtype == dtype
    result = special.gammainc(a, x)
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
def test_gamma_inverse_gradients(p, a, dtype):
    """Test that gradients of gammap_inverse are computed correctly."""
    p, a = cast(p, dtype), cast(a, dtype)
    # Compute x such that gammainc(a, x) = p
    x = gammap_inverse(p, a)

    # Manual gradient: 1 / d/dx gammainc(a, x)
    manual_grad = 1 / jax.grad(lambda x_val: special.gammainc(a, x_val))(x)

    # Autodiff gradient
    auto_grad = jax.grad(gammap_inverse)(p, a)
    assert auto_grad.dtype == dtype

    assert jnp.abs(manual_grad - auto_grad) < 1e-6
