"""Shared pytest configuration.

x64 is enabled here, once, before any test module (or numerax) is
imported, so a test file behaves the same whether it is run alone or with
the whole suite. float32 is still exercised by casting inputs to the
dtype under test: see the ``dtype`` fixture.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(params=[jnp.float32, jnp.float64], ids=["f32", "f64"])
def dtype(request):
    """Floating-point dtype under test; every test using it runs twice."""
    return request.param
