"""
Test suite for chi-squared percent point function (ppf).
"""

import jax
import jax.numpy as jnp
import pytest
from scipy.stats import chi2 as scipy_chi2

from numerax.stats import chi2


@pytest.mark.parametrize(
    "q,df",
    [
        (0.1, 0.5),
        (0.3, 1.0),
        (0.5, 1.5),
        (0.8, 2.0),
        (0.95, 3.0),
    ],
)
def test_chi2_ppf_against_scipy(q, df):
    """Test PPF values against SciPy reference implementation."""
    scipy_result = scipy_chi2.ppf(q, df)
    numerax_result = chi2.ppf(q, df)
    assert jnp.allclose(numerax_result, scipy_result, rtol=5e-6)


@pytest.mark.parametrize(
    "q,df,loc,scale",
    [
        (0.5, 2.0, 0, 1),
        (0.5, 2.0, 1.0, 1),
        (0.5, 2.0, 0, 2.0),
        (0.5, 2.0, 3.0, 1.5),
    ],
)
def test_chi2_ppf_location_scale(q, df, loc, scale):
    """Test PPF with location and scale parameters."""
    standard_ppf = scipy_chi2.ppf(q, df)
    expected = loc + scale * standard_ppf
    result = chi2.ppf(q, df, loc=loc, scale=scale)
    assert jnp.allclose(result, expected, rtol=5e-6)


def test_chi2_ppf_vectorized():
    """Test PPF with vectorized inputs."""
    q_vals = jnp.array([0.1, 0.25, 0.5, 0.75, 0.9])

    # Test with array q, scalar df
    scipy_results = scipy_chi2.ppf(q_vals, 2.0)
    numerax_results = chi2.ppf(q_vals, 2.0)
    assert jnp.allclose(numerax_results, scipy_results, rtol=5e-6)

    # Test with scalar q, array df
    df_vals = jnp.array([0.5, 1.0, 2.0, 5.0, 10.0])
    scipy_results = scipy_chi2.ppf(0.5, df_vals)
    numerax_results = chi2.ppf(0.5, df_vals)
    assert jnp.allclose(numerax_results, scipy_results, rtol=5e-6)


@pytest.mark.parametrize(
    "q,df",
    [
        (0.2, 0.5),
        (0.4, 1.0),
        (0.6, 1.5),
        (0.8, 2.0),
        (0.9, 2.5),
    ],
)
def test_chi2_ppf_gradients(q, df):
    """Test gradients against numerical differentiation of SciPy."""
    delta = 1e-6

    # Numerical gradient w.r.t. q using SciPy
    dq_numerical = (
        scipy_chi2.ppf(q + delta, df) - scipy_chi2.ppf(q - delta, df)
    ) / (2 * delta)

    # JAX automatic differentiation
    dq_jax = jax.grad(lambda q_val: chi2.ppf(q_val, df))(q)

    # Compare with relative tolerance
    assert jnp.allclose(dq_jax, dq_numerical, rtol=1e-4)
