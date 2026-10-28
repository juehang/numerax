"""
Test suite for chi-squared percent point function (ppf).
"""

import jax
import jax.numpy as jnp
import pytest
from scipy.stats import chi2 as scipy_chi2

from numerax.stats import chi2

from ._precision import cast

# float32 ppf cases fail: see the known-bug note in
# test_gammap_inverse.py.


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
def test_chi2_ppf_against_scipy(q, df, dtype):
    """Test PPF values against SciPy reference implementation."""
    q, df = cast(q, dtype), cast(df, dtype)
    scipy_result = scipy_chi2.ppf(float(q), float(df))
    numerax_result = chi2.ppf(q, df)
    assert numerax_result.dtype == dtype
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
def test_chi2_ppf_location_scale(q, df, loc, scale, dtype):
    """Test PPF with location and scale parameters."""
    q, df, loc, scale = (cast(v, dtype) for v in (q, df, loc, scale))
    standard_ppf = scipy_chi2.ppf(float(q), float(df))
    expected = float(loc) + float(scale) * standard_ppf
    result = chi2.ppf(q, df, loc=loc, scale=scale)
    assert result.dtype == dtype
    assert jnp.allclose(result, expected, rtol=5e-6)


def test_chi2_ppf_vectorized(dtype):
    """Test PPF with vectorized inputs."""
    q_vals = cast([0.1, 0.25, 0.5, 0.75, 0.9], dtype)
    df = cast(2.0, dtype)

    # Test with array q, scalar df
    scipy_results = scipy_chi2.ppf(q_vals.astype(jnp.float64), 2.0)
    numerax_results = chi2.ppf(q_vals, df)
    assert numerax_results.dtype == dtype
    assert jnp.allclose(numerax_results, scipy_results, rtol=5e-6)

    # Test with scalar q, array df
    q = cast(0.5, dtype)
    df_vals = cast([0.5, 1.0, 2.0, 5.0, 10.0], dtype)
    scipy_results = scipy_chi2.ppf(0.5, df_vals.astype(jnp.float64))
    numerax_results = chi2.ppf(q, df_vals)
    assert numerax_results.dtype == dtype
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
def test_chi2_ppf_gradients(q, df, dtype):
    """Test gradients against numerical differentiation of SciPy."""
    qd, dfd = cast(q, dtype), cast(df, dtype)
    q, df = float(qd), float(dfd)
    delta = 1e-6

    # Numerical gradient w.r.t. q using SciPy
    dq_numerical = (
        scipy_chi2.ppf(q + delta, df) - scipy_chi2.ppf(q - delta, df)
    ) / (2 * delta)

    # JAX automatic differentiation
    dq_jax = jax.grad(lambda q_val: chi2.ppf(q_val, dfd))(qd)
    assert dq_jax.dtype == dtype

    # Compare with relative tolerance
    assert jnp.allclose(dq_jax, dq_numerical, rtol=1e-4)
