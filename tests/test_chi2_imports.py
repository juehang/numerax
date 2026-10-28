"""
Test that all chi-squared functions can be imported and called.
"""

import jax.numpy as jnp

from numerax.stats import chi2

from ._precision import cast

# float32 ppf cases fail: see the known-bug note in
# test_gammap_inverse.py.


def _check_finite(result, dtype):
    assert result.dtype == dtype
    assert jnp.isfinite(result)


def test_pdf(dtype):
    """Test pdf function works."""
    _check_finite(chi2.pdf(cast(1.0, dtype), df=cast(2.0, dtype)), dtype)


def test_logpdf(dtype):
    """Test logpdf function works."""
    _check_finite(chi2.logpdf(cast(1.0, dtype), df=cast(2.0, dtype)), dtype)


def test_cdf(dtype):
    """Test cdf function works."""
    _check_finite(chi2.cdf(cast(1.0, dtype), df=cast(2.0, dtype)), dtype)


def test_logcdf(dtype):
    """Test logcdf function works."""
    _check_finite(chi2.logcdf(cast(1.0, dtype), df=cast(2.0, dtype)), dtype)


def test_sf(dtype):
    """Test sf function works."""
    _check_finite(chi2.sf(cast(1.0, dtype), df=cast(2.0, dtype)), dtype)


def test_logsf(dtype):
    """Test logsf function works."""
    _check_finite(chi2.logsf(cast(1.0, dtype), df=cast(2.0, dtype)), dtype)


def test_ppf(dtype):
    """Test ppf function works."""
    _check_finite(chi2.ppf(cast(0.5, dtype), df=cast(2.0, dtype)), dtype)


def test_loc_scale_parameters(dtype):
    """Test that loc and scale parameters work."""
    x, df, loc, scale = (cast(v, dtype) for v in (1.0, 2.0, 0.5, 2.0))
    _check_finite(chi2.pdf(x, df=df, loc=loc, scale=scale), dtype)
