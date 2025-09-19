"""
Test that all chi-squared functions can be imported and called.
"""

import jax.numpy as jnp

from numerax.stats import chi2


def test_pdf():
    """Test pdf function works."""
    result = chi2.pdf(1.0, df=2.0)
    assert jnp.isfinite(result)


def test_logpdf():
    """Test logpdf function works."""
    result = chi2.logpdf(1.0, df=2.0)
    assert jnp.isfinite(result)


def test_cdf():
    """Test cdf function works."""
    result = chi2.cdf(1.0, df=2.0)
    assert jnp.isfinite(result)


def test_logcdf():
    """Test logcdf function works."""
    result = chi2.logcdf(1.0, df=2.0)
    assert jnp.isfinite(result)


def test_sf():
    """Test sf function works."""
    result = chi2.sf(1.0, df=2.0)
    assert jnp.isfinite(result)


def test_logsf():
    """Test logsf function works."""
    result = chi2.logsf(1.0, df=2.0)
    assert jnp.isfinite(result)


def test_ppf():
    """Test ppf function works."""
    result = chi2.ppf(0.5, df=2.0)
    assert jnp.isfinite(result)


def test_loc_scale_parameters():
    """Test that loc and scale parameters work."""
    result = chi2.pdf(1.0, df=2.0, loc=0.5, scale=2.0)
    assert jnp.isfinite(result)
