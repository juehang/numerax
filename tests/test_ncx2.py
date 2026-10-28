"""Test suite for the non-central chi-squared distribution."""

import jax
import jax.numpy as jnp
import pytest
from scipy.stats import ncx2 as scipy_ncx2

from numerax.stats import chi2, ncx2

from ._precision import cast, check

# float32 tolerances cover the float32 accuracy of ive (up to ~6e-6 away
# from its Olver regime) plus the conditioning of exp(logpdf).
_PDF_RTOL = {jnp.float32: 2e-5, jnp.float64: 1e-6}
_LOGPDF_ATOL = {jnp.float32: 2e-5, jnp.float64: 1e-5}


def _cast_all(dtype, *values):
    """Cast inputs to dtype; also return the float64 values they round to,
    at which the scipy references are evaluated."""
    cast_values = tuple(cast(v, dtype) for v in values)
    return cast_values, tuple(float(v) for v in cast_values)


# Grid covering small/large x, df below/above 2, and nc including the
# central (nc=0) case and nc=2 (which discriminates the lambda vs
# lambda^2 parameterization -- nc=1 would not).
_DF_VALUES = [0.5, 1.0, 2.0, 5.0, 10.0, 25.0]
_NC_VALUES = [0.0, 0.5, 1.0, 2.0, 5.0]
_X_VALUES = [0.01, 0.1, 1.0, 5.0, 25.0, 100.0]
_GRID = [
    (x, df, nc) for df in _DF_VALUES for nc in _NC_VALUES for x in _X_VALUES
]


@pytest.mark.parametrize(("x", "df", "nc"), _GRID)
def test_pdf_against_scipy(x, df, nc, dtype):
    """numerax.stats.ncx2.pdf matches scipy.stats.ncx2.pdf."""
    args, (x, df, nc) = _cast_all(dtype, x, df, nc)
    expected = scipy_ncx2.pdf(x, df, nc=nc)
    check(ncx2.pdf(*args), expected, dtype, _PDF_RTOL[dtype])


@pytest.mark.parametrize(("x", "df", "nc"), _GRID)
def test_logpdf_against_scipy(x, df, nc, dtype):
    """numerax.stats.ncx2.logpdf matches scipy.stats.ncx2.logpdf."""
    args, (x, df, nc) = _cast_all(dtype, x, df, nc)
    expected = scipy_ncx2.logpdf(x, df, nc=nc)
    actual = ncx2.logpdf(*args)
    assert actual.dtype == dtype
    # Compare in log-space with an absolute tolerance (relative error in
    # the density is ~1e-6, i.e. an additive ~1e-6 in the log).
    assert abs(float(actual) - expected) < _LOGPDF_ATOL[dtype]


@pytest.mark.parametrize(
    ("x", "df", "nc"),
    [
        (50.0, 1.0, 50.0),  # z = sqrt(nc*x) = 50 -> Hankel regime, nu < 0
        (120.0, 30.0, 40.0),  # large z, large order -> Olver regime
        (80.0, 3.0, 90.0),
    ],
)
def test_pdf_and_grad_large_z(x, df, nc, dtype):
    """Drive ive into its z >= 30 asymptotic regimes through ncx2."""
    args, (x, df, nc) = _cast_all(dtype, x, df, nc)
    expected = scipy_ncx2.pdf(x, df, nc=nc)
    check(ncx2.pdf(*args), expected, dtype, _PDF_RTOL[dtype])
    analytic = float(jax.grad(ncx2.logpdf, argnums=0)(*args))
    h = 1e-5
    fd = (
        scipy_ncx2.logpdf(x + h, df, nc=nc)
        - scipy_ncx2.logpdf(x - h, df, nc=nc)
    ) / (2 * h)
    assert abs(analytic - fd) / abs(fd) < 1e-4


def test_pdf_spot_check_nc_two(dtype):
    """Explicit nc=2 spot check (z=sqrt(8), nu=0.5)."""
    args, (x, df, nc) = _cast_all(dtype, 4.0, 3.0, 2.0)
    expected = scipy_ncx2.pdf(x, df, nc=nc)
    check(ncx2.pdf(*args), expected, dtype, _PDF_RTOL[dtype])


@pytest.mark.parametrize("x", [0.5, 1.0, 2.0, 5.0, 20.0])
@pytest.mark.parametrize("df", [0.5, 2.0, 5.0])
def test_nc_zero_matches_central_chi2(x, df, dtype):
    """nc=0 reduces to the central chi-squared distribution."""
    (x, df, nc), _ = _cast_all(dtype, x, df, 0.0)
    rtol = 1e-10 if dtype == jnp.float64 else 1e-6
    pdf, logpdf = ncx2.pdf(x, df, nc), ncx2.logpdf(x, df, nc)
    assert pdf.dtype == logpdf.dtype == dtype
    assert jnp.allclose(pdf, chi2.pdf(x, df), rtol=rtol)
    assert jnp.allclose(logpdf, chi2.logpdf(x, df), rtol=rtol)


@pytest.mark.parametrize(
    ("loc", "scale"),
    [(1.0, 1.0), (0.0, 2.0), (3.0, 1.5), (-2.0, 0.5)],
)
def test_loc_scale_against_scipy(loc, scale, dtype):
    """loc/scale parameterization matches scipy."""
    args, (x, df, nc, loc, scale) = _cast_all(dtype, 8.0, 3.0, 2.0, loc, scale)
    expected = scipy_ncx2.pdf(x, df, nc=nc, loc=loc, scale=scale)
    check(ncx2.pdf(*args), expected, dtype, _PDF_RTOL[dtype])


def test_vectorized_and_broadcasting(dtype):
    """Array inputs and broadcasting produce per-element correct values."""
    x_arr = cast([0.5, 1.0, 5.0], dtype)
    df_arr = cast([1.0, 2.0, 5.0], dtype)
    nc = cast(1.5, dtype)
    actual = ncx2.pdf(x_arr, df_arr, nc)
    for i, (x, df) in enumerate(
        zip(x_arr.tolist(), df_arr.tolist(), strict=True)
    ):
        expected = scipy_ncx2.pdf(x, df, nc=float(nc))
        check(actual[i], expected, dtype, _PDF_RTOL[dtype])


def test_jit(dtype):
    """Works under jax.jit."""
    jit_pdf = jax.jit(ncx2.pdf)
    for values in [(1.0, 2.0, 1.0), (5.0, 3.0, 2.0)]:
        args, (x, df, nc) = _cast_all(dtype, *values)
        expected = scipy_ncx2.pdf(x, df, nc=nc)
        check(jit_pdf(*args), expected, dtype, _PDF_RTOL[dtype])


def test_vmap(dtype):
    """Composes with jax.vmap."""
    x_arr = cast([0.5, 1.0, 5.0], dtype)
    df_arr = cast([1.0, 2.0, 5.0], dtype)
    nc_arr = cast([0.5, 1.0, 2.0], dtype)
    out = jax.vmap(ncx2.pdf)(x_arr, df_arr, nc_arr)
    for i in range(3):
        expected = scipy_ncx2.pdf(
            float(x_arr[i]), float(df_arr[i]), nc=float(nc_arr[i])
        )
        check(out[i], expected, dtype, _PDF_RTOL[dtype])


@pytest.mark.parametrize(
    ("x", "df", "nc"), [(4.0, 3.0, 2.0), (1.0, 5.0, 1.0), (8.0, 2.0, 5.0)]
)
def test_grad_wrt_x_matches_finite_diff(x, df, nc, dtype):
    """d logpdf / dx matches a finite-difference of scipy."""
    args, (x, df, nc) = _cast_all(dtype, x, df, nc)
    grad = jax.grad(ncx2.logpdf, argnums=0)(*args)
    assert grad.dtype == dtype
    analytic = float(grad)
    h = 1e-5
    fd = (
        scipy_ncx2.logpdf(x + h, df, nc=nc)
        - scipy_ncx2.logpdf(x - h, df, nc=nc)
    ) / (2 * h)
    assert abs(analytic - fd) / abs(fd) < 1e-4


@pytest.mark.parametrize(
    ("x", "df", "nc"),
    [(2.0, 0.5, 2.0), (2.0, 1.0, 1.0), (5.0, 1.5, 3.0)],
)
def test_grad_small_df_matches_finite_diff(x, df, nc, dtype):
    """df < 2 gives Bessel order nu = df/2 - 1 in (-1, 0); the gradient
    path through ive must stay finite and match scipy."""
    args, (x, df, nc) = _cast_all(dtype, x, df, nc)
    h = 1e-6
    for arg in (0, 2):
        analytic = float(jax.grad(ncx2.logpdf, argnums=arg)(*args))
        assert jnp.isfinite(jnp.asarray(analytic))
        if arg == 0:
            fd = (
                scipy_ncx2.logpdf(x + h, df, nc=nc)
                - scipy_ncx2.logpdf(x - h, df, nc=nc)
            ) / (2 * h)
        else:
            fd = (
                scipy_ncx2.logpdf(x, df, nc=nc + h)
                - scipy_ncx2.logpdf(x, df, nc=nc - h)
            ) / (2 * h)
        assert abs(analytic - fd) / abs(fd) < 1e-4


@pytest.mark.parametrize(
    ("x", "df", "nc"), [(4.0, 3.0, 2.0), (1.0, 5.0, 1.0), (8.0, 2.0, 5.0)]
)
def test_grad_wrt_nc_matches_finite_diff(x, df, nc, dtype):
    """d logpdf / d nc matches a finite-difference of scipy."""
    args, (x, df, nc) = _cast_all(dtype, x, df, nc)
    grad = jax.grad(ncx2.logpdf, argnums=2)(*args)
    assert grad.dtype == dtype
    analytic = float(grad)
    h = 1e-5
    fd = (
        scipy_ncx2.logpdf(x, df, nc=nc + h)
        - scipy_ncx2.logpdf(x, df, nc=nc - h)
    ) / (2 * h)
    assert abs(analytic - fd) / abs(fd) < 1e-4


def test_grad_wrt_loc_scale_finite(dtype):
    """Gradients w.r.t. loc and scale are finite (differentiable)."""
    args, _ = _cast_all(dtype, 8.0, 3.0, 2.0, 1.0, 2.0)
    for argnums in (3, 4):
        g = jax.grad(ncx2.logpdf, argnums=argnums)(*args)
        assert g.dtype == dtype
        assert jnp.isfinite(g)


def test_grad_wrt_nc_at_zero_is_finite_zero(dtype):
    """At exactly nc=0 (a domain boundary) the nc-gradient is finite and
    returned as 0, while any nc>0 (even tiny) gives the correct score."""
    args, _ = _cast_all(dtype, 6.0, 3.0, 0.0)
    g0 = float(jax.grad(ncx2.logpdf, argnums=2)(*args))
    assert g0 == 0.0
    # Just above the boundary the true score 0.5*(x/df - 1) is recovered.
    args, _ = _cast_all(dtype, 6.0, 3.0, 1e-8)
    g_eps = float(jax.grad(ncx2.logpdf, argnums=2)(*args))
    assert abs(g_eps - 0.5 * (6.0 / 3.0 - 1.0)) < 1e-5


def test_grad_wrt_df_raises(dtype):
    """Differentiating w.r.t. df is unsupported (the underlying ive has
    no order-derivative) and must raise rather than mislead."""
    args, _ = _cast_all(dtype, 4.0, 3.0, 2.0)
    with pytest.raises(TypeError):
        jax.grad(ncx2.logpdf, argnums=1)(*args)
    with pytest.raises(TypeError):
        jax.grad(ncx2.pdf, argnums=1)(*args)


@pytest.mark.parametrize("df", [0.5, 2.0, 5.0])
def test_below_support_is_zero(df, dtype):
    """x < loc gives zero density / -inf logpdf for any nc."""
    for nc in [0.0, 2.0]:
        (x, df_, nc_, loc), _ = _cast_all(dtype, -1.0, df, nc, 0.0)
        pdf = ncx2.pdf(x, df_, nc_, loc=loc)
        logpdf = ncx2.logpdf(x, df_, nc_, loc=loc)
        assert pdf.dtype == logpdf.dtype == dtype
        assert float(pdf) == 0.0
        assert float(logpdf) == -jnp.inf
