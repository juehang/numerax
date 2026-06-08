"""Test suite for the non-central chi-squared distribution."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402
from scipy.stats import ncx2 as scipy_ncx2  # noqa: E402

from numerax.stats import chi2, ncx2  # noqa: E402

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
def test_pdf_against_scipy(x, df, nc):
    """numerax.stats.ncx2.pdf matches scipy.stats.ncx2.pdf."""
    expected = scipy_ncx2.pdf(x, df, nc=nc)
    actual = float(ncx2.pdf(x, df, nc))
    if expected == 0.0:
        assert actual < 1e-300
    else:
        assert abs(actual - expected) / abs(expected) < 1e-6


@pytest.mark.parametrize(("x", "df", "nc"), _GRID)
def test_logpdf_against_scipy(x, df, nc):
    """numerax.stats.ncx2.logpdf matches scipy.stats.ncx2.logpdf."""
    expected = scipy_ncx2.logpdf(x, df, nc=nc)
    actual = float(ncx2.logpdf(x, df, nc))
    # Compare in log-space with an absolute tolerance (relative error in
    # the density is ~1e-6, i.e. an additive ~1e-6 in the log).
    assert abs(actual - expected) < 1e-5


@pytest.mark.parametrize(
    ("x", "df", "nc"),
    [
        (50.0, 1.0, 50.0),  # z = sqrt(nc*x) = 50 -> Hankel regime, nu < 0
        (120.0, 30.0, 40.0),  # large z, large order -> Olver regime
        (80.0, 3.0, 90.0),
    ],
)
def test_pdf_and_grad_large_z(x, df, nc):
    """Drive ive into its z >= 30 asymptotic regimes through ncx2."""
    expected = scipy_ncx2.pdf(x, df, nc=nc)
    assert abs(float(ncx2.pdf(x, df, nc)) - expected) / abs(expected) < 1e-6
    analytic = float(jax.grad(ncx2.logpdf, argnums=0)(x, df, nc))
    h = 1e-5
    fd = (
        scipy_ncx2.logpdf(x + h, df, nc=nc)
        - scipy_ncx2.logpdf(x - h, df, nc=nc)
    ) / (2 * h)
    assert abs(analytic - fd) / abs(fd) < 1e-4


def test_pdf_spot_check_nc_two():
    """Explicit nc=2 spot check (z=sqrt(8), nu=0.5)."""
    expected = scipy_ncx2.pdf(4.0, 3, nc=2.0)
    actual = float(ncx2.pdf(4.0, 3.0, 2.0))
    assert abs(actual - expected) / abs(expected) < 1e-6


@pytest.mark.parametrize("x", [0.5, 1.0, 2.0, 5.0, 20.0])
@pytest.mark.parametrize("df", [0.5, 2.0, 5.0])
def test_nc_zero_matches_central_chi2(x, df):
    """nc=0 reduces to the central chi-squared distribution."""
    assert jnp.allclose(ncx2.pdf(x, df, 0.0), chi2.pdf(x, df), rtol=1e-10)
    assert jnp.allclose(
        ncx2.logpdf(x, df, 0.0), chi2.logpdf(x, df), rtol=1e-10
    )


@pytest.mark.parametrize(
    ("loc", "scale"),
    [(1.0, 1.0), (0.0, 2.0), (3.0, 1.5), (-2.0, 0.5)],
)
def test_loc_scale_against_scipy(loc, scale):
    """loc/scale parameterization matches scipy."""
    x, df, nc = 8.0, 3.0, 2.0
    expected = scipy_ncx2.pdf(x, df, nc=nc, loc=loc, scale=scale)
    actual = float(ncx2.pdf(x, df, nc, loc, scale))
    assert abs(actual - expected) / abs(expected) < 1e-6


def test_vectorized_and_broadcasting():
    """Array inputs and broadcasting produce per-element correct values."""
    x_arr = jnp.array([0.5, 1.0, 5.0])
    df_arr = jnp.array([1.0, 2.0, 5.0])
    nc = 1.5
    actual = ncx2.pdf(x_arr, df_arr, nc)
    for i, (x, df) in enumerate(
        zip(x_arr.tolist(), df_arr.tolist(), strict=True)
    ):
        expected = scipy_ncx2.pdf(x, df, nc=nc)
        assert abs(float(actual[i]) - expected) / abs(expected) < 1e-6


def test_jit():
    """Works under jax.jit."""
    jit_pdf = jax.jit(ncx2.pdf)
    for x, df, nc in [(1.0, 2.0, 1.0), (5.0, 3.0, 2.0)]:
        expected = scipy_ncx2.pdf(x, df, nc=nc)
        assert abs(float(jit_pdf(x, df, nc)) - expected) / expected < 1e-6


def test_vmap():
    """Composes with jax.vmap."""
    x_arr = jnp.array([0.5, 1.0, 5.0])
    df_arr = jnp.array([1.0, 2.0, 5.0])
    nc_arr = jnp.array([0.5, 1.0, 2.0])
    out = jax.vmap(ncx2.pdf)(x_arr, df_arr, nc_arr)
    for i in range(3):
        expected = scipy_ncx2.pdf(
            float(x_arr[i]), float(df_arr[i]), nc=float(nc_arr[i])
        )
        assert abs(float(out[i]) - expected) / abs(expected) < 1e-6


@pytest.mark.parametrize(
    ("x", "df", "nc"), [(4.0, 3.0, 2.0), (1.0, 5.0, 1.0), (8.0, 2.0, 5.0)]
)
def test_grad_wrt_x_matches_finite_diff(x, df, nc):
    """d logpdf / dx matches a finite-difference of scipy."""
    analytic = float(jax.grad(ncx2.logpdf, argnums=0)(x, df, nc))
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
def test_grad_small_df_matches_finite_diff(x, df, nc):
    """df < 2 gives Bessel order nu = df/2 - 1 in (-1, 0); the gradient
    path through ive must stay finite and match scipy."""
    h = 1e-6
    for arg in (0, 2):
        analytic = float(jax.grad(ncx2.logpdf, argnums=arg)(x, df, nc))
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
def test_grad_wrt_nc_matches_finite_diff(x, df, nc):
    """d logpdf / d nc matches a finite-difference of scipy."""
    analytic = float(jax.grad(ncx2.logpdf, argnums=2)(x, df, nc))
    h = 1e-5
    fd = (
        scipy_ncx2.logpdf(x, df, nc=nc + h)
        - scipy_ncx2.logpdf(x, df, nc=nc - h)
    ) / (2 * h)
    assert abs(analytic - fd) / abs(fd) < 1e-4


def test_grad_wrt_loc_scale_finite():
    """Gradients w.r.t. loc and scale are finite (differentiable)."""
    g_loc = float(jax.grad(ncx2.logpdf, argnums=3)(8.0, 3.0, 2.0, 1.0, 2.0))
    g_scale = float(jax.grad(ncx2.logpdf, argnums=4)(8.0, 3.0, 2.0, 1.0, 2.0))
    assert jnp.isfinite(jnp.asarray(g_loc))
    assert jnp.isfinite(jnp.asarray(g_scale))


def test_grad_wrt_nc_at_zero_is_finite_zero():
    """At exactly nc=0 (a domain boundary) the nc-gradient is finite and
    returned as 0, while any nc>0 (even tiny) gives the correct score."""
    g0 = float(jax.grad(ncx2.logpdf, argnums=2)(6.0, 3.0, 0.0))
    assert g0 == 0.0
    # Just above the boundary the true score 0.5*(x/df - 1) is recovered.
    g_eps = float(jax.grad(ncx2.logpdf, argnums=2)(6.0, 3.0, 1e-8))
    assert abs(g_eps - 0.5 * (6.0 / 3.0 - 1.0)) < 1e-5


def test_grad_wrt_df_raises():
    """Differentiating w.r.t. df is unsupported (the underlying ive has
    no order-derivative) and must raise rather than mislead."""
    with pytest.raises(TypeError):
        jax.grad(ncx2.logpdf, argnums=1)(4.0, 3.0, 2.0)
    with pytest.raises(TypeError):
        jax.grad(ncx2.pdf, argnums=1)(4.0, 3.0, 2.0)


@pytest.mark.parametrize("df", [0.5, 2.0, 5.0])
def test_below_support_is_zero(df):
    """x < loc gives zero density / -inf logpdf for any nc."""
    for nc in [0.0, 2.0]:
        assert float(ncx2.pdf(-1.0, df, nc, loc=0.0)) == 0.0
        assert float(ncx2.logpdf(-1.0, df, nc, loc=0.0)) == -jnp.inf
