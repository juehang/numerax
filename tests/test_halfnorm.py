"""Test suite for the half-normal distribution."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy.stats import halfnorm as scipy_halfnorm  # noqa: E402
from scipy.stats import norm as scipy_norm  # noqa: E402

from numerax.stats import halfnorm  # noqa: E402

# x straddles the y = 1 branch switch in logcdf/logsf and reaches the
# upper tail; q includes both endpoints and the q = 1/2 switch in ppf.
_X_VALUES = [0.0, 1e-8, 0.1, 0.5, 0.999, 1.0, 1.001, 3.0, 10.0, 30.0]
_Q_VALUES = [0.0, 0.01, 0.1, 0.3, 0.4999, 0.5, 0.5001, 0.9, 0.99, 1.0]
_X_FUNCS = ["pdf", "logpdf", "cdf", "logcdf", "sf", "logsf"]
_Q_FUNCS = ["ppf", "isf"]

# Relative tolerance against scipy. Deep in the upper tail (x = 30) the
# erfc-based functions are conditioned at ~2 * (x/sqrt(2))**2 ulps, so
# both libraries sit at ~1e-13 there; 1e-12 leaves headroom.
_RTOL = 1e-12


def _assert_close(actual, expected, rtol=_RTOL):
    actual = float(actual)
    if np.isinf(expected) or expected == 0.0:
        assert actual == expected
    else:
        assert abs(actual - expected) / abs(expected) < rtol


@pytest.mark.parametrize("fn", _X_FUNCS)
@pytest.mark.parametrize("x", _X_VALUES)
def test_x_functions_against_scipy(fn, x):
    """pdf/logpdf/cdf/logcdf/sf/logsf match scipy.stats.halfnorm."""
    _assert_close(getattr(halfnorm, fn)(x), getattr(scipy_halfnorm, fn)(x))


@pytest.mark.parametrize("fn", _Q_FUNCS)
@pytest.mark.parametrize("q", _Q_VALUES)
def test_q_functions_against_scipy(fn, q):
    """ppf/isf match scipy.stats.halfnorm."""
    _assert_close(getattr(halfnorm, fn)(q), getattr(scipy_halfnorm, fn)(q))


@pytest.mark.parametrize("q", [1e-300, 1e-100, 1e-10])
def test_ppf_tiny_q(q):
    """ppf keeps tiny q (scipy rounds (1+q)/2 to 1/2 and loses it).

    For tiny q, ppf(q) = sqrt(pi/2) q (1 + pi q^2 / 12 + ...), so the
    leading term is exact to double precision here."""
    _assert_close(halfnorm.ppf(q), np.sqrt(np.pi / 2) * q, rtol=1e-15)


@pytest.mark.parametrize("q", [1e-300, 1e-100, 1e-10])
def test_isf_tiny_q(q):
    """isf is accurate deep in the upper tail: isf(q) = -ndtri(q/2)."""
    _assert_close(halfnorm.isf(q), -scipy_norm.ppf(q / 2))


@pytest.mark.parametrize("x", [1e-300, 1e-100, 1e-8])
def test_small_x_relative_accuracy(x):
    """cdf ~ c = sqrt(2/pi) x and logsf ~ log1p(-c) keep relative
    accuracy near the origin (the neglected erf terms are O(x^2)
    relative)."""
    c = np.sqrt(2 / np.pi) * x
    _assert_close(halfnorm.cdf(x), c, rtol=1e-15)
    _assert_close(halfnorm.logsf(x), np.log1p(-c), rtol=1e-15)
    _assert_close(halfnorm.logcdf(x), np.log(c), rtol=1e-15)


@pytest.mark.parametrize("x", [20.0, 25.0, 34.999, 35.0, 35.001, 50.0])
def test_logsf_and_grad_upper_tail(x):
    """logsf and its gradient stay accurate through the erfc -> asymptotic
    switch at y = 35 (and where jax's log_ndtr loses ~1e-9 in the
    gradient). d logsf/dx = -pdf/sf, evaluated in log space."""
    expected = np.log(2.0) + scipy_norm.logsf(x)
    _assert_close(halfnorm.logsf(x), expected, rtol=1e-14)
    grad = -np.exp(scipy_norm.logpdf(x) - scipy_norm.logsf(x))
    _assert_close(jax.grad(halfnorm.logsf)(x), grad, rtol=1e-12)


@pytest.mark.parametrize("x", [40.0, 100.0, 1e3])
def test_logsf_past_erfc_underflow(x):
    """logsf stays finite and accurate where sf underflows to zero
    (scipy.stats.halfnorm.logsf returns -inf here)."""
    expected = np.log(2.0) + scipy_norm.logsf(x)
    assert float(halfnorm.sf(x)) == 0.0
    _assert_close(halfnorm.logsf(x), expected)


@pytest.mark.parametrize(
    ("loc", "scale"),
    [(1.0, 1.0), (0.0, 2.0), (3.0, 1.5), (-2.0, 0.5)],
)
def test_loc_scale_against_scipy(loc, scale):
    """loc/scale parameterization matches scipy for every function."""
    x = loc + 1.7 * scale
    for fn in _X_FUNCS:
        expected = getattr(scipy_halfnorm, fn)(x, loc=loc, scale=scale)
        _assert_close(getattr(halfnorm, fn)(x, loc, scale), expected)
    for fn in _Q_FUNCS:
        expected = getattr(scipy_halfnorm, fn)(0.3, loc=loc, scale=scale)
        _assert_close(getattr(halfnorm, fn)(0.3, loc, scale), expected)


def test_below_support():
    """x < loc: zero density, cdf 0, sf 1 (and their logs)."""
    x, loc = -1.0, 0.0
    assert float(halfnorm.pdf(x, loc)) == 0.0
    assert float(halfnorm.logpdf(x, loc)) == -jnp.inf
    assert float(halfnorm.cdf(x, loc)) == 0.0
    assert float(halfnorm.logcdf(x, loc)) == -jnp.inf
    assert float(halfnorm.sf(x, loc)) == 1.0
    assert float(halfnorm.logsf(x, loc)) == 0.0


def test_quantile_endpoints_and_domain():
    """ppf/isf endpoints map to loc/inf, and q outside [0, 1] is nan."""
    loc = 2.0
    assert float(halfnorm.ppf(0.0, loc)) == loc
    assert float(halfnorm.ppf(1.0, loc)) == jnp.inf
    assert float(halfnorm.isf(0.0, loc)) == jnp.inf
    assert float(halfnorm.isf(1.0, loc)) == loc
    for q in [-0.1, 1.1]:
        assert jnp.isnan(halfnorm.ppf(q))
        assert jnp.isnan(halfnorm.isf(q))


@pytest.mark.parametrize("x", [1e-8, 0.1, 1.0, 2.0, 3.0])
def test_ppf_cdf_round_trip(x):
    """ppf(cdf(x)) recovers x."""
    _assert_close(halfnorm.ppf(halfnorm.cdf(x)), x)


@pytest.mark.parametrize("x", [0.1, 1.0, 3.0, 10.0, 30.0])
def test_isf_sf_round_trip(x):
    """isf(sf(x)) recovers x, including deep in the upper tail."""
    _assert_close(halfnorm.isf(halfnorm.sf(x)), x)


def test_vectorized_and_broadcasting():
    """Array inputs broadcast against array loc/scale."""
    x = jnp.array([[0.5], [1.0], [5.0]])
    scale = jnp.array([1.0, 2.0])
    actual = halfnorm.pdf(x, 0.0, scale)
    assert actual.shape == (3, 2)
    expected = scipy_halfnorm.pdf(np.asarray(x), scale=np.asarray(scale))
    np.testing.assert_allclose(actual, expected, rtol=_RTOL)


def test_jit():
    """Works under jax.jit."""
    args = (0.3, 0.1, 2.0)
    for fn in _X_FUNCS + _Q_FUNCS:
        jitted = jax.jit(getattr(halfnorm, fn))
        _assert_close(jitted(*args), getattr(halfnorm, fn)(*args))


def test_vmap():
    """Composes with jax.vmap."""
    x = jnp.array([0.5, 1.0, 5.0])
    loc = jnp.array([0.0, 0.5, 1.0])
    scale = jnp.array([1.0, 2.0, 0.5])
    out = jax.vmap(halfnorm.logpdf)(x, loc, scale)
    expected = scipy_halfnorm.logpdf(
        np.asarray(x), loc=np.asarray(loc), scale=np.asarray(scale)
    )
    np.testing.assert_allclose(out, expected, rtol=_RTOL)


@pytest.mark.parametrize("x", [0.1, 0.5, 1.0, 1.5, 3.0, 20.0])
def test_grad_wrt_x_closed_forms(x):
    """x-gradients match their closed forms on both sides of y = 1."""
    p = scipy_halfnorm.pdf(x)
    g = lambda fn: float(jax.grad(getattr(halfnorm, fn))(x))  # noqa: E731
    _assert_close(g("logpdf"), -x, rtol=1e-12)
    _assert_close(g("cdf"), p, rtol=1e-12)
    _assert_close(g("sf"), -p, rtol=1e-12)
    _assert_close(g("logcdf"), p / scipy_halfnorm.cdf(x), rtol=1e-10)
    # d logsf / dx = -pdf / sf, the inverse Mills ratio form.
    expected = -np.exp(scipy_halfnorm.logpdf(x) - scipy_halfnorm.logsf(x))
    _assert_close(g("logsf"), expected, rtol=1e-10)


@pytest.mark.parametrize("q", [0.01, 0.3, 0.5, 0.7, 0.99])
def test_grad_wrt_q_closed_forms(q):
    """d ppf/dq = 1/pdf(ppf(q)) and d isf/dq = -1/pdf(isf(q))."""
    g_ppf = float(jax.grad(halfnorm.ppf)(q))
    g_isf = float(jax.grad(halfnorm.isf)(q))
    _assert_close(g_ppf, 1 / scipy_halfnorm.pdf(scipy_halfnorm.ppf(q)), 1e-10)
    _assert_close(g_isf, -1 / scipy_halfnorm.pdf(scipy_halfnorm.isf(q)), 1e-10)


def test_grad_wrt_loc_scale():
    """loc/scale gradients match closed forms for logpdf and ppf."""
    x, loc, scale = 3.0, 1.0, 2.0
    y = (x - loc) / scale
    g_loc = float(jax.grad(halfnorm.logpdf, argnums=1)(x, loc, scale))
    g_scale = float(jax.grad(halfnorm.logpdf, argnums=2)(x, loc, scale))
    _assert_close(g_loc, y / scale)
    _assert_close(g_scale, (y**2 - 1) / scale)
    q = 0.3
    assert float(jax.grad(halfnorm.ppf, argnums=1)(q, loc, scale)) == 1.0
    _assert_close(
        jax.grad(halfnorm.ppf, argnums=2)(q, loc, scale),
        scipy_halfnorm.ppf(q),
    )


@pytest.mark.parametrize("fn", _X_FUNCS)
@pytest.mark.parametrize("x", [-1.0, 0.0])
def test_grad_finite_at_and_below_support(fn, x):
    """Gradients at x = loc and below the support are finite (no NaN
    leaks from the unselected branches)."""
    for argnums in (0, 1, 2):
        g = jax.grad(getattr(halfnorm, fn), argnums=argnums)(x, 0.0, 1.0)
        assert jnp.isfinite(g)


def test_grad_at_support_edge_is_one_sided():
    """At x = loc, cdf/sf/logsf give the right derivative +-pdf(0)."""
    p0 = np.sqrt(2 / np.pi)
    _assert_close(jax.grad(halfnorm.cdf)(0.0), p0)
    _assert_close(jax.grad(halfnorm.sf)(0.0), -p0)
    _assert_close(jax.grad(halfnorm.logsf)(0.0), -p0)
    assert float(halfnorm.logsf(0.0)) == 0.0


@pytest.mark.parametrize("fn", _Q_FUNCS)
@pytest.mark.parametrize("q", [0.0, 0.5])
def test_grad_finite_at_quantile_boundaries(fn, q):
    """ppf/isf gradients are finite at q = 0 and at the q = 1/2 switch."""
    g = jax.grad(getattr(halfnorm, fn))(q)
    if fn == "isf" and q == 0.0:
        # isf diverges at q = 0, so its derivative does too.
        assert jnp.isinf(g) or jnp.abs(g) > 1e10
    else:
        assert jnp.isfinite(g)
