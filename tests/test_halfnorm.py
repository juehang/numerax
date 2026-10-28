"""Test suite for the half-normal distribution.

Every test runs in both float32 and float64 (x64 is enabled in
conftest.py, and inputs are cast to the dtype under test). References are
float64 values from scipy or closed forms, evaluated at the exact
dtype-rounded input.
"""

import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import halfnorm as scipy_halfnorm
from scipy.stats import norm as scipy_norm

from numerax.stats import halfnorm

from ._precision import cast, check, per_dtype, rtol_for

_X_FUNCS = ["pdf", "logpdf", "cdf", "logcdf", "sf", "logsf"]
_Q_FUNCS = ["ppf", "isf"]

# x straddles the y = 1 switch in logcdf/logsf and the logsf asymptotic
# switch in each dtype (y = 8 in float32, y = 20 in float64), and reaches
# past where float32 erfc underflows (y ~ 13.3). q includes both endpoints
# and the q = 1/2 switch in ppf.
_X_VALUES = [
    *[0.0, 1e-8, 0.1, 0.5, 0.999, 1.0, 1.001, 3.0],
    *[7.999, 8.0, 8.001, 10.0, 13.5, 19.999, 20.0, 20.001, 30.0],
]
_Q_VALUES = [0.0, 0.01, 0.1, 0.3, 0.4999, 0.5, 0.5001, 0.9, 0.99, 1.0]

# Functions carrying e^{-x^2/2}, whose relative error grows like x^2 ulps
# (rounding of y / sqrt(2) inside erfc/exp). logcdf ~ -sf in the upper
# tail. Their x-gradients (all proportional to the pdf) are conditioned
# the same way, as is the gradient of cdf.
_TAIL_CONDITIONED = {"pdf", "sf", "logcdf"}
_TAIL_CONDITIONED_GRAD = _TAIL_CONDITIONED | {"cdf"}


def _mills(x):
    """Inverse Mills ratio phi(x) / Phi(-x), evaluated in log space."""
    return np.exp(scipy_norm.logpdf(x) - scipy_norm.logsf(x))


def _quantile_cond(q, x):
    """Relative condition number |q x'(q) / x| of a quantile function at
    probability q with quantile x, where |x'(q)| = 1 / pdf(x)."""
    return q / (x * scipy_halfnorm.pdf(x))


@pytest.mark.parametrize("fn", _X_FUNCS)
@pytest.mark.parametrize("x", _X_VALUES)
def test_x_functions_against_scipy(fn, x, dtype):
    """pdf/logpdf/cdf/logcdf/sf/logsf match scipy.stats.halfnorm."""
    xd = cast(x, dtype)
    cond = float(xd) ** 2 if fn in _TAIL_CONDITIONED else 0.0
    expected = getattr(scipy_halfnorm, fn)(float(xd))
    check(getattr(halfnorm, fn)(xd), expected, dtype, rtol_for(dtype, cond))


@pytest.mark.parametrize("fn", _Q_FUNCS)
@pytest.mark.parametrize("q", _Q_VALUES)
def test_q_functions_against_scipy(fn, q, dtype):
    """ppf/isf match scipy.stats.halfnorm."""
    qd = cast(q, dtype)
    expected = getattr(scipy_halfnorm, fn)(float(qd))
    cond = 0.0
    if 0.0 < float(qd) < 1.0:
        p = float(qd) if fn == "ppf" else 1.0 - float(qd)
        cond = _quantile_cond(p, expected)
    check(getattr(halfnorm, fn)(qd), expected, dtype, rtol_for(dtype, cond))


@pytest.mark.parametrize(
    ("dtype", "q"), per_dtype([1e-30, 1e-20, 1e-10], [1e-300, 1e-100, 1e-10])
)
def test_ppf_tiny_q(dtype, q):
    """ppf keeps tiny q (scipy rounds (1+q)/2 to 1/2 and loses it).

    For tiny q, ppf(q) = sqrt(pi/2) q (1 + pi q^2 / 12 + ...), so the
    leading term is exact to working precision here."""
    qd = cast(q, dtype)
    expected = np.sqrt(np.pi / 2) * float(qd)
    check(halfnorm.ppf(qd), expected, dtype, rtol_for(dtype, f64=1e-15))


@pytest.mark.parametrize(
    ("dtype", "q"), per_dtype([1e-30, 1e-20, 1e-10], [1e-300, 1e-100, 1e-10])
)
def test_isf_tiny_q(dtype, q):
    """isf is accurate deep in the upper tail: isf(q) = -ndtri(q/2)."""
    qd = cast(q, dtype)
    check(
        halfnorm.isf(qd),
        -scipy_norm.ppf(float(qd) / 2),
        dtype,
        rtol_for(dtype),
    )


@pytest.mark.parametrize(
    ("dtype", "x"), per_dtype([1e-30, 1e-20, 1e-8], [1e-300, 1e-100, 1e-8])
)
def test_small_x_relative_accuracy(dtype, x):
    """cdf ~ c = sqrt(2/pi) x and logsf ~ log1p(-c) keep relative
    accuracy near the origin (the neglected erf terms are O(x^2)
    relative)."""
    xd = cast(x, dtype)
    c = np.sqrt(2 / np.pi) * float(xd)
    rtol = rtol_for(dtype, f64=1e-15)
    check(halfnorm.cdf(xd), c, dtype, rtol)
    check(halfnorm.logsf(xd), np.log1p(-c), dtype, rtol)
    check(halfnorm.logcdf(xd), np.log(c), dtype, rtol)


@pytest.mark.parametrize(
    ("dtype", "x"),
    per_dtype(
        [5.0, 7.999, 8.0, 8.001, 13.5, 30.0],
        [15.0, 19.999, 20.0, 20.001, 25.0, 50.0],
    ),
)
def test_logsf_and_grad_upper_tail(dtype, x):
    """logsf and its gradient stay accurate through the erfc -> asymptotic
    switch (y = 8 in float32, y = 20 in float64; the latter is also where
    jax's log_ndtr loses ~1e-9 in the gradient), and past float32 erfc
    underflow. d logsf/dx = -phi/Phi(-x), the inverse Mills ratio."""
    xd = cast(x, dtype)
    expected = np.log(2.0) + scipy_norm.logsf(float(xd))
    check(halfnorm.logsf(xd), expected, dtype, rtol_for(dtype, f64=1e-14))
    grad = jax.grad(halfnorm.logsf)(xd)
    check(grad, -_mills(float(xd)), dtype, rtol_for(dtype))


@pytest.mark.parametrize(
    ("dtype", "x"),
    per_dtype(
        [2.0, 5.0, 7.999, 8.0, 9.5, 13.5, 30.0],
        [2.0, 10.0, 19.999, 20.0, 26.5, 30.0, 50.0],
    ),
)
def test_logsf_second_derivative(dtype, x):
    """d^2 logsf/dx^2 = m (x - m) for the inverse Mills ratio m stays
    finite and accurate (log(erfc) would form an underflowing erfc**2
    here if the asymptotic switch came later). The m - x cancellation
    costs ~x^2 ulps, hence the looser tolerance."""
    xd = cast(x, dtype)
    m = _mills(float(xd))
    hess = jax.hessian(halfnorm.logsf)(xd)
    check(
        hess, m * (float(xd) - m), dtype, rtol_for(dtype, f64=1e-9, f32=2e-5)
    )


@pytest.mark.parametrize(
    ("dtype", "x"), per_dtype([15.0, 30.0, 1e3], [40.0, 100.0, 1e3])
)
def test_logsf_past_erfc_underflow(dtype, x):
    """logsf stays finite and accurate where sf underflows to zero
    (scipy.stats.halfnorm.logsf returns -inf here in float64)."""
    xd = cast(x, dtype)
    assert float(halfnorm.sf(xd)) == 0.0
    expected = np.log(2.0) + scipy_norm.logsf(float(xd))
    check(halfnorm.logsf(xd), expected, dtype, rtol_for(dtype))


@pytest.mark.parametrize(
    ("loc", "scale"),
    [(1.0, 1.0), (0.0, 2.0), (3.0, 1.5), (-2.0, 0.5)],
)
def test_loc_scale_against_scipy(loc, scale, dtype):
    """loc/scale parameterization matches scipy for every function."""
    x, loc, scale = (cast(v, dtype) for v in (loc + 1.7 * scale, loc, scale))
    ref = {"loc": float(loc), "scale": float(scale)}
    # Rounding of x - loc in float32 perturbs y by ~(|x| + |loc|) ulps.
    cond = (abs(float(x)) + abs(float(loc))) / float(scale) * 1.7
    for fn in _X_FUNCS:
        expected = getattr(scipy_halfnorm, fn)(float(x), **ref)
        actual = getattr(halfnorm, fn)(x, loc, scale)
        check(actual, expected, dtype, rtol_for(dtype, cond))
    q = cast(0.3, dtype)
    for fn in _Q_FUNCS:
        expected = getattr(scipy_halfnorm, fn)(float(q), **ref)
        actual = getattr(halfnorm, fn)(q, loc, scale)
        check(actual, expected, dtype, rtol_for(dtype))


def test_below_support(dtype):
    """x < loc: zero density, cdf 0, sf 1 (and their logs)."""
    x, loc = cast(-1.0, dtype), cast(0.0, dtype)
    for fn, expected in [
        ("pdf", 0.0),
        ("logpdf", -np.inf),
        ("cdf", 0.0),
        ("logcdf", -np.inf),
        ("sf", 1.0),
        ("logsf", 0.0),
    ]:
        check(getattr(halfnorm, fn)(x, loc), expected, dtype, 0.0)


def test_quantile_endpoints_and_domain(dtype):
    """ppf/isf endpoints map to loc/inf, and q outside [0, 1] is nan."""
    loc = cast(2.0, dtype)
    zero, one = cast(0.0, dtype), cast(1.0, dtype)
    check(halfnorm.ppf(zero, loc), 2.0, dtype, 0.0)
    check(halfnorm.ppf(one, loc), np.inf, dtype, 0.0)
    check(halfnorm.isf(zero, loc), np.inf, dtype, 0.0)
    check(halfnorm.isf(one, loc), 2.0, dtype, 0.0)
    for q in [-0.1, 1.1]:
        assert jnp.isnan(halfnorm.ppf(cast(q, dtype)))
        assert jnp.isnan(halfnorm.isf(cast(q, dtype)))


@pytest.mark.parametrize("x", [1e-8, 0.1, 1.0, 2.0, 3.0])
def test_ppf_cdf_round_trip(x, dtype):
    """ppf(cdf(x)) recovers x, up to the rounding of cdf(x) amplified by
    the conditioning of ppf."""
    xd = cast(x, dtype)
    cond = _quantile_cond(scipy_halfnorm.cdf(float(xd)), float(xd))
    check(
        halfnorm.ppf(halfnorm.cdf(xd)), float(xd), dtype, rtol_for(dtype, cond)
    )


@pytest.mark.parametrize(
    ("dtype", "x"),
    per_dtype([0.1, 1.0, 3.0, 8.0, 12.0], [0.1, 1.0, 3.0, 10.0, 30.0]),
)
def test_isf_sf_round_trip(dtype, x):
    """isf(sf(x)) recovers x, including deep in the upper tail."""
    xd = cast(x, dtype)
    check(halfnorm.isf(halfnorm.sf(xd)), float(xd), dtype, rtol_for(dtype))


def test_vectorized_and_broadcasting(dtype):
    """Array inputs broadcast against array loc/scale."""
    x = cast([[0.5], [1.0], [5.0]], dtype)
    scale = cast([1.0, 2.0], dtype)
    actual = halfnorm.pdf(x, cast(0.0, dtype), scale)
    assert actual.shape == (3, 2)
    assert actual.dtype == dtype
    expected = scipy_halfnorm.pdf(np.asarray(x), scale=np.asarray(scale))
    np.testing.assert_allclose(actual, expected, rtol=rtol_for(dtype, 25.0))


def test_jit(dtype):
    """Works under jax.jit."""
    args = tuple(cast(v, dtype) for v in (0.3, 0.1, 2.0))
    for fn in _X_FUNCS + _Q_FUNCS:
        jitted = jax.jit(getattr(halfnorm, fn))
        expected = getattr(halfnorm, fn)(*args)
        check(jitted(*args), expected, dtype, rtol_for(dtype))


def test_vmap(dtype):
    """Composes with jax.vmap."""
    x = cast([0.5, 1.0, 5.0], dtype)
    loc = cast([0.0, 0.5, 1.0], dtype)
    scale = cast([1.0, 2.0, 0.5], dtype)
    out = jax.vmap(halfnorm.logpdf)(x, loc, scale)
    assert out.dtype == dtype
    expected = scipy_halfnorm.logpdf(
        np.asarray(x), loc=np.asarray(loc), scale=np.asarray(scale)
    )
    np.testing.assert_allclose(out, expected, rtol=rtol_for(dtype, 8.0))


@pytest.mark.parametrize("fn", _X_FUNCS)
@pytest.mark.parametrize("x", [v for v in _X_VALUES if v > 0])
def test_grad_wrt_x_closed_forms(fn, x, dtype):
    """x-gradients match their closed forms, across every branch switch.

    logpdf' = -x, cdf' = pdf, sf' = -pdf, logcdf' = pdf / cdf, and
    logsf' = -phi / Phi(-x), the inverse Mills ratio (pdf' = -x pdf)."""
    xd = cast(x, dtype)
    xf = float(xd)
    p = scipy_halfnorm.pdf(xf)
    expected = {
        "pdf": -xf * p,
        "logpdf": -xf,
        "cdf": p,
        "logcdf": p / scipy_halfnorm.cdf(xf),
        "sf": -p,
        "logsf": -_mills(xf),
    }[fn]
    cond = xf**2 if fn in _TAIL_CONDITIONED_GRAD else 0.0
    grad = jax.grad(getattr(halfnorm, fn))(xd)
    assert jnp.isfinite(grad)
    check(grad, expected, dtype, rtol_for(dtype, cond))


@pytest.mark.parametrize(
    ("dtype", "x"),
    per_dtype(
        [1e-30, 1e-20, 1e-8, 5e-8], [1e-300, 1e-100, 1e-20, 1e-17, 1e-8]
    ),
)
def test_logcdf_grad_tiny_x(dtype, x):
    """logcdf ~ log(x) + const as x -> 0+, so its gradient is 1/x. The
    unused log1p(-erfc) branch must not leak log1p(-1) = -inf here."""
    xd, loc, scale = cast(x, dtype), cast(0.0, dtype), cast(1.0, dtype)
    xf = float(xd)
    for argnums, expected in [(0, 1 / xf), (1, -1 / xf), (2, -1.0)]:
        g = jax.grad(halfnorm.logcdf, argnums=argnums)(xd, loc, scale)
        check(g, expected, dtype, rtol_for(dtype))


@pytest.mark.parametrize("q", [0.01, 0.3, 0.5, 0.7, 0.99])
def test_grad_wrt_q_closed_forms(q, dtype):
    """d ppf/dq = 1/pdf(ppf(q)) and d isf/dq = -1/pdf(isf(q)).

    The relative condition number of 1/pdf(x(q)) in q is q x x'(q)."""
    qd = cast(q, dtype)
    qf = float(qd)
    for fn, sign, x in [
        ("ppf", 1.0, scipy_halfnorm.ppf(qf)),
        ("isf", -1.0, scipy_halfnorm.isf(qf)),
    ]:
        g = jax.grad(getattr(halfnorm, fn))(qd)
        inv_pdf = 1 / scipy_halfnorm.pdf(x)
        cond = qf * x * inv_pdf
        check(g, sign * inv_pdf, dtype, rtol_for(dtype, cond))


def test_grad_wrt_loc_scale(dtype):
    """loc/scale gradients match closed forms for logpdf and ppf."""
    x, loc, scale = (cast(v, dtype) for v in (3.0, 1.0, 2.0))
    y = (float(x) - float(loc)) / float(scale)
    g_loc = jax.grad(halfnorm.logpdf, argnums=1)(x, loc, scale)
    g_scale = jax.grad(halfnorm.logpdf, argnums=2)(x, loc, scale)
    check(g_loc, y / float(scale), dtype, rtol_for(dtype))
    check(g_scale, (y**2 - 1) / float(scale), dtype, rtol_for(dtype))
    q = cast(0.3, dtype)
    check(jax.grad(halfnorm.ppf, argnums=1)(q, loc, scale), 1.0, dtype, 0.0)
    check(
        jax.grad(halfnorm.ppf, argnums=2)(q, loc, scale),
        scipy_halfnorm.ppf(float(q)),
        dtype,
        rtol_for(dtype),
    )


@pytest.mark.parametrize("fn", _X_FUNCS)
@pytest.mark.parametrize("x", [-1.0, 0.0])
def test_grad_finite_at_and_below_support(fn, x, dtype):
    """Gradients at x = loc and below the support are finite (no NaN
    leaks from the unselected branches)."""
    args = (cast(x, dtype), cast(0.0, dtype), cast(1.0, dtype))
    for argnums in (0, 1, 2):
        g = jax.grad(getattr(halfnorm, fn), argnums=argnums)(*args)
        assert g.dtype == dtype
        assert jnp.isfinite(g)


def test_grad_at_support_edge_is_one_sided(dtype):
    """At x = loc, cdf/sf/logsf give the right derivative +-pdf(0)."""
    p0 = np.sqrt(2 / np.pi)
    zero = cast(0.0, dtype)
    check(jax.grad(halfnorm.cdf)(zero), p0, dtype, rtol_for(dtype))
    check(jax.grad(halfnorm.sf)(zero), -p0, dtype, rtol_for(dtype))
    check(jax.grad(halfnorm.logsf)(zero), -p0, dtype, rtol_for(dtype))
    check(halfnorm.logsf(zero), 0.0, dtype, 0.0)


@pytest.mark.parametrize("fn", _Q_FUNCS)
@pytest.mark.parametrize("q", [0.0, 0.5])
def test_grad_finite_at_quantile_boundaries(fn, q, dtype):
    """ppf/isf gradients are finite at q = 0 and at the q = 1/2 switch."""
    g = jax.grad(getattr(halfnorm, fn))(cast(q, dtype))
    assert g.dtype == dtype
    if fn == "isf" and q == 0.0:
        # isf diverges at q = 0, so its derivative does too.
        assert jnp.isinf(g) or jnp.abs(g) > 1e10
    else:
        assert jnp.isfinite(g)


@pytest.mark.parametrize("fn", _X_FUNCS + _Q_FUNCS)
def test_nan_propagates(fn, dtype):
    """nan inputs give nan (not a below-support value), as in scipy."""
    out = getattr(halfnorm, fn)(cast(np.nan, dtype))
    assert out.dtype == dtype
    assert jnp.isnan(out)


def test_import_before_enabling_x64():
    """Enabling x64 after importing numerax still gives float64 accuracy
    (module constants must not be frozen as float32 arrays). float64-only
    by construction."""
    script = """
import numerax
import jax
jax.config.update("jax_enable_x64", True)
from scipy.stats import halfnorm as ref
h = numerax.stats.halfnorm
for got, want in [
    (h.cdf(1.0), ref.cdf(1.0)),
    (h.sf(30.0), ref.sf(30.0)),
    (h.logsf(5.0), ref.logsf(5.0)),
    (h.ppf(0.3), ref.ppf(0.3)),
]:
    assert abs(float(got) - want) / abs(want) < 1e-12, (float(got), want)
"""
    # Trusted input: a fixed script run by the current interpreter.
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
