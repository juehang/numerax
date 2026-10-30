"""Test suite for the split-normal distribution.

Every test runs in both float32 and float64 (x64 is enabled in
conftest.py, and inputs are cast to the dtype under test). scipy has no
split normal, so references are float64 values assembled from
``scipy.stats.norm`` and ``scipy.stats.halfnorm`` piecewise, evaluated at
the exact dtype-rounded input; quantile references are found by root
finding on the reference cdf/sf. Independent checks against
``scipy.stats.norm`` (equal scales) and numerical integration of the
density cover the piecewise formulas themselves.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import erfinv
from scipy.stats import halfnorm as scipy_halfnorm
from scipy.stats import norm as scipy_norm

from numerax.stats import splitnorm

from ._precision import cast, check, per_dtype, rtol_for

_X_FUNCS = ["pdf", "logpdf", "cdf", "logcdf", "sf", "logsf"]
_Q_FUNCS = ["ppf", "isf"]

# (loc, scale_left, scale_right): symmetric, wider on the right (w_left =
# 0.2), wider on the left with an offset mode (w_left ~ 0.81), and
# strongly asymmetric each way (one weight ~ 1e-3, the other ~ 1).
_PARAMS = [
    (0.0, 1.0, 1.0),
    (0.0, 0.5, 2.0),
    (1.5, 3.0, 0.7),
    (0.0, 1e-3, 1.0),
    (0.0, 1.0, 1e-3),
]
# Scale ratios at which one weight rounds to exactly one in each dtype.
_EXTREME_RATIO = {jnp.float32: 1e-8, jnp.float64: 1e-17}

# Standardized offsets y = (x - loc) / scale on each side of the mode.
# They straddle the halfnorm.logsf switches (y = 1, and the asymptotic
# switch at 8 in float32 and 20 in float64) and reach past where float32
# erfc underflows (y ~ 13.3).
_Y_VALUES = [
    *[-30.0, -20.0, -13.5, -8.0, -3.0, -1.0, -0.3, -1e-8],
    *[0.0, 1e-8, 0.3, 1.0, 3.0, 8.0, 13.5, 20.0, 30.0],
]
_Q_VALUES = [0.0, 1e-6, 0.01, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 0.99, 1.0]

# Tighter than the rtol_for default (1e-12), which would hide errors of a
# few hundred ulps.
_F64_RTOL = 1e-14


def _weights(sl, sr):
    return sl / (sl + sr), sr / (sl + sr)


def _log_weights(sl, sr):
    return -np.log1p(sr / sl), -np.log1p(sl / sr)


def _half_logsf(y):
    """log of the standard half-normal survival function at y >= 0, in
    a form with relative accuracy (log 2 + norm.logsf(y) cancels near
    y = 0)."""
    if y < 1:
        return np.log1p(-scipy_halfnorm.cdf(y))
    return np.log(2.0) + scipy_norm.logsf(y)


def _log_near(w_near, w_far, cdf_far, sf_far):
    """log(w_near + w_far cdf_far) = log1p(-w_far sf_far), in whichever
    form is accurate."""
    total = w_near + w_far * cdf_far
    return np.log(total) if total < 0.5 else np.log1p(-w_far * sf_far)


def _ref(fn, x, loc, sl, sr):
    """float64 reference for an x-function, from scipy pieces."""
    wl, wr = _weights(sl, sr)
    log_wl, log_wr = _log_weights(sl, sr)
    d = x - loc
    y = d / (sl if d < 0 else sr)
    if fn == "logpdf":
        return np.log(2 / (sl + sr)) + scipy_norm.logpdf(y)
    if fn == "pdf":
        return 2 / (sl + sr) * scipy_norm.pdf(y)
    if fn in ("cdf", "logcdf"):
        if d < 0:
            lo = log_wl + _half_logsf(-y)
            return lo if fn == "logcdf" else np.exp(lo)
        if fn == "cdf":
            return wl + wr * scipy_halfnorm.cdf(y)
        return _log_near(wl, wr, scipy_halfnorm.cdf(y), 2 * scipy_norm.sf(y))
    if d >= 0:
        hi = log_wr + _half_logsf(y)
        return hi if fn == "logsf" else np.exp(hi)
    if fn == "sf":
        return wr + wl * scipy_halfnorm.cdf(-y)
    return _log_near(wr, wl, scipy_halfnorm.cdf(-y), 2 * scipy_norm.cdf(y))


def _ref_quantile(fn, q, loc, sl, sr):
    """float64 reference for ppf/isf by root finding on the reference
    cdf/sf. The smaller of the two tail probabilities is matched, so
    1 - q (exact for q >= 1/2) never loses information."""
    p_below, p_above = (q, 1 - q) if fn == "ppf" else (1 - q, q)
    if p_below in (0.0, 1.0):
        return -np.inf if p_below == 0.0 else np.inf
    if p_below <= 0.5:
        target, ref = p_below, "cdf"
    else:
        target, ref = p_above, "sf"
    return brentq(
        lambda x: _ref(ref, x, loc, sl, sr) - target,
        loc - 40 * sl,
        loc + 40 * sr,
        xtol=1e-300,
        rtol=4 * np.finfo(float).eps,
        maxiter=500,
    )


def _check_quantile(actual, expected, fn, loc, sl, sr, dtype):
    """``actual`` matches ``expected`` to within the problem's own
    conditioning. q itself is exact, so beyond relative rounding of the
    result (and of the loc shift) the only unavoidable error is in the
    probability the quantile is computed from: the tail probability p in
    the tails of a piece, and in its central half the offset q - w from
    the weight w (computed from the scales, hence rounded) that the
    quantile is measured from. Forming 1 - q for q < 1/2 is not
    allowed for."""
    assert actual.dtype == dtype
    actual = float(actual)
    if np.isinf(expected):
        assert actual == expected
        return
    eps = float(jnp.finfo(dtype).eps)
    wl, wr = _weights(sl, sr)
    left = expected < loc
    w = wl if left else wr
    p_tail = _ref("cdf" if left else "sf", expected, loc, sl, sr)
    w_ref = wl if fn == "ppf" else wr
    if p_tail > 0.5 * w:
        offset = abs(_ref("cdf", expected, loc, sl, sr) - wl)
        q_err = w_ref + offset
    else:
        q_err = p_tail
    pdf = np.exp(_ref("logpdf", expected, loc, sl, sr))
    allowed = rtol_for(dtype, f64=_F64_RTOL) * abs(expected) + 8 * eps * (
        q_err / pdf + abs(loc)
    )
    assert abs(actual - expected) <= allowed, (actual, expected)


def _tail_conditioned(fn, d):
    """Values (and x-gradients) carrying e^{-y^2/2} on the given side of
    the mode, whose relative error grows like y^2 ulps. logcdf ~ -sf
    above the mode and logsf ~ -cdf below it."""
    if fn == "pdf":
        return True
    if d < 0:
        return fn in ("cdf", "logsf")
    return fn in ("sf", "logcdf")


def _args(dtype, *values):
    return tuple(cast(v, dtype) for v in values)


@pytest.mark.parametrize("params", _PARAMS)
@pytest.mark.parametrize("fn", _X_FUNCS)
@pytest.mark.parametrize("y", _Y_VALUES)
def test_x_functions_against_reference(y, fn, params, dtype):
    """pdf/logpdf/cdf/logcdf/sf/logsf match the piecewise scipy
    reference on both sides of the mode and far into both tails."""
    loc, sl, sr = params
    x = loc + y * (sl if y < 0 else sr)
    xd, locd, sld, srd = _args(dtype, x, loc, sl, sr)
    xf, locf, slf, srf = (float(v) for v in (xd, locd, sld, srd))
    d = xf - locf
    yf = d / (slf if d < 0 else srf)
    cond = yf**2 if _tail_conditioned(fn, d) else 0.0
    expected = _ref(fn, xf, locf, slf, srf)
    actual = getattr(splitnorm, fn)(xd, locd, sld, srd)
    check(actual, expected, dtype, rtol_for(dtype, cond, f64=_F64_RTOL))


@pytest.mark.parametrize("params", _PARAMS)
@pytest.mark.parametrize("fn", _Q_FUNCS)
@pytest.mark.parametrize("q", _Q_VALUES)
def test_q_functions_against_reference(q, fn, params, dtype):
    """ppf/isf match the reference, including at the w_left (ppf) and
    w_right (isf) switches and on the far side of a small weight (e.g.
    ppf(0.01) with w_left ~ 1e-3, where forming 1 - q loses digits)."""
    qd, locd, sld, srd = _args(dtype, q, *params)
    qf, locf, slf, srf = (float(v) for v in (qd, locd, sld, srd))
    expected = _ref_quantile(fn, qf, locf, slf, srf)
    actual = getattr(splitnorm, fn)(qd, locd, sld, srd)
    _check_quantile(actual, expected, fn, locf, slf, srf, dtype)


@pytest.mark.parametrize("fn", _X_FUNCS)
@pytest.mark.parametrize("x", [-40.0, -8.0, -1.0, 0.0, 0.5, 3.0, 40.0])
def test_equal_scales_is_normal(fn, x, dtype):
    """With scale_left = scale_right the split normal is N(loc, scale^2):
    an independent check against scipy.stats.norm."""
    xd, loc, scale = _args(dtype, x, 1.0, 2.0)
    yf = (float(xd) - 1.0) / 2.0
    expected = getattr(scipy_norm, fn)(float(xd), 1.0, 2.0)
    cond = yf**2 if _tail_conditioned(fn, yf) else 0.0
    actual = getattr(splitnorm, fn)(xd, loc, scale, scale)
    check(actual, expected, dtype, rtol_for(dtype, cond))


@pytest.mark.parametrize("fn", _Q_FUNCS)
@pytest.mark.parametrize("q", [1e-30, 1e-10, 0.1, 0.5, 0.9, 1 - 1e-6])
def test_equal_scales_quantiles_are_normal(fn, q, dtype):
    """ppf/isf with equal scales match scipy.stats.norm."""
    qd, loc, scale = _args(dtype, q, 1.0, 2.0)
    expected = getattr(scipy_norm, fn)(float(qd), 1.0, 2.0)
    # The loc shift cancels near q = 1/2, where x ~ loc.
    cond = 2.0 / max(abs(expected), 1e-300)
    actual = getattr(splitnorm, fn)(qd, loc, scale, scale)
    check(actual, expected, dtype, rtol_for(dtype, cond))


@pytest.mark.parametrize("params", _PARAMS[1:])
@pytest.mark.parametrize("x", [-4.0, -0.5, 0.0, 0.7, 1.5, 4.0])
def test_cdf_is_integrated_pdf(x, params):
    """cdf equals the numerical integral of pdf, and the density
    integrates to one. float64 only."""
    loc, sl, sr = params
    x = jnp.float64(x)

    def density(t):
        return float(splitnorm.pdf(jnp.float64(t), loc, sl, sr))

    lower = quad(density, -np.inf, loc)[0]
    upper = quad(density, loc, np.inf)[0]
    np.testing.assert_allclose(lower + upper, 1.0, rtol=1e-10)
    # Integrate each side of the mode separately (the density's second
    # derivative jumps there).
    integral = quad(density, -np.inf, min(float(x), loc))[0]
    if float(x) > loc:
        integral += quad(density, loc, float(x))[0]
    np.testing.assert_allclose(
        splitnorm.cdf(x, loc, sl, sr), integral, rtol=1e-8
    )


@pytest.mark.parametrize(
    ("dtype", "q"), per_dtype([1e-30, 1e-20, 1e-10], [1e-300, 1e-100, 1e-10])
)
def test_quantiles_tiny_q(dtype, q):
    """ppf(q) is accurate deep in the lower tail and isf(q) deep in the
    upper tail (no 1 - q is formed)."""
    qd, loc, sl, sr = _args(dtype, q, 0.0, 0.5, 2.0)
    wl, wr = _weights(float(sl), float(sr))
    rtol = rtol_for(dtype)
    expected = 0.5 * scipy_norm.ppf(float(qd) / (2 * wl))
    check(splitnorm.ppf(qd, loc, sl, sr), expected, dtype, rtol)
    expected = -2.0 * scipy_norm.ppf(float(qd) / (2 * wr))
    check(splitnorm.isf(qd, loc, sl, sr), expected, dtype, rtol)


@pytest.mark.parametrize(
    ("dtype", "case"),
    per_dtype(
        [(0.01, 1e-3), (1e-6, 1e-8), (0.3, 0.1)],
        [(0.01, 1e-3), (1e-3, 1e-4), (0.3, 0.1)],
    ),
)
def test_quantiles_far_side_of_small_weight(dtype, case):
    """ppf(q) for q a little above a small w_left (and isf mirrored) is
    accurate: it inverts F = w_left + w_right erf(x / sqrt(2) s_right)
    from the offset q - w_left, rather than from 1 - q, whose rounding
    costs ~(1 - q) / q relative."""
    q, ratio = case
    qd, loc, one, small = _args(dtype, q, 0.0, 1.0, ratio)
    qf, rf = float(qd), float(small)
    wl, wr = _weights(rf, 1.0)
    expected = np.sqrt(2) * erfinv((qf - wl) / wr)
    rtol = rtol_for(dtype, cond=2.0, f64=_F64_RTOL)
    check(splitnorm.ppf(qd, loc, small, one), expected, dtype, rtol)
    check(splitnorm.isf(qd, loc, one, small), -expected, dtype, rtol)


@pytest.mark.parametrize(
    ("dtype", "y"), per_dtype([15.0, 30.0, 1e3], [40.0, 100.0, 1e3])
)
def test_log_tails_past_underflow(dtype, y):
    """logcdf (lower tail) and logsf (upper tail) stay finite and
    accurate where cdf and sf underflow to zero."""
    loc, sl, sr = 0.0, 0.5, 2.0
    wl, wr = _weights(sl, sr)
    lo, hi, locd, sld, srd = _args(dtype, -y * sl, y * sr, loc, sl, sr)
    assert float(splitnorm.cdf(lo, locd, sld, srd)) == 0.0
    assert float(splitnorm.sf(hi, locd, sld, srd)) == 0.0
    rtol = rtol_for(dtype)
    expected = np.log(2 * wl) + scipy_norm.logcdf(float(lo) / sl)
    check(splitnorm.logcdf(lo, locd, sld, srd), expected, dtype, rtol)
    expected = np.log(2 * wr) + scipy_norm.logsf(float(hi) / sr)
    check(splitnorm.logsf(hi, locd, sld, srd), expected, dtype, rtol)


@pytest.mark.parametrize(
    ("dtype", "y"),
    per_dtype([3.0, 8.0, 13.5, 30.0], [3.0, 20.0, 30.0, 50.0]),
)
def test_log_tail_gradients(dtype, y):
    """d logcdf/dx in the lower tail and d logsf/dx in the upper tail
    are the inverse Mills ratio of the relevant side, divided by its
    scale."""
    loc, sl, sr = 0.0, 0.5, 2.0
    lo, hi, locd, sld, srd = _args(dtype, -y * sl, y * sr, loc, sl, sr)
    rtol = rtol_for(dtype)
    for fn, x, scale, sign in [
        ("logcdf", lo, sl, 1.0),
        ("logsf", hi, sr, -1.0),
    ]:
        yf = abs(float(x)) / scale
        mills = np.exp(scipy_norm.logpdf(yf) - scipy_norm.logsf(yf))
        grad = jax.grad(getattr(splitnorm, fn))(x, locd, sld, srd)
        check(grad, sign * mills / scale, dtype, rtol)


@pytest.mark.parametrize("params", _PARAMS)
@pytest.mark.parametrize("q", [1e-6, 0.1, 0.2, 0.5, 0.8, 0.9, 1 - 1e-6])
def test_quantile_round_trips(q, params, dtype):
    """The reference cdf of ppf(q), and sf of isf(q), recover q."""
    qd, locd, sld, srd = _args(dtype, q, *params)
    qf, locf, slf, srf = (float(v) for v in (qd, locd, sld, srd))
    x = float(splitnorm.ppf(qd, locd, sld, srd))
    # Rounding x perturbs cdf(x) by ~|x| pdf(x) / q ulps.
    cond = abs(x) * np.exp(_ref("logpdf", x, locf, slf, srf)) / qf
    np.testing.assert_allclose(
        _ref("cdf", x, locf, slf, srf), qf, rtol=rtol_for(dtype, cond)
    )
    x = float(splitnorm.isf(qd, locd, sld, srd))
    cond = abs(x) * np.exp(_ref("logpdf", x, locf, slf, srf)) / qf
    np.testing.assert_allclose(
        _ref("sf", x, locf, slf, srf), qf, rtol=rtol_for(dtype, cond)
    )


def test_quantile_endpoints_and_domain(dtype):
    """ppf/isf map the endpoints to -inf/inf, and q outside [0, 1] is
    nan."""
    loc, sl, sr = _args(dtype, 2.0, 0.5, 2.0)
    zero, one = cast(0.0, dtype), cast(1.0, dtype)
    check(splitnorm.ppf(zero, loc, sl, sr), -np.inf, dtype, 0.0)
    check(splitnorm.ppf(one, loc, sl, sr), np.inf, dtype, 0.0)
    check(splitnorm.isf(zero, loc, sl, sr), np.inf, dtype, 0.0)
    check(splitnorm.isf(one, loc, sl, sr), -np.inf, dtype, 0.0)
    for q in [-0.1, 1.1]:
        assert jnp.isnan(splitnorm.ppf(cast(q, dtype), loc, sl, sr))
        assert jnp.isnan(splitnorm.isf(cast(q, dtype), loc, sl, sr))


@pytest.mark.parametrize(
    ("sl", "sr"), [(0.5, 2.0), (1e-4, 1.0), (1.0, 1e-4), (1e-7, 1.0)]
)
def test_mode(sl, sr, dtype):
    """At x = loc: density sqrt(2/pi) / (sl + sr), cdf w_left, sf
    w_right; ppf(w_left) and isf(w_right) return loc. The log-weights
    keep relative accuracy when a weight is close to one (log of the
    rounded quotient would be off by thousands of ulps here)."""
    loc, sl, sr = _args(dtype, 1.5, sl, sr)
    slf, srf = float(sl), float(sr)
    wl, wr = _weights(slf, srf)
    log_wl, log_wr = _log_weights(slf, srf)
    rtol = rtol_for(dtype, f64=_F64_RTOL)
    p0 = np.sqrt(2 / np.pi) / (slf + srf)
    check(splitnorm.pdf(loc, loc, sl, sr), p0, dtype, rtol)
    check(splitnorm.cdf(loc, loc, sl, sr), wl, dtype, rtol)
    check(splitnorm.logcdf(loc, loc, sl, sr), log_wl, dtype, rtol)
    check(splitnorm.sf(loc, loc, sl, sr), wr, dtype, rtol)
    check(splitnorm.logsf(loc, loc, sl, sr), log_wr, dtype, rtol)
    # The weights themselves are rounded, so loc is recovered to
    # ~eps * scale.
    atol = 8 * float(jnp.finfo(dtype).eps) * max(slf, srf)
    w_left, w_right = cast(wl, dtype), cast(wr, dtype)
    assert abs(float(splitnorm.ppf(w_left, loc, sl, sr)) - 1.5) <= atol
    assert abs(float(splitnorm.isf(w_right, loc, sl, sr)) - 1.5) <= atol


@pytest.mark.parametrize("fn", _X_FUNCS)
def test_continuous_at_mode(fn, dtype):
    """Values and x-gradients agree just either side of the mode (the
    density has a continuous first derivative there)."""
    loc, sl, sr = _args(dtype, 0.0, 0.5, 2.0)
    eps = float(jnp.finfo(dtype).eps)
    below, above = _args(dtype, -1e3 * eps, 1e3 * eps)
    f = getattr(splitnorm, fn)
    tol = 1e4 * eps
    np.testing.assert_allclose(
        f(below, loc, sl, sr), f(above, loc, sl, sr), rtol=tol
    )
    g = jax.grad(f)
    np.testing.assert_allclose(
        g(below, loc, sl, sr), g(above, loc, sl, sr), rtol=tol, atol=tol
    )


def test_vectorized_and_broadcasting(dtype):
    """Array inputs broadcast against array loc and scales."""
    x = cast([[-1.0], [0.5], [3.0]], dtype)
    sl = cast([0.5, 1.0], dtype)
    sr = cast([2.0, 1.0], dtype)
    actual = splitnorm.logpdf(x, cast(0.0, dtype), sl, sr)
    assert actual.shape == (3, 2)
    assert actual.dtype == dtype
    for i, j in np.ndindex(3, 2):
        expected = _ref(
            "logpdf", float(x[i, 0]), 0.0, float(sl[j]), float(sr[j])
        )
        check(actual[i, j], expected, dtype, rtol_for(dtype, 10.0))


def test_jit(dtype):
    """Works under jax.jit."""
    for x in [0.3, -0.3]:
        args = _args(dtype, x, 0.1, 0.5, 2.0)
        for fn in _X_FUNCS + _Q_FUNCS:
            if fn in _Q_FUNCS:
                args = (jnp.abs(args[0]), *args[1:])
            jitted = jax.jit(getattr(splitnorm, fn))
            expected = getattr(splitnorm, fn)(*args)
            check(jitted(*args), expected, dtype, rtol_for(dtype))


def test_vmap(dtype):
    """Composes with jax.vmap."""
    x = cast([-1.0, 1.0, 5.0], dtype)
    loc = cast([0.0, 0.5, 1.0], dtype)
    sl = cast([1.0, 2.0, 0.5], dtype)
    sr = cast([0.5, 1.0, 3.0], dtype)
    out = jax.vmap(splitnorm.logcdf)(x, loc, sl, sr)
    assert out.dtype == dtype
    for i in range(3):
        expected = _ref(
            "logcdf", float(x[i]), float(loc[i]), float(sl[i]), float(sr[i])
        )
        check(out[i], expected, dtype, rtol_for(dtype, 10.0))


@pytest.mark.parametrize("fn", _X_FUNCS)
@pytest.mark.parametrize("y", [v for v in _Y_VALUES if abs(v) <= 20])
def test_grad_wrt_x_closed_forms(fn, y, dtype):
    """x-gradients match their closed forms on both sides of the mode.

    logpdf' = -d / s^2 (s the scale on that side), cdf' = pdf,
    sf' = -pdf, logcdf' = pdf / cdf, logsf' = -pdf / sf, and
    pdf' = -d / s^2 pdf."""
    loc, sl, sr = 0.0, 0.5, 2.0
    s = sl if y < 0 else sr
    xd, locd, sld, srd = _args(dtype, y * s, loc, sl, sr)
    xf = float(xd)
    log_p = _ref("logpdf", xf, loc, sl, sr)
    slope = -xf / s**2
    expected = {
        "pdf": slope * np.exp(log_p),
        "logpdf": slope,
        "cdf": np.exp(log_p),
        "logcdf": np.exp(log_p - _ref("logcdf", xf, loc, sl, sr)),
        "sf": -np.exp(log_p),
        "logsf": -np.exp(log_p - _ref("logsf", xf, loc, sl, sr)),
    }[fn]
    # The far-tail log-probability gradients are Mills ratios, which are
    # well conditioned; everything else carries the pdf.
    far_tail = (fn == "logcdf" and y < 0) or (fn == "logsf" and y > 0)
    cond = 0.0 if fn == "logpdf" or far_tail else (xf / s) ** 2
    grad = jax.grad(getattr(splitnorm, fn))(xd, locd, sld, srd)
    assert jnp.isfinite(grad)
    if expected == 0.0:
        assert float(grad) == 0.0
    else:
        check(grad, expected, dtype, rtol_for(dtype, cond))


@pytest.mark.parametrize("y", [-2.0, -0.5, 0.5, 2.0])
def test_grad_wrt_loc_scales(y, dtype):
    """loc and scale gradients of logpdf match closed forms:
    d/dloc = d / s^2, d/ds_side = d^2 / s^3 - 1 / (sl + sr) for the side
    x is on, and -1 / (sl + sr) for the other side."""
    loc, sl, sr = 1.0, 0.5, 2.0
    s = sl if y < 0 else sr
    xd, locd, sld, srd = _args(dtype, loc + y * s, loc, sl, sr)
    d = float(xd) - loc
    grads = jax.grad(splitnorm.logpdf, argnums=(1, 2, 3))(xd, locd, sld, srd)
    own = d**2 / s**3 - 1 / (sl + sr)
    other = -1 / (sl + sr)
    expected = (d / s**2, own, other) if y < 0 else (d / s**2, other, own)
    for g, e in zip(grads, expected, strict=True):
        check(g, e, dtype, rtol_for(dtype, 10.0))


@pytest.mark.parametrize("q", [0.01, 0.15, 0.2, 0.25, 0.7, 0.99])
def test_grad_wrt_q_closed_forms(q, dtype):
    """d ppf/dq = 1 / pdf(ppf(q)) and d isf/dq = -1 / pdf(isf(q)), on
    both sides of the mode (w_left = 0.2 here)."""
    qd, locd, sld, srd = _args(dtype, q, 0.0, 0.5, 2.0)
    for fn, sign in [("ppf", 1.0), ("isf", -1.0)]:
        x = _ref_quantile(fn, float(qd), 0.0, 0.5, 2.0)
        inv_pdf = np.exp(-_ref("logpdf", x, 0.0, 0.5, 2.0))
        s = 0.5 if x < 0 else 2.0
        # 1/pdf(x(q)) has relative condition q |x| / s^2 x'(q) in q.
        cond = float(qd) * abs(x) / s**2 * inv_pdf
        g = jax.grad(getattr(splitnorm, fn))(qd, locd, sld, srd)
        check(g, sign * inv_pdf, dtype, rtol_for(dtype, cond))


@pytest.mark.parametrize("fn", _X_FUNCS)
@pytest.mark.parametrize("x", [-1e3, -30.0, -1.0, 0.0, 1.0, 30.0, 1e3])
def test_grad_finite(fn, x, dtype):
    """Gradients w.r.t. every argument are finite at the mode and far
    into both tails (no NaN leaks from unselected branches)."""
    args = _args(dtype, x, 0.0, 0.5, 2.0)
    grads = jax.grad(getattr(splitnorm, fn), argnums=(0, 1, 2, 3))(*args)
    for g in grads:
        assert g.dtype == dtype
        assert jnp.isfinite(g)


@pytest.mark.parametrize("fn", _X_FUNCS + _Q_FUNCS)
@pytest.mark.parametrize("narrow", ["left", "right"])
@pytest.mark.parametrize("y", [-3.0, 0.0, 3.0])
def test_grad_finite_extreme_ratio(fn, narrow, y, dtype):
    """Gradients stay finite when one weight rounds to exactly one (the
    log1p(-w sf) form would otherwise hit log1p(-1) = -inf). ppf/isf are
    evaluated at q = 0.25, 0.5, 0.75 instead (the cdf at y would round
    to 0 or 1, where the quantile itself is infinite)."""
    r = _EXTREME_RATIO[dtype]
    sl, sr = (r, 1.0) if narrow == "left" else (1.0, r)
    x = y * (sl if y < 0 else sr)
    if fn in _Q_FUNCS:
        x = 0.5 + y / 12
    args = _args(dtype, x, 0.0, sl, sr)
    grads = jax.grad(getattr(splitnorm, fn), argnums=(0, 1, 2, 3))(*args)
    for g in grads:
        assert g.dtype == dtype
        assert jnp.isfinite(g)


@pytest.mark.parametrize("fn", _Q_FUNCS)
@pytest.mark.parametrize("q", [0.1, 0.2, 0.5, 0.6, 0.8, 0.9])
def test_grad_finite_at_quantile_switches(fn, q, dtype):
    """ppf/isf gradients are finite at and between their switches: with
    w_left = 0.2, ppf switches at q = 0.1, 0.2 and 0.6, and isf at
    q = 0.4, 0.8 and 0.9."""
    args = _args(dtype, q, 0.0, 0.5, 2.0)
    grads = jax.grad(getattr(splitnorm, fn), argnums=(0, 1, 2, 3))(*args)
    for g in grads:
        assert g.dtype == dtype
        assert jnp.isfinite(g)


def _dcdf_dscales(x, loc, sl, sr):
    """(dF/d scale_left, dF/d scale_right) at fixed x, in closed form."""
    s = sl + sr
    if x < loc:
        y = (x - loc) / sl
        two_cdf = 2 * scipy_norm.cdf(y)
        return (
            two_cdf * sr / s**2 - sl / s * 2 * scipy_norm.pdf(y) * y / sl,
            -two_cdf * sl / s**2,
        )
    y = (x - loc) / sr
    two_sf = 2 * scipy_norm.sf(y)
    return (
        two_sf * sr / s**2,
        -(two_sf * sl / s**2 + sr / s * 2 * scipy_norm.pdf(y) * y / sr),
    )


@pytest.mark.parametrize("params", _PARAMS[1:3])
@pytest.mark.parametrize("q", [0.05, 0.15, 0.5, 0.85, 0.95])
def test_quantile_grad_wrt_loc_scales(q, params, dtype):
    """ppf/isf gradients w.r.t. loc and the scales follow from implicit
    differentiation of F(x) = q: dx/dloc = 1 and dx/ds = -(dF/ds) / pdf
    (the same for isf, since S = 1 - F)."""
    qd, locd, sld, srd = _args(dtype, q, *params)
    qf, locf, slf, srf = (float(v) for v in (qd, locd, sld, srd))
    for fn in _Q_FUNCS:
        x = _ref_quantile(fn, qf, locf, slf, srf)
        pdf = np.exp(_ref("logpdf", x, locf, slf, srf))
        d_sl, d_sr = _dcdf_dscales(x, locf, slf, srf)
        grads = jax.grad(getattr(splitnorm, fn), argnums=(1, 2, 3))(
            qd, locd, sld, srd
        )
        expected = (1.0, -d_sl / pdf, -d_sr / pdf)
        for g, e in zip(grads, expected, strict=True):
            check(g, e, dtype, rtol_for(dtype, 100.0))


@pytest.mark.parametrize(
    ("dtype", "y"),
    per_dtype([2.0, 8.0, 9.5, 30.0], [2.0, 20.0, 26.5, 50.0]),
)
def test_log_tail_second_derivatives(dtype, y):
    """d^2/dx^2 of logcdf in the lower tail and logsf in the upper tail
    is m (y - m) / s^2, for the inverse Mills ratio m of that side (as in
    halfnorm, with the m - y cancellation costing ~y^2 ulps)."""
    loc, sl, sr = 0.0, 0.5, 2.0
    lo, hi, locd, sld, srd = _args(dtype, -y * sl, y * sr, loc, sl, sr)
    for fn, x, scale in [("logcdf", lo, sl), ("logsf", hi, sr)]:
        yf = abs(float(x)) / scale
        m = np.exp(scipy_norm.logpdf(yf) - scipy_norm.logsf(yf))
        hess = jax.hessian(getattr(splitnorm, fn))(x, locd, sld, srd)
        check(
            hess,
            m * (yf - m) / scale**2,
            dtype,
            rtol_for(dtype, f64=1e-9, f32=2e-5),
        )


@pytest.mark.parametrize("fn", _X_FUNCS)
def test_second_derivative_at_mode(fn, dtype):
    """Second x-derivatives jump at the mode; the value for the side
    x >= loc is returned, e.g. logpdf'' = -1 / scale_right^2, and every
    function's second derivative is finite there."""
    args = _args(dtype, 0.0, 0.0, 0.5, 2.0)
    hess = jax.hessian(getattr(splitnorm, fn))(*args)
    assert jnp.isfinite(hess)
    if fn == "logpdf":
        check(hess, -1 / 4.0, dtype, rtol_for(dtype))


def test_default_and_integer_args():
    """Default loc/scales and integer arguments work, and the result
    follows the floating input (or JAX's default float for all-integer
    input)."""
    out = splitnorm.logpdf(jnp.float32(0.5))
    assert out.dtype == jnp.float32
    check(out, _ref("logpdf", 0.5, 0.0, 1.0, 1.0), jnp.float32, 1e-6)
    out = splitnorm.cdf(1, 0, 1, 2)
    assert jnp.issubdtype(out.dtype, jnp.floating)
    np.testing.assert_allclose(out, _ref("cdf", 1.0, 0.0, 1.0, 2.0))
    out = splitnorm.ppf(jnp.float32(0.3), 0, 1, 2)
    assert out.dtype == jnp.float32


@pytest.mark.parametrize("fn", _X_FUNCS + _Q_FUNCS)
def test_nan_propagates(fn, dtype):
    """nan inputs give nan."""
    args = _args(dtype, np.nan, 0.0, 0.5, 2.0)
    out = getattr(splitnorm, fn)(*args)
    assert out.dtype == dtype
    assert jnp.isnan(out)
