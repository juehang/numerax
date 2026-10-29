"""Test suite for modified Bessel functions of the first kind."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import iv as scipy_iv
from scipy.special import ive as scipy_ive

from numerax.special import iv, ive

from ._precision import cast, check

# Relative accuracy of ive: ~1e-6 by design in float64; in float32 the
# error is up to ~6e-6 (~50 ulps).
_IVE_RTOL = {jnp.float32: 1e-5, jnp.float64: 1e-6}

# Reference v values: a mix of zero, simple half-integers (Rice /
# non-central chi^2 use case), integers, and arbitrary irrationals.
# The arbitrary values stress code paths that simple rationals can hit
# by accident (e.g. an integer-only fast path).
_V_VALUES = [
    0.0,
    0.25,
    0.5,
    0.75,
    1.0,
    1.234,
    2.5,
    5.0,
    7.71828,
    9.99,
    10.0,
    10.01,
    19.7,
    50.0,
    123.456,
    200.0,
]

# z values spanning all three regimes including the boundary at z=30.
_Z_VALUES = [1e-2, 1.0, 10.0, 29.9, 30.0, 30.1, 100.0, 1000.0, 10000.0]

_GRID_VZ = [(v, z) for v in _V_VALUES for z in _Z_VALUES]


@pytest.mark.parametrize(("v", "z"), _GRID_VZ)
def test_ive_against_scipy(v, z, dtype):
    """numerax.ive matches scipy.special.ive (1e-6 relative in float64,
    1e-5 in float32). Values below the dtype's range may underflow."""
    vd, zd = cast(v, dtype), cast(z, dtype)
    expected = scipy_ive(float(vd), float(zd))
    check(ive(vd, zd), expected, dtype, _IVE_RTOL[dtype])


def test_ive_vectorized(dtype):
    """Array inputs produce per-element correct results."""
    v_arr = cast([0.0, 0.5, 1.0, 5.0, 50.0], dtype)
    z_arr = cast([1.0, 10.0, 100.0, 1000.0, 50.0], dtype)
    actual = ive(v_arr, z_arr)
    for i, (v, z) in enumerate(
        zip(v_arr.tolist(), z_arr.tolist(), strict=False)
    ):
        check(actual[i], scipy_ive(v, z), dtype, _IVE_RTOL[dtype])


def test_ive_broadcasting(dtype):
    """Scalar v with array z broadcasts."""
    z_arr = cast([1.0, 10.0, 100.0], dtype)
    actual = ive(cast(0.5, dtype), z_arr)
    for i, z in enumerate(z_arr.tolist()):
        check(actual[i], scipy_ive(0.5, z), dtype, _IVE_RTOL[dtype])


def test_ive_matches_i0e(dtype):
    """ive(0, z) reproduces jax.scipy.special.i0e."""
    z_arr = cast([0.5, 1.0, 5.0, 50.0, 500.0], dtype)
    actual = ive(cast(0.0, dtype), z_arr)
    assert actual.dtype == dtype
    rtol = 1e-12 if dtype == jnp.float64 else 1e-6
    assert jnp.allclose(actual, jax.scipy.special.i0e(z_arr), rtol=rtol)


def test_ive_matches_i1e(dtype):
    """ive(1, z) reproduces jax.scipy.special.i1e."""
    z_arr = cast([0.5, 1.0, 5.0, 50.0, 500.0], dtype)
    actual = ive(cast(1.0, dtype), z_arr)
    assert actual.dtype == dtype
    rtol = 1e-12 if dtype == jnp.float64 else 1e-6
    assert jnp.allclose(actual, jax.scipy.special.i1e(z_arr), rtol=rtol)


def test_ive_half_integer_closed_form(dtype):
    r"""ive(0.5, z) == sqrt(2/(pi*z)) * sinh(z) * exp(-z), closed form."""
    z_arr = cast([0.1, 1.0, 5.0, 25.0], dtype)
    z64 = np.asarray(z_arr, np.float64)
    expected = np.sqrt(2.0 / (np.pi * z64)) * np.sinh(z64) * np.exp(-z64)
    actual = ive(cast(0.5, dtype), z_arr)
    assert actual.dtype == dtype
    rtol = 1e-10 if dtype == jnp.float64 else _IVE_RTOL[dtype]
    assert jnp.allclose(actual, expected, rtol=rtol)


def test_ive_z_zero(dtype):
    """ive(v, 0) = 1 for v==0, 0 for v > 0."""
    zero = cast(0.0, dtype)
    assert float(ive(zero, zero)) == pytest.approx(1.0)
    for v in [0.5, 1.0, 5.0, 50.0]:
        out = ive(cast(v, dtype), zero)
        assert out.dtype == dtype
        assert float(out) == 0.0


def test_ive_jit(dtype):
    """ive works under jax.jit."""
    jit_ive = jax.jit(ive)
    for v, z in [(0.0, 1.0), (5.0, 100.0), (50.0, 50.0)]:
        actual = jit_ive(cast(v, dtype), cast(z, dtype))
        check(actual, scipy_ive(v, z), dtype, _IVE_RTOL[dtype])


def test_ive_vmap(dtype):
    """ive composes with jax.vmap."""
    v_arr = cast([0.0, 1.0, 5.0], dtype)
    z_arr = cast([1.0, 10.0, 100.0], dtype)
    vmapped = jax.vmap(ive)(v_arr, z_arr)
    for i, (v, z) in enumerate(
        zip(v_arr.tolist(), z_arr.tolist(), strict=False)
    ):
        check(vmapped[i], scipy_ive(v, z), dtype, _IVE_RTOL[dtype])


@pytest.mark.parametrize(
    ("v", "z"),
    [
        (v, z)
        for v in _V_VALUES
        # Skip z=0.01 (z+h crosses zero for any reasonable h on this scale)
        # and the largest z (finite-diff round-off dominates).
        for z in [0.5, 1.0, 5.0, 25.0, 50.0, 200.0, 1000.0]
    ],
)
def test_ive_grad_matches_finite_diff(v, z, dtype):
    """Approximate dive/dz as a limit via scipy and compare.

    This is the "independent" gradient check: the reference is computed
    by central differences on ``scipy.special.ive``, so it does not
    depend on our DLMF 10.29.1 recurrence being correct. If the JVP
    were derived or coded wrongly, this catches it.
    """
    vd, zd = cast(v, dtype), cast(z, dtype)
    v, z = float(vd), float(zd)
    grad = jax.grad(ive, argnums=1)(vd, zd)
    assert grad.dtype == dtype
    analytic = float(grad)
    h = 1e-5 * max(1.0, z)
    fd = (scipy_ive(v, z + h) - scipy_ive(v, z - h)) / (2 * h)
    # Tolerance: finite-difference truncation error ~ h^2 |f'''|, plus
    # forward-eval relative error ~1e-6 on each scipy_ive call. 1e-4
    # is generous but defends against the latter when |fd| is small.
    # In float32 the JVP's (ive_{v-1} + ive_{v+1})/2 - ive_v cancels
    # whenever |ive'| << |ive| (|ive / ive'| ~ 2z at large z), so it is
    # also allowed an error of the float32 ive tolerance times |ive|.
    if abs(fd) < 1e-20:
        assert abs(analytic) < 1e-12
    else:
        tol = 1e-4 * abs(fd)
        if dtype == jnp.float32:
            tol += _IVE_RTOL[dtype] * scipy_ive(v, z)
        assert abs(analytic - fd) <= tol


@pytest.mark.parametrize("v", [-0.75, -0.5, -0.25])
@pytest.mark.parametrize("z", [0.3, 1.5, 5.0, 40.0])
def test_ive_grad_negative_order(v, z, dtype):
    """Gradient w.r.t. z is finite and correct for orders v in (-1, 0),
    which arise e.g. in the non-central chi^2 density with df < 2."""
    vd, zd = cast(v, dtype), cast(z, dtype)
    v, z = float(vd), float(zd)
    grad = jax.grad(ive, argnums=1)(vd, zd)
    assert grad.dtype == dtype
    analytic = float(grad)
    assert jnp.isfinite(jnp.asarray(analytic))
    h = 1e-6
    fd = (scipy_ive(v, z + h) - scipy_ive(v, z - h)) / (2 * h)
    assert abs(analytic - fd) / abs(fd) < 1e-4


def test_ive_grad_at_v_zero(dtype):
    """At v=0 the JVP uses the I_{-1}=I_1 symmetry; sanity-check
    against scipy's finite difference directly (not against the
    recurrence-derived expression).
    """
    z = 1.5
    grad_fn = jax.grad(ive, argnums=1)
    analytic = float(grad_fn(cast(0.0, dtype), cast(z, dtype)))
    h = 1e-5
    fd = (scipy_ive(0.0, z + h) - scipy_ive(0.0, z - h)) / (2 * h)
    assert abs(analytic - fd) / abs(fd) < 1e-4


def test_ive_grad_finite_at_z_zero(dtype):
    """grad(ive)(v, 0) returns a finite value (not NaN) for v >= 0."""
    grad_fn = jax.grad(ive, argnums=1)
    for v in [0.0, 0.5, 1.0, 5.0]:
        g = grad_fn(cast(v, dtype), cast(0.0, dtype))
        assert g.dtype == dtype
        assert jnp.isfinite(g)


def test_ive_grad_at_z_zero_v_one(dtype):
    """At z=0 the only nonzero finite derivative is at v=1, where
    d/dz ive(1, z)|_0 = 1/2. This pins the z=0 boundary behaviour that
    the symmetric recurrence provides (a forward v/z form cannot)."""
    g = float(jax.grad(ive, argnums=1)(cast(1.0, dtype), cast(0.0, dtype)))
    assert abs(g - 0.5) < 1e-12


@pytest.mark.parametrize(
    ("v", "z"),
    [
        (0.0, 1.0),
        (0.5, 5.0),
        (5.0, 5.0),
        (10.0, 100.0),
        (0.5, 50.0),
    ],
)
def test_iv_against_scipy(v, z, dtype):
    """numerax.iv matches scipy.special.iv where scipy is finite (and
    overflows to inf where the value exceeds the dtype's range)."""
    vd, zd = cast(v, dtype), cast(z, dtype)
    expected = scipy_iv(float(vd), float(zd))
    assert jnp.isfinite(jnp.asarray(expected))
    check(iv(vd, zd), expected, dtype, _IVE_RTOL[dtype])


def test_iv_overflows_to_inf(dtype):
    """For z > ~709 (~88 in float32), iv overflows naturally to +inf
    (not NaN)."""
    result = float(iv(cast(0.0, dtype), cast(1000.0, dtype)))
    assert jnp.isinf(jnp.asarray(result))
    assert result > 0


def test_iv_grad_via_chain_rule(dtype):
    """iv = ive * exp(z) gradient propagates correctly through autodiff."""
    grad_iv = jax.grad(iv, argnums=1)
    grad_ive = jax.grad(ive, argnums=1)
    v, z = cast(2.0, dtype), cast(3.0, dtype)
    expected = float(grad_ive(v, z)) * float(jnp.exp(z)) + float(
        ive(v, z)
    ) * float(jnp.exp(z))
    actual = float(grad_iv(v, z))
    rtol = 1e-10 if dtype == jnp.float64 else 1e-6
    assert abs(actual - expected) / abs(expected) < rtol


def test_ive_grad_wrt_order_raises(dtype):
    """Differentiating ive w.r.t. the order v is unsupported and must
    raise (rather than silently returning a wrong gradient)."""
    with pytest.raises(TypeError):
        jax.grad(ive, argnums=0)(cast(0.5, dtype), cast(1.5, dtype))


def test_iv_grad_wrt_order_raises(dtype):
    """The order-derivative guard propagates through iv = ive * exp(z)."""
    with pytest.raises(TypeError):
        jax.grad(iv, argnums=0)(cast(0.5, dtype), cast(1.5, dtype))


def test_ive_grad_wrt_order_raises_when_order_depends_on_input(dtype):
    """Order tangent reaching ive through a composition also raises;
    differentiating only through z (argnums for the z-feeding input)
    still works."""

    def order_path(a, z):
        # a feeds the order argument -> must raise
        return ive(a / 2.0 - 1.0, z)

    a, z = cast(3.0, dtype), cast(1.5, dtype)
    with pytest.raises(TypeError):
        jax.grad(order_path, argnums=0)(a, z)

    # Differentiating the same composition w.r.t. z is fine.
    g = float(jax.grad(order_path, argnums=1)(a, z))
    assert jnp.isfinite(jnp.asarray(g))
