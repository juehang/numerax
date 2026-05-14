"""Test suite for modified Bessel functions of the first kind."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402
from scipy.special import iv as scipy_iv  # noqa: E402
from scipy.special import ive as scipy_ive  # noqa: E402

from numerax.special import iv, ive  # noqa: E402

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
def test_ive_against_scipy(v, z):
    """numerax.ive matches scipy.special.ive within 1e-6 relative."""
    expected = scipy_ive(v, z)
    actual = float(ive(v, z))
    if expected == 0.0:
        # scipy underflow point; numerax may also underflow or return tiny
        assert actual == 0.0 or abs(actual) < 1e-300
    else:
        assert abs(actual - expected) / abs(expected) < 1e-6


def test_ive_vectorized():
    """Array inputs produce per-element correct results."""
    v_arr = jnp.array([0.0, 0.5, 1.0, 5.0, 50.0])
    z_arr = jnp.array([1.0, 10.0, 100.0, 1000.0, 50.0])
    actual = ive(v_arr, z_arr)
    for i, (v, z) in enumerate(
        zip(v_arr.tolist(), z_arr.tolist(), strict=False)
    ):
        assert (
            abs(float(actual[i]) - scipy_ive(v, z)) / abs(scipy_ive(v, z))
            < 1e-6
        )


def test_ive_broadcasting():
    """Scalar v with array z broadcasts."""
    z_arr = jnp.array([1.0, 10.0, 100.0])
    actual = ive(0.5, z_arr)
    for i, z in enumerate(z_arr.tolist()):
        expected = scipy_ive(0.5, z)
        assert abs(float(actual[i]) - expected) / abs(expected) < 1e-6


def test_ive_matches_i0e():
    """ive(0, z) reproduces jax.scipy.special.i0e."""
    z_arr = jnp.array([0.5, 1.0, 5.0, 50.0, 500.0])
    assert jnp.allclose(
        ive(0.0, z_arr), jax.scipy.special.i0e(z_arr), rtol=1e-12
    )


def test_ive_matches_i1e():
    """ive(1, z) reproduces jax.scipy.special.i1e."""
    z_arr = jnp.array([0.5, 1.0, 5.0, 50.0, 500.0])
    assert jnp.allclose(
        ive(1.0, z_arr), jax.scipy.special.i1e(z_arr), rtol=1e-12
    )


def test_ive_half_integer_closed_form():
    r"""ive(0.5, z) == sqrt(2/(pi*z)) * sinh(z) * exp(-z), closed form."""
    z_arr = jnp.array([0.1, 1.0, 5.0, 25.0])
    expected = (
        jnp.sqrt(2.0 / (jnp.pi * z_arr)) * jnp.sinh(z_arr) * jnp.exp(-z_arr)
    )
    actual = ive(0.5, z_arr)
    assert jnp.allclose(actual, expected, rtol=1e-10)


def test_ive_z_zero():
    """ive(v, 0) = 1 for v==0, 0 for v > 0."""
    assert float(ive(0.0, 0.0)) == pytest.approx(1.0)
    for v in [0.5, 1.0, 5.0, 50.0]:
        assert float(ive(v, 0.0)) == 0.0


def test_ive_jit():
    """ive works under jax.jit."""
    jit_ive = jax.jit(ive)
    for v, z in [(0.0, 1.0), (5.0, 100.0), (50.0, 50.0)]:
        actual = float(jit_ive(v, z))
        assert abs(actual - scipy_ive(v, z)) / abs(scipy_ive(v, z)) < 1e-6


def test_ive_vmap():
    """ive composes with jax.vmap."""
    v_arr = jnp.array([0.0, 1.0, 5.0])
    z_arr = jnp.array([1.0, 10.0, 100.0])
    vmapped = jax.vmap(ive)(v_arr, z_arr)
    for i, (v, z) in enumerate(
        zip(v_arr.tolist(), z_arr.tolist(), strict=False)
    ):
        assert (
            abs(float(vmapped[i]) - scipy_ive(v, z)) / abs(scipy_ive(v, z))
            < 1e-6
        )


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
def test_ive_grad_matches_finite_diff(v, z):
    """Approximate dive/dz as a limit via scipy and compare.

    This is the "independent" gradient check: the reference is computed
    by central differences on ``scipy.special.ive``, so it does not
    depend on our DLMF 10.29.1 recurrence being correct. If the JVP
    were derived or coded wrongly, this catches it.
    """
    grad_fn = jax.grad(ive, argnums=1)
    analytic = float(grad_fn(v, z))
    h = 1e-5 * max(1.0, z)
    fd = (scipy_ive(v, z + h) - scipy_ive(v, z - h)) / (2 * h)
    # Tolerance: finite-difference truncation error ~ h^2 |f'''|, plus
    # forward-eval relative error ~1e-6 on each scipy_ive call. 1e-4
    # is generous but defends against the latter when |fd| is small.
    if abs(fd) < 1e-20:
        assert abs(analytic) < 1e-12
    else:
        assert abs(analytic - fd) / abs(fd) < 1e-4


def test_ive_grad_at_v_zero():
    """At v=0 the JVP uses the I_{-1}=I_1 symmetry; sanity-check
    against scipy's finite difference directly (not against the
    recurrence-derived expression).
    """
    z = 1.5
    grad_fn = jax.grad(ive, argnums=1)
    analytic = float(grad_fn(0.0, z))
    h = 1e-5
    fd = (scipy_ive(0.0, z + h) - scipy_ive(0.0, z - h)) / (2 * h)
    assert abs(analytic - fd) / abs(fd) < 1e-4


def test_ive_grad_finite_at_z_zero():
    """grad(ive)(v, 0) returns a finite value (not NaN) for v >= 0."""
    grad_fn = jax.grad(ive, argnums=1)
    for v in [0.0, 0.5, 1.0, 5.0]:
        g = float(grad_fn(v, 0.0))
        assert jnp.isfinite(jnp.asarray(g))


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
def test_iv_against_scipy(v, z):
    """numerax.iv matches scipy.special.iv where scipy is finite."""
    expected = scipy_iv(v, z)
    assert jnp.isfinite(jnp.asarray(expected))
    actual = float(iv(v, z))
    assert abs(actual - expected) / abs(expected) < 1e-6


def test_iv_overflows_to_inf():
    """For z > ~709, iv overflows naturally to +inf (not NaN)."""
    result = float(iv(0.0, 1000.0))
    assert jnp.isinf(jnp.asarray(result))
    assert result > 0


def test_iv_grad_via_chain_rule():
    """iv = ive * exp(z) gradient propagates correctly through autodiff."""
    grad_iv = jax.grad(iv, argnums=1)
    grad_ive = jax.grad(ive, argnums=1)
    v, z = 2.0, 3.0
    expected = float(grad_ive(v, z)) * float(jnp.exp(z)) + float(
        ive(v, z)
    ) * float(jnp.exp(z))
    actual = float(grad_iv(v, z))
    assert abs(actual - expected) / abs(expected) < 1e-10
