r"""
Modified Bessel functions of the first kind for real order.

Provides ``ive(v, z) = exp(-z) * I_v(z)`` (the numerically stable
primitive) and ``iv(v, z) = exp(z) * ive(v, z)`` as a thin wrapper.
``ive`` is the primary implementation because ``exp(z)`` overflows for
``z > ~709`` (float64) while ``ive`` decays like ``1/sqrt(2*pi*z)`` and
stays finite well past that point.

Three regimes are computed in parallel and selected per element with
``jnp.where``:

- power series (DLMF 10.25.2) in log-space for small/moderate ``z``,
- Hankel large-argument asymptotic (DLMF 10.40.1) for large ``z`` and
  small/moderate ``v``,
- Olver uniform asymptotic (DLMF 10.41.3) for large ``v``.

Crossover thresholds are calibrated empirically; see
``scripts/calibrate_bessel.py`` for the calibration procedure and the
``scripts/calibration_plots/`` directory for the relative-error heatmaps.
"""

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike

_SERIES_N_TERMS = 75
_HANKEL_N_TERMS = 15
_OLVER_N_TERMS = 5  # U_0 ... U_4

# Regime crossover thresholds. Calibrated in scripts/calibrate_bessel.py;
# these defaults are starting values that get refined empirically.
_OLVER_V_THRESHOLD = 50.0
_SERIES_Z_FLOOR = 20.0
_SERIES_Z_COEFF = 2.0


def _series_z_threshold(v):
    """Power-series upper bound on z, as a function of v."""
    return jnp.maximum(
        _SERIES_Z_FLOOR, _SERIES_Z_COEFF * jnp.sqrt(v + 1.0)
    )


def _ive_series(v: ArrayLike, z: ArrayLike) -> ArrayLike:
    r"""Power series for ``ive(v, z)`` in log-space (DLMF 10.25.2).

    Computes
    $$ive(v, z) = e^{-z} (z/2)^v / \Gamma(v+1) \sum_{k=0}^{N-1} r_k$$
    with $r_0 = 1$ and $r_{k+1} = r_k \cdot (z^2/4) / [(k+1)(v+k+1)]$.
    The prefactor is taken in log-space to avoid catastrophic underflow
    (e.g. $(z/2)^v / \Gamma(v+1)$ for $v=200, z=1$).

    Off-regime inputs ``z <= 0`` are replaced with a dummy 1.0; the
    caller is responsible for masking the result.
    """
    v = jnp.asarray(v)
    z = jnp.asarray(z)
    out_shape = jnp.broadcast_shapes(v.shape, z.shape)
    v = jnp.broadcast_to(v, out_shape)
    z = jnp.broadcast_to(z, out_shape)

    z_safe = jnp.where(z > 0, z, 1.0)
    half_z_sq = z_safe * z_safe / 4.0

    ks = jnp.arange(1, _SERIES_N_TERMS, dtype=z_safe.dtype)
    log_half_z_sq = jnp.log(half_z_sq)[..., None]
    log_i = jnp.log(ks)
    log_v_plus_i = jnp.log(v[..., None] + ks)
    log_ratios = log_half_z_sq - log_i - log_v_plus_i

    log_r_tail = jnp.cumsum(log_ratios, axis=-1)
    zero = jnp.zeros(out_shape + (1,), dtype=log_r_tail.dtype)
    log_r = jnp.concatenate([zero, log_r_tail], axis=-1)
    log_S = jax.scipy.special.logsumexp(log_r, axis=-1)

    log_ive = (
        -z_safe
        + v * jnp.log(z_safe / 2.0)
        - jax.scipy.special.gammaln(v + 1.0)
        + log_S
    )
    return jnp.exp(log_ive)


def _ive_hankel(v: ArrayLike, z: ArrayLike) -> ArrayLike:
    r"""Hankel large-argument asymptotic for ``ive(v, z)`` (DLMF 10.40.1).

    $$ive(v, z) \sim \frac{1}{\sqrt{2\pi z}}
       \sum_{k=0}^{K-1} \frac{(-1)^k a_k(v)}{z^k}$$
    with $a_0 = 1$ and $a_k = a_{k-1}(4v^2 - (2k-1)^2)/(8k)$. Truncated
    at fixed $K$; valid in the large-$z$, moderate-$v$ region.
    """
    v = jnp.asarray(v)
    z = jnp.asarray(z)
    out_shape = jnp.broadcast_shapes(v.shape, z.shape)
    v = jnp.broadcast_to(v, out_shape)
    z = jnp.broadcast_to(z, out_shape)

    z_safe = jnp.where(z > 0, z, 1.0)
    four_vsq = 4.0 * v * v

    ks = jnp.arange(1, _HANKEL_N_TERMS + 1, dtype=z_safe.dtype)
    numerators = four_vsq[..., None] - (2.0 * ks - 1.0) ** 2
    signed_factors = -numerators / (8.0 * ks * z_safe[..., None])
    cum_factors = jnp.cumprod(signed_factors, axis=-1)
    ones = jnp.ones(out_shape + (1,), dtype=cum_factors.dtype)
    terms = jnp.concatenate([ones, cum_factors], axis=-1)
    series_sum = jnp.sum(terms, axis=-1)
    return series_sum / jnp.sqrt(2.0 * jnp.pi * z_safe)


# --- Olver U_k(p) coefficients (DLMF 10.41.9 for k=1..3; k=4 derived
# from the DLMF 10.41.10 recurrence
#   U_{k+1}(p) = (1/2) p^2 (1-p^2) U_k'(p)
#              + (1/8) \int_0^p (1 - 5 t^2) U_k(t) dt.
# Polynomials presented over a common denominator per row; evaluated
# via Horner in p^2 below.
#
# U_0(p) = 1
# U_1(p) = (3 p - 5 p^3) / 24
# U_2(p) = (81 p^2 - 462 p^4 + 385 p^6) / 1152
# U_3(p) = (30375 p^3 - 369603 p^5 + 765765 p^7 - 425425 p^9) / 414720
# U_4(p) = (4465125 p^4 - 94121676 p^6 + 349922430 p^8
#          - 446185740 p^10 + 185910725 p^12) / 39813120


def _olver_uk_series(p):
    """Stack U_0(p) ... U_4(p) along the last axis (no 1/v^k factor)."""
    p2 = p * p
    u0 = jnp.ones_like(p)
    u1 = p * (3.0 - 5.0 * p2) / 24.0
    u2 = p2 * (81.0 + p2 * (-462.0 + p2 * 385.0)) / 1152.0
    u3 = (
        p
        * p2
        * (
            30375.0
            + p2 * (-369603.0 + p2 * (765765.0 + p2 * -425425.0))
        )
        / 414720.0
    )
    u4 = (
        p2
        * p2
        * (
            4465125.0
            + p2
            * (
                -94121676.0
                + p2
                * (
                    349922430.0
                    + p2 * (-446185740.0 + p2 * 185910725.0)
                )
            )
        )
        / 39813120.0
    )
    return jnp.stack([u0, u1, u2, u3, u4], axis=-1)


def _ive_olver(v: ArrayLike, z: ArrayLike) -> ArrayLike:
    r"""Olver uniform asymptotic expansion for ``ive(v, z)`` (DLMF 10.41.3).

    Let $x = z/v$, $p = 1/\sqrt{1+x^2}$,
    $\eta = \sqrt{1+x^2} + \log(x/(1+\sqrt{1+x^2}))$. Then
    $$ive(v, z) \sim \frac{1}{\sqrt{2\pi v}}
       \frac{e^{v\eta - z}}{(1+x^2)^{1/4}}
       \sum_{k=0}^{K-1} \frac{U_k(p)}{v^k}.$$

    Off-regime inputs (``v`` small) are guarded; caller masks.
    """
    v = jnp.asarray(v)
    z = jnp.asarray(z)
    out_shape = jnp.broadcast_shapes(v.shape, z.shape)
    v = jnp.broadcast_to(v, out_shape)
    z = jnp.broadcast_to(z, out_shape)

    v_safe = jnp.where(v > 1.0, v, _OLVER_V_THRESHOLD)
    z_safe = jnp.where(z > 0, z, 1.0)

    x = z_safe / v_safe
    one_plus_x2 = 1.0 + x * x
    sqrt_one_plus_x2 = jnp.sqrt(one_plus_x2)
    p = 1.0 / sqrt_one_plus_x2
    eta = sqrt_one_plus_x2 + jnp.log(x / (1.0 + sqrt_one_plus_x2))

    prefactor = jnp.exp(v_safe * eta - z_safe) / (
        jnp.sqrt(2.0 * jnp.pi * v_safe) * one_plus_x2**0.25
    )

    uks = _olver_uk_series(p)
    ks = jnp.arange(_OLVER_N_TERMS, dtype=v_safe.dtype)
    inv_v_powers = (1.0 / v_safe)[..., None] ** ks
    series = jnp.sum(uks * inv_v_powers, axis=-1)

    return prefactor * series
