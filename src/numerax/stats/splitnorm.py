"""
Split-normal distribution functions.

This module provides the density, distribution, and quantile functions of
the split-normal (two-piece normal) distribution: a normal distribution
with mode ``loc`` whose left and right halves have different scales. It is
commonly used to describe asymmetric uncertainties. JAX and scipy do not
provide it. The signatures follow the ``scipy.stats`` conventions, with
the single ``scale`` replaced by ``scale_left`` and ``scale_right``. The
distribution functions are built on [`halfnorm`][numerax.stats.halfnorm],
every function is evaluated in a form that stays accurate in both tails in
float32 and float64, and all functions are compatible with JAX
transformations (JIT, grad, vmap).
"""

import math

import jax.numpy as jnp
from jax.scipy.special import erfinv, ndtri
from jaxtyping import ArrayLike

from . import halfnorm

# Python floats (not jnp arrays) so they adopt the dtype of the inputs at
# call time, whatever the x64 setting was when this module was imported.
_SQRT2 = math.sqrt(2.0)
_HALF_LOG_2_OVER_PI = 0.5 * math.log(2.0 / math.pi)


def _weights(scale_left, scale_right):
    """Probability mass on each side of the mode."""
    scale_left = jnp.asarray(scale_left)
    scale_right = jnp.asarray(scale_right)
    total = scale_left + scale_right
    return scale_left / total, scale_right / total


def _log_weights(scale_left, scale_right):
    """Logs of the probability mass on each side of the mode.

    log(w_left) = -log1p(scale_right / scale_left) keeps relative accuracy
    when w_left is close to one, where log of the rounded quotient would
    not (and likewise for w_right)."""
    scale_left = jnp.asarray(scale_left)
    scale_right = jnp.asarray(scale_right)
    return (
        -jnp.log1p(scale_right / scale_left),
        -jnp.log1p(scale_left / scale_right),
    )


def _split(x, loc, scale_left, scale_right):
    """Offset from the mode and the probability mass on each side."""
    w_left, w_right = _weights(scale_left, scale_right)
    return jnp.asarray(x) - jnp.asarray(loc), w_left, w_right


def _log1m_mixture(w_near, w_far, cdf_far, sf_far):
    # log(w_near + w_far * cdf_far) = log1p(-w_far * sf_far), the log of the
    # probability on the near side of x when x is past the mode on the far
    # side. log1p keeps relative accuracy while the result is near zero;
    # the direct sum (no cancellation) does once log1p would be
    # ill-conditioned. Safe substitutes keep each form finite where it is
    # not selected: log1p(-a) hits log1p(-1) = -inf when w_far rounds to
    # one, which would leak a NaN gradient through the selection.
    a = w_far * sf_far
    use_log1p = a < 0.5
    a = jnp.where(use_log1p, a, 0.25)
    total = jnp.where(use_log1p, 1.0, w_near + w_far * cdf_far)
    return jnp.where(use_log1p, jnp.log1p(-a), jnp.log(total))


def logpdf(
    x: ArrayLike,
    loc: ArrayLike = 0,
    scale_left: ArrayLike = 1,
    scale_right: ArrayLike = 1,
) -> ArrayLike:
    r"""
    Split-normal log probability density function.

    ## Overview

    Computes the natural logarithm of the probability density function of
    the split-normal distribution, a normal distribution with mode
    ``loc`` whose two halves have separate scales.

    ## Mathematical Background

    With mode $\mu = \text{loc}$, left scale $\sigma_L$ and right scale
    $\sigma_R$, the density is

    $$f(x) = \sqrt{\frac{2}{\pi}}\, \frac{1}{\sigma_L + \sigma_R}
    \begin{cases}
    e^{-(x - \mu)^2 / 2\sigma_L^2}, & x < \mu, \\
    e^{-(x - \mu)^2 / 2\sigma_R^2}, & x \ge \mu,
    \end{cases}$$

    which is continuous (with continuous derivative) at the mode. It
    reduces to $N(\mu, \sigma^2)$ when $\sigma_L = \sigma_R = \sigma$.

    Equivalently, the split normal is a mixture of two half-normals with
    disjoint supports: with probability $w_L = \sigma_L / (\sigma_L +
    \sigma_R)$, $X = \mu - \sigma_L |Z|$, and with probability $w_R =
    \sigma_R / (\sigma_L + \sigma_R)$, $X = \mu + \sigma_R |Z|$. Writing
    $F_h$ and $S_h$ for the standard half-normal CDF and survival
    function, this gives

    $$F(x) = \begin{cases}
    w_L\, S_h\!\left(\tfrac{\mu - x}{\sigma_L}\right), & x < \mu, \\
    w_L + w_R\, F_h\!\left(\tfrac{x - \mu}{\sigma_R}\right), & x \ge \mu,
    \end{cases}$$

    and the mirror image for $S(x) = 1 - F(x)$. The quantile function is
    $\mu - \sigma_L F_h^{-1}\!\left(\tfrac{w_L - q}{w_L}\right)$ for
    $q < w_L$ and $\mu + \sigma_R F_h^{-1}\!\left(\tfrac{q - w_L}{w_R}
    \right)$ otherwise, where $F_h^{-1}(p) = \sqrt{2}\,\text{erfinv}(p)
    = -\Phi^{-1}\!\left(\tfrac{1 - p}{2}\right)$.

    Every function in this module evaluates these in a form that is
    accurate in the relevant regime: tail probabilities are computed
    directly from the half-normal survival function (never as $1 -
    \text{something}$); ``logcdf`` and ``logsf`` use
    [`halfnorm.logsf`][numerax.stats.halfnorm.logsf] on the side of the
    mode where they decay, so they stay accurate past where the
    probabilities underflow, and switch between $\text{log1p}$ and a
    direct sum on the other side; the log-weights are evaluated as
    $\log w_L = -\text{log1p}(\sigma_R/\sigma_L)$, accurate even when
    $w_L$ is close to one; and ``ppf`` and ``isf``, like
    [`halfnorm.ppf`][numerax.stats.halfnorm.ppf], use ``erfinv`` within
    the central half of each piece and $\Phi^{-1}$ of the tail
    probability beyond it, so no regime loses precision to forming
    $1 - q$.

    ## Args

    - **x**: Quantile values. Scalar or array.
    - **loc**: Location of the mode (default: 0). Scalar or array.
    - **scale_left**: Scale below the mode (must be positive, default:
      1). Scalar or array.
    - **scale_right**: Scale above the mode (must be positive, default:
      1). Scalar or array.

    ## Returns

    Log probability density values. Shape follows JAX broadcasting rules.

    ## Example

    ```python
    import jax
    import jax.numpy as jnp
    import numerax

    # A measurement of 3.0 with uncertainties -0.5 / +1.2
    lp = numerax.stats.splitnorm.logpdf(3.4, loc=3.0, scale_left=0.5,
                                        scale_right=1.2)

    # Vectorized
    x_vals = jnp.array([-1.0, 0.0, 1.0, 5.0])
    lps = numerax.stats.splitnorm.logpdf(x_vals, 0.0, 1.0, 2.0)

    # Differentiable in x, loc, and both scales
    grad_fn = jax.grad(numerax.stats.splitnorm.logpdf, argnums=(1, 2, 3))
    d_loc, d_left, d_right = grad_fn(1.0, 0.0, 1.0, 2.0)
    ```

    ## Notes

    - **Differentiable**: w.r.t. ``x``, ``loc``, ``scale_left`` and
      ``scale_right`` (and ``q`` for ``ppf``/``isf``). Branch selections
      use safe substitutes so gradients stay NaN-free at finite points,
      including for extreme scale ratios. The density and its first
      derivative in ``x`` are continuous at the mode, and every function
      is continuous in all its arguments there, so first derivatives at
      the mode are two-sided. Second and higher derivatives in ``x`` jump
      at the mode; there the values for the side $x \ge \mu$ are
      returned.
    - **Precision**: results follow the input dtype. Accuracy is close to
      machine precision in both float32 and float64; far in either tail
      ``pdf``, ``cdf`` and ``sf`` are limited by the conditioning of
      $e^{-y^2/2}$, about $y^2$ ulps, as in
      [`halfnorm`][numerax.stats.halfnorm]. Near $q = w_L$ (``ppf``) or
      $q = w_R$ (``isf``), the quantile is limited by the rounding of the
      weight itself.
    - **Broadcasting**: Supports JAX array broadcasting for all
      parameters.
    - **Support**: the whole real line. ``nan`` inputs propagate to
      ``nan``.
    """
    d = jnp.asarray(x) - jnp.asarray(loc)
    scale_left = jnp.asarray(scale_left)
    scale_right = jnp.asarray(scale_right)
    scale = jnp.where(d < 0, scale_left, scale_right)
    y = d / scale
    return _HALF_LOG_2_OVER_PI - 0.5 * y**2 - jnp.log(scale_left + scale_right)


def pdf(
    x: ArrayLike,
    loc: ArrayLike = 0,
    scale_left: ArrayLike = 1,
    scale_right: ArrayLike = 1,
) -> ArrayLike:
    r"""
    Split-normal probability density function.

    Computes $\sqrt{2/\pi}\, e^{-y^2/2} / (\sigma_L + \sigma_R)$ for
    $y = (x - \text{loc})/\sigma$, where $\sigma$ is ``scale_left`` below
    the mode and ``scale_right`` above it. This is ``exp(logpdf(...))``;
    see [`logpdf`][numerax.stats.splitnorm.logpdf] for the full
    mathematical background and conventions.

    ## Args

    - **x**: Quantile values. Scalar or array.
    - **loc**: Location of the mode (default: 0). Scalar or array.
    - **scale_left**: Scale below the mode (must be positive, default:
      1). Scalar or array.
    - **scale_right**: Scale above the mode (must be positive, default:
      1). Scalar or array.

    ## Returns

    Probability density values. Shape follows JAX broadcasting rules.
    """
    return jnp.exp(logpdf(x, loc, scale_left, scale_right))


def cdf(
    x: ArrayLike,
    loc: ArrayLike = 0,
    scale_left: ArrayLike = 1,
    scale_right: ArrayLike = 1,
) -> ArrayLike:
    r"""
    Split-normal cumulative distribution function.

    Computes $w_L\, S_h((\mu - x)/\sigma_L)$ below the mode, which keeps
    relative accuracy in the lower tail, and $w_L + w_R\, F_h((x -
    \mu)/\sigma_R)$ above it, a sum of non-negative terms. See
    [`logpdf`][numerax.stats.splitnorm.logpdf] for the notation and
    mathematical background.

    ## Args

    - **x**: Quantile values. Scalar or array.
    - **loc**: Location of the mode (default: 0). Scalar or array.
    - **scale_left**: Scale below the mode (must be positive, default:
      1). Scalar or array.
    - **scale_right**: Scale above the mode (must be positive, default:
      1). Scalar or array.

    ## Returns

    Cumulative probabilities. Shape follows JAX broadcasting rules.
    """
    d, w_left, w_right = _split(x, loc, scale_left, scale_right)
    # Each halfnorm call is below its support (and returns a constant) on
    # the side where it is not selected, so no extra substitutes are
    # needed.
    return jnp.where(
        d < 0,
        w_left * halfnorm.sf(-d, 0.0, scale_left),
        w_left + w_right * halfnorm.cdf(d, 0.0, scale_right),
    )


def logcdf(
    x: ArrayLike,
    loc: ArrayLike = 0,
    scale_left: ArrayLike = 1,
    scale_right: ArrayLike = 1,
) -> ArrayLike:
    r"""
    Split-normal log cumulative distribution function.

    Below the mode, computes $\log w_L + \log S_h((\mu - x)/\sigma_L)$
    using [`halfnorm.logsf`][numerax.stats.halfnorm.logsf], which stays
    accurate far past where the CDF underflows. Above the mode, computes
    $\text{log1p}(-w_R\, S_h((x - \mu)/\sigma_R))$ while the CDF exceeds
    $1/2$ and $\log(w_L + w_R\, F_h((x - \mu)/\sigma_R))$ otherwise, so
    that neither regime loses precision. See
    [`logpdf`][numerax.stats.splitnorm.logpdf] for the notation and
    mathematical background.

    ## Args

    - **x**: Quantile values. Scalar or array.
    - **loc**: Location of the mode (default: 0). Scalar or array.
    - **scale_left**: Scale below the mode (must be positive, default:
      1). Scalar or array.
    - **scale_right**: Scale above the mode (must be positive, default:
      1). Scalar or array.

    ## Returns

    Log cumulative probabilities. Shape follows JAX broadcasting rules.
    """
    d, w_left, w_right = _split(x, loc, scale_left, scale_right)
    log_w_left, _ = _log_weights(scale_left, scale_right)
    return jnp.where(
        d < 0,
        log_w_left + halfnorm.logsf(-d, 0.0, scale_left),
        _log1m_mixture(
            w_left,
            w_right,
            halfnorm.cdf(d, 0.0, scale_right),
            halfnorm.sf(d, 0.0, scale_right),
        ),
    )


def sf(
    x: ArrayLike,
    loc: ArrayLike = 0,
    scale_left: ArrayLike = 1,
    scale_right: ArrayLike = 1,
) -> ArrayLike:
    r"""
    Split-normal survival function ($1 - \text{cdf}$).

    Computes $w_R\, S_h((x - \mu)/\sigma_R)$ above the mode, which keeps
    relative accuracy in the upper tail, and $w_R + w_L\, F_h((\mu -
    x)/\sigma_L)$ below it. See
    [`logpdf`][numerax.stats.splitnorm.logpdf] for the notation and
    mathematical background.

    ## Args

    - **x**: Quantile values. Scalar or array.
    - **loc**: Location of the mode (default: 0). Scalar or array.
    - **scale_left**: Scale below the mode (must be positive, default:
      1). Scalar or array.
    - **scale_right**: Scale above the mode (must be positive, default:
      1). Scalar or array.

    ## Returns

    Survival probabilities. Shape follows JAX broadcasting rules.
    """
    d, w_left, w_right = _split(x, loc, scale_left, scale_right)
    return jnp.where(
        d < 0,
        w_right + w_left * halfnorm.cdf(-d, 0.0, scale_left),
        w_right * halfnorm.sf(d, 0.0, scale_right),
    )


def logsf(
    x: ArrayLike,
    loc: ArrayLike = 0,
    scale_left: ArrayLike = 1,
    scale_right: ArrayLike = 1,
) -> ArrayLike:
    r"""
    Split-normal log survival function.

    The mirror image of [`logcdf`][numerax.stats.splitnorm.logcdf]: above
    the mode, computes $\log w_R + \log S_h((x - \mu)/\sigma_R)$ using
    [`halfnorm.logsf`][numerax.stats.halfnorm.logsf]; below it, computes
    $\text{log1p}(-w_L\, S_h((\mu - x)/\sigma_L))$ while the survival
    function exceeds $1/2$ and $\log(w_R + w_L\, F_h((\mu - x)/\sigma_L))$
    otherwise. See [`logpdf`][numerax.stats.splitnorm.logpdf] for the
    notation and mathematical background.

    ## Args

    - **x**: Quantile values. Scalar or array.
    - **loc**: Location of the mode (default: 0). Scalar or array.
    - **scale_left**: Scale below the mode (must be positive, default:
      1). Scalar or array.
    - **scale_right**: Scale above the mode (must be positive, default:
      1). Scalar or array.

    ## Returns

    Log survival probabilities. Shape follows JAX broadcasting rules.
    """
    d, w_left, w_right = _split(x, loc, scale_left, scale_right)
    _, log_w_right = _log_weights(scale_left, scale_right)
    return jnp.where(
        d < 0,
        _log1m_mixture(
            w_right,
            w_left,
            halfnorm.cdf(-d, 0.0, scale_left),
            halfnorm.sf(-d, 0.0, scale_left),
        ),
        log_w_right + halfnorm.logsf(d, 0.0, scale_right),
    )


def _quantile(p_below, p_above, delta, loc, scale_left, scale_right):
    # The x with P(X <= x) = p_below and P(X > x) = p_above, where
    # delta = p_below - w_left = w_right - p_above is passed in whichever
    # form is exact. Within the piece on x's side, the half-normal tail
    # probability beyond x is t = p_below / w_left (left) or
    # p_above / w_right (right). As in halfnorm.ppf, x is
    # loc + scale * sqrt(2) * erfinv(delta / w) for t > 1/2 (the central
    # half, where delta is small and accurate) and loc -+ scale *
    # ndtri(t / 2) beyond it (where the tail probability is), so neither
    # loses precision to a 1 - q. Unselected forms get safe substitutes.
    scale_left = jnp.asarray(scale_left)
    scale_right = jnp.asarray(scale_right)
    w_left, w_right = _weights(scale_left, scale_right)
    left = delta < 0
    w = jnp.where(left, w_left, w_right)
    p_tail = jnp.where(left, p_below, p_above)
    central = p_tail > 0.5 * w
    t = jnp.where(central, 0.25, p_tail / (2.0 * w))
    z = jnp.where(left, ndtri(t), -ndtri(t))
    u = jnp.where(central, delta / w, 0.0)
    std = jnp.where(central, _SQRT2 * erfinv(u), z)
    scale = jnp.where(left, scale_left, scale_right)
    return jnp.asarray(loc) + scale * std


def ppf(
    q: ArrayLike,
    loc: ArrayLike = 0,
    scale_left: ArrayLike = 1,
    scale_right: ArrayLike = 1,
) -> ArrayLike:
    r"""
    Split-normal percent point function (inverse CDF).

    Inverts the half-normal piece on the relevant side of the mode. In
    the central half of a piece ($w_L/2 < q < w_L + w_R/2$) it computes
    $\mu + \sigma \sqrt{2}\, \text{erfinv}((q - w_L)/w)$, with $\sigma$
    and $w$ the scale and weight of that side; in the lower tail
    $\mu + \sigma_L \Phi^{-1}(q / 2w_L)$; and in the upper tail
    $\mu - \sigma_R \Phi^{-1}((1 - q) / 2w_R)$, where $1 - q$ is exact.
    Tiny $q$, $q$ close to one, and $q$ on the far side of a small $w_L$
    all keep full precision. Returns ``-inf`` at $q = 0$, ``inf`` at
    $q = 1$, and ``nan`` outside $[0, 1]$. See
    [`logpdf`][numerax.stats.splitnorm.logpdf] for the notation and
    mathematical background.

    ## Args

    - **q**: Probability values in $[0, 1]$. Scalar or array.
    - **loc**: Location of the mode (default: 0). Scalar or array.
    - **scale_left**: Scale below the mode (must be positive, default:
      1). Scalar or array.
    - **scale_right**: Scale above the mode (must be positive, default:
      1). Scalar or array.

    ## Returns

    Quantiles $x$ where $P(X \leq x) = q$. Shape follows JAX broadcasting
    rules.
    """
    q = jnp.asarray(q)
    w_left, _ = _weights(scale_left, scale_right)
    x = _quantile(q, 1.0 - q, q - w_left, loc, scale_left, scale_right)
    return jnp.where((q >= 0) & (q <= 1), x, jnp.nan)


def isf(
    q: ArrayLike,
    loc: ArrayLike = 0,
    scale_left: ArrayLike = 1,
    scale_right: ArrayLike = 1,
) -> ArrayLike:
    r"""
    Split-normal inverse survival function.

    The mirror image of [`ppf`][numerax.stats.splitnorm.ppf]: computes
    ``ppf(1 - q)`` without forming $1 - q$ except in the lower tail
    ($q > 1 - w_L/2$), where it is exact, so upper-tail quantiles for tiny
    $q$ and quantiles on the far side of a small $w_R$ stay accurate.
    Returns ``inf`` at $q = 0$, ``-inf`` at $q = 1$, and ``nan`` outside
    $[0, 1]$. See
    [`logpdf`][numerax.stats.splitnorm.logpdf] for the notation and
    mathematical background.

    ## Args

    - **q**: Survival probabilities in $[0, 1]$. Scalar or array.
    - **loc**: Location of the mode (default: 0). Scalar or array.
    - **scale_left**: Scale below the mode (must be positive, default:
      1). Scalar or array.
    - **scale_right**: Scale above the mode (must be positive, default:
      1). Scalar or array.

    ## Returns

    Quantiles $x$ where $P(X > x) = q$. Shape follows JAX broadcasting
    rules.
    """
    q = jnp.asarray(q)
    _, w_right = _weights(scale_left, scale_right)
    x = _quantile(1.0 - q, q, w_right - q, loc, scale_left, scale_right)
    return jnp.where((q >= 0) & (q <= 1), x, jnp.nan)
