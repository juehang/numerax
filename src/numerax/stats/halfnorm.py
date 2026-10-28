"""
Half-normal distribution functions.

This module provides the density, distribution, and quantile functions of
the half-normal distribution (the distribution of $|Z|$ for a standard
normal $Z$), which JAX does not provide. The signatures mirror
``scipy.stats.halfnorm`` (including ``loc`` and ``scale``), each function is
evaluated in a form that stays accurate in the tails, and all functions are
compatible with JAX transformations (JIT, grad, vmap).
"""

import jax.numpy as jnp
from jax.scipy.special import erf, erfc, erfinv, ndtri
from jaxtyping import ArrayLike

_SQRT2 = jnp.sqrt(2.0)

# logsf switches from log(erfc) to the asymptotic Mills-ratio expansion
# here, comfortably before erfc(y / sqrt(2)) leaves the normal range.
_LOGSF_TAIL = 35.0
# (-1)^k (2k-1)!! for k = 1..6; the next term is < 1e-16 at y = 35.
_MILLS_COEFFS = (-1.0, 3.0, -15.0, 105.0, -945.0, 10395.0)


def _standardize(x, loc, scale):
    x = jnp.asarray(x)
    loc = jnp.asarray(loc)
    scale = jnp.asarray(scale)
    return (x - loc) / scale, scale


def logpdf(
    x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1
) -> ArrayLike:
    r"""
    Half-normal log probability density function.

    ## Overview

    Computes the natural logarithm of the probability density function of
    the half-normal distribution in the location-scale family.

    ## Mathematical Background

    The half-normal distribution is the distribution of $X = |Z|$ for
    $Z \sim N(0, 1)$. Its standard ($\text{loc}=0$, $\text{scale}=1$)
    density on the support $x \ge 0$ is

    $$f(x) = \sqrt{\frac{2}{\pi}}\, e^{-x^2/2},$$

    with distribution and survival functions

    $$F(x) = \text{erf}\!\left(\tfrac{x}{\sqrt{2}}\right), \qquad
    S(x) = \text{erfc}\!\left(\tfrac{x}{\sqrt{2}}\right)
    = 2\,\Phi(-x),$$

    where $\Phi$ is the standard normal CDF. The quantile function is
    $F^{-1}(q) = \sqrt{2}\,\text{erfinv}(q) = -\Phi^{-1}\!\left(
    \tfrac{1 - q}{2}\right)$.

    The location-scale family follows the usual convention
    $f(x; \text{loc}, \text{scale}) =
    \tfrac{1}{\text{scale}} f\!\left(
    \tfrac{x - \text{loc}}{\text{scale}}\right)$.

    Every function in this module evaluates whichever of the equivalent
    forms is accurate in the relevant regime: ``cdf`` uses ``erf`` (exact
    relative accuracy near the origin) and ``sf`` uses ``erfc`` (exact
    relative accuracy in the upper tail); ``logsf`` switches to an
    asymptotic expansion so it stays accurate far past where ``erfc``
    underflows; ``ppf`` switches
    between ``erfinv`` and ``ndtri`` at $q = 1/2$ so that neither tiny
    $q$ nor $q$ close to one loses precision; and ``isf`` evaluates
    $-\Phi^{-1}(q/2)$ directly rather than forming $1 - q$.

    ## Args

    - **x**: Quantile values. Scalar or array.
    - **loc**: Location parameter (default: 0). Scalar or array.
    - **scale**: Scale parameter (must be positive, default: 1). Scalar
      or array.

    ## Returns

    Log probability density values. Shape follows JAX broadcasting rules.

    ## Example

    ```python
    import jax
    import jax.numpy as jnp
    import numerax

    # Single value
    lp = numerax.stats.halfnorm.logpdf(1.0)

    # Vectorized, with location and scale
    x_vals = jnp.array([0.5, 1.0, 2.0, 5.0])
    lps = numerax.stats.halfnorm.logpdf(x_vals, loc=0.0, scale=2.0)

    # Differentiable in x, loc, and scale
    grad_fn = jax.grad(numerax.stats.halfnorm.logpdf, argnums=2)
    sensitivity = grad_fn(1.0, 0.0, 2.0)  # d logpdf / d scale
    ```

    ## Notes

    - **Differentiable**: w.r.t. ``x``, ``loc``, and ``scale`` (and ``q``
      for ``ppf``/``isf``). Branch selections use safe substitutes so
      gradients stay NaN-free at interior points.
    - **Broadcasting**: Supports JAX array broadcasting for all
      parameters.
    - **Support boundary**: for $x < \text{loc}$ the density is zero
      (``logpdf`` returns ``-inf``), matching ``scipy.stats.halfnorm``.
      $x = \text{loc}$ is in the support, where the density is
      $\sqrt{2/\pi}/\text{scale}$.
    """
    y, scale = _standardize(x, loc, scale)
    log_std = 0.5 * jnp.log(2.0 / jnp.pi) - 0.5 * y**2
    return jnp.where(y >= 0, log_std - jnp.log(scale), -jnp.inf)


def pdf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> ArrayLike:
    r"""
    Half-normal probability density function.

    Computes $\sqrt{2/\pi}\, e^{-y^2/2} / \text{scale}$ for
    $y = (x - \text{loc})/\text{scale} \ge 0$, and zero below the
    support. This is ``exp(logpdf(...))``; see
    [`logpdf`][numerax.stats.halfnorm.logpdf] for the full mathematical
    background and conventions.

    ## Args

    - **x**: Quantile values. Scalar or array.
    - **loc**: Location parameter (default: 0). Scalar or array.
    - **scale**: Scale parameter (must be positive, default: 1). Scalar
      or array.

    ## Returns

    Probability density values. Shape follows JAX broadcasting rules.
    """
    return jnp.exp(logpdf(x, loc, scale))


def cdf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> ArrayLike:
    r"""
    Half-normal cumulative distribution function.

    Computes $\text{erf}(y/\sqrt{2})$ for
    $y = (x - \text{loc})/\text{scale} \ge 0$, and zero below the
    support. Using ``erf`` rather than $2\Phi(y) - 1$ keeps full relative
    accuracy for $y$ near zero. See
    [`logpdf`][numerax.stats.halfnorm.logpdf] for the mathematical
    background.

    ## Args

    - **x**: Quantile values. Scalar or array.
    - **loc**: Location parameter (default: 0). Scalar or array.
    - **scale**: Scale parameter (must be positive, default: 1). Scalar
      or array.

    ## Returns

    Cumulative probabilities. Shape follows JAX broadcasting rules.
    """
    y, _ = _standardize(x, loc, scale)
    return jnp.where(y >= 0, erf(y / _SQRT2), 0.0)


def logcdf(
    x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1
) -> ArrayLike:
    r"""
    Half-normal log cumulative distribution function.

    Computes $\log \text{erf}(y/\sqrt{2})$ for small $y$ and
    $\text{log1p}(-\text{erfc}(y/\sqrt{2}))$ for $y \ge 1$, where
    $y = (x - \text{loc})/\text{scale}$, so that neither regime loses
    precision. Returns ``-inf`` for $y \le 0$. See
    [`logpdf`][numerax.stats.halfnorm.logpdf] for the mathematical
    background.

    ## Args

    - **x**: Quantile values. Scalar or array.
    - **loc**: Location parameter (default: 0). Scalar or array.
    - **scale**: Scale parameter (must be positive, default: 1). Scalar
      or array.

    ## Returns

    Log cumulative probabilities. Shape follows JAX broadcasting rules.
    """
    y, _ = _standardize(x, loc, scale)
    # Safe substitute keeps log(erf(0)) = -inf (and its gradient) out of
    # the unselected branch, so reverse-mode gradients stay NaN-free.
    safe_y = jnp.where(y > 0, y, 1.0)
    log_std = jnp.where(
        safe_y < 1.0,
        jnp.log(erf(safe_y / _SQRT2)),
        jnp.log1p(-erfc(safe_y / _SQRT2)),
    )
    return jnp.where(y > 0, log_std, -jnp.inf)


def sf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> ArrayLike:
    r"""
    Half-normal survival function ($1 - \text{cdf}$).

    Computes $\text{erfc}(y/\sqrt{2})$ for
    $y = (x - \text{loc})/\text{scale} \ge 0$, and one below the
    support, with full relative accuracy in the upper tail. See
    [`logpdf`][numerax.stats.halfnorm.logpdf] for the mathematical
    background.

    ## Args

    - **x**: Quantile values. Scalar or array.
    - **loc**: Location parameter (default: 0). Scalar or array.
    - **scale**: Scale parameter (must be positive, default: 1). Scalar
      or array.

    ## Returns

    Survival probabilities. Shape follows JAX broadcasting rules.
    """
    y, _ = _standardize(x, loc, scale)
    return jnp.where(y >= 0, erfc(y / _SQRT2), 1.0)


def logsf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> ArrayLike:
    r"""
    Half-normal log survival function.

    Evaluates $\log S(y)$ for $y = (x - \text{loc})/\text{scale}$ in
    three regimes: $\text{log1p}(-\text{erf}(y/\sqrt{2}))$ for $y < 1$,
    $\log \text{erfc}(y/\sqrt{2})$ for $1 \le y < 35$, and beyond that
    (where ``erfc`` approaches underflow) the asymptotic expansion

    $$\log S(y) = \tfrac12 \log\tfrac{2}{\pi} - \tfrac{y^2}{2} - \log y
    + \log\!\left(1 + \sum_{k=1}^{6} \frac{(-1)^k (2k-1)!!}{y^{2k}}
    \right),$$

    whose truncation error is below $10^{-16}$ there. Returns zero for
    $y \le 0$. See [`logpdf`][numerax.stats.halfnorm.logpdf] for the
    mathematical background.

    This is more accurate than $\log 2 + $ ``jax.scipy.special.log_ndtr``
    $(-y)$, whose asymptotic branch loses ~1e-11 relative accuracy in the
    value (~1e-9 in the gradient) around $y = 20$.

    ## Args

    - **x**: Quantile values. Scalar or array.
    - **loc**: Location parameter (default: 0). Scalar or array.
    - **scale**: Scale parameter (must be positive, default: 1). Scalar
      or array.

    ## Returns

    Log survival probabilities. Shape follows JAX broadcasting rules.
    """
    y, _ = _standardize(x, loc, scale)
    small = y < 1.0
    tail = y >= _LOGSF_TAIL
    # Safe substitutes keep each branch finite where it is not selected
    # (e.g. log1p(-erf) hits log1p(-1) = -inf for large y), so no NaN
    # gradient leaks through the selections.
    y_lo = jnp.where(small, y, 0.5)
    y_mid = jnp.where(small | tail, 2.0, y)
    y_hi = jnp.where(tail, y, 2.0 * _LOGSF_TAIL)

    inv_y2 = 1.0 / y_hi**2
    series = 0.0
    for c in reversed(_MILLS_COEFFS):
        series = (series + c) * inv_y2
    log_tail = (
        0.5 * jnp.log(2.0 / jnp.pi)
        - 0.5 * y_hi**2
        - jnp.log(y_hi)
        + jnp.log1p(series)
    )

    log_std = jnp.where(
        small,
        jnp.log1p(-erf(y_lo / _SQRT2)),
        jnp.where(tail, log_tail, jnp.log(erfc(y_mid / _SQRT2))),
    )
    return jnp.where(y >= 0, log_std, 0.0)


def ppf(q: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> ArrayLike:
    r"""
    Half-normal percent point function (inverse CDF).

    Computes $\sqrt{2}\,\text{erfinv}(q)$ for $q < 1/2$ and
    $-\Phi^{-1}\!\left(\tfrac{1 - q}{2}\right)$ for $q \ge 1/2$, then
    applies $\text{loc} + \text{scale} \cdot (\cdot)$. The first form
    keeps tiny $q$ from being rounded away; in the second, $1 - q$ is
    exact. Returns ``loc`` at $q = 0$, ``inf`` at $q = 1$, and ``nan``
    outside $[0, 1]$. See [`logpdf`][numerax.stats.halfnorm.logpdf] for
    the mathematical background.

    ## Args

    - **q**: Probability values in $[0, 1]$. Scalar or array.
    - **loc**: Location parameter (default: 0). Scalar or array.
    - **scale**: Scale parameter (must be positive, default: 1). Scalar
      or array.

    ## Returns

    Quantiles $x$ where $P(X \leq x) = q$. Shape follows JAX broadcasting
    rules.
    """
    q = jnp.asarray(q)
    loc = jnp.asarray(loc)
    scale = jnp.asarray(scale)

    # Safe substitutes keep each branch finite where it is not selected.
    lower = q < 0.5
    q_lo = jnp.where(lower, q, 0.25)
    q_hi = jnp.where(lower, 0.75, q)
    std = jnp.where(lower, _SQRT2 * erfinv(q_lo), -ndtri((1.0 - q_hi) / 2.0))
    std = jnp.where((q >= 0) & (q <= 1), std, jnp.nan)
    return loc + scale * std


def isf(q: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> ArrayLike:
    r"""
    Half-normal inverse survival function.

    Computes $-\Phi^{-1}(q/2)$, i.e. ``ppf(1 - q)`` without forming
    $1 - q$, so upper-tail quantiles for tiny $q$ stay accurate, then
    applies $\text{loc} + \text{scale} \cdot (\cdot)$. Returns ``inf``
    at $q = 0$, ``loc`` at $q = 1$, and ``nan`` outside $[0, 1]$. See
    [`logpdf`][numerax.stats.halfnorm.logpdf] for the mathematical
    background.

    ## Args

    - **q**: Survival probabilities in $[0, 1]$. Scalar or array.
    - **loc**: Location parameter (default: 0). Scalar or array.
    - **scale**: Scale parameter (must be positive, default: 1). Scalar
      or array.

    ## Returns

    Quantiles $x$ where $P(X > x) = q$. Shape follows JAX broadcasting
    rules.
    """
    q = jnp.asarray(q)
    loc = jnp.asarray(loc)
    scale = jnp.asarray(scale)

    std = jnp.where((q >= 0) & (q <= 1), -ndtri(q / 2.0), jnp.nan)
    return loc + scale * std
