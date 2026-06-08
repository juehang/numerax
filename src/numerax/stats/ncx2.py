"""
Non-central chi-squared distribution functions.

This module provides the probability density function (``pdf``) and its
logarithm (``logpdf``) for the non-central chi-squared distribution,
built on the numerically stable scaled modified Bessel function
[`ive`][numerax.special.ive]. The signatures mirror
``scipy.stats.ncx2`` (including ``loc`` and ``scale``), and all
functions are compatible with JAX transformations (JIT, grad, vmap).
"""

import jax.numpy as jnp
from jax.scipy.stats.chi2 import logpdf as _chi2_logpdf
from jaxtyping import ArrayLike

from numerax.special import ive


def logpdf(
    x: ArrayLike,
    df: ArrayLike,
    nc: ArrayLike,
    loc: ArrayLike = 0,
    scale: ArrayLike = 1,
) -> ArrayLike:
    r"""
    Non-central chi-squared log probability density function.

    ## Overview

    Computes the natural logarithm of the probability density function of
    the non-central chi-squared distribution with $\text{df}$ degrees of
    freedom and non-centrality parameter $\lambda$ (``nc``), in the
    location-scale family.

    ## Mathematical Background

    The non-central chi-squared distribution is the distribution of the
    sum of squares of $k$ independent unit-variance normal random
    variables with means $\mu_1, \ldots, \mu_k$:

    $$X = \sum_{i=1}^{k} (Z_i + \mu_i)^2, \qquad Z_i \sim N(0, 1),$$

    with degrees of freedom $k = \text{df}$ and non-centrality parameter
    $\lambda = \sum_{i=1}^{k} \mu_i^2$. Note that ``nc`` is $\lambda$
    itself (matching ``scipy.stats.ncx2``), not $\lambda^2$.

    The standard ($\text{loc}=0$, $\text{scale}=1$) density is

    $$f(x; k, \lambda) = \tfrac12 e^{-(x + \lambda)/2}
        \left(\tfrac{x}{\lambda}\right)^{\nu/2}
        I_{\nu}\!\left(\sqrt{\lambda x}\right),
        \qquad \nu = \tfrac{k}{2} - 1,$$

    where $I_{\nu}$ is the modified Bessel function of the first kind.
    Using $I_{\nu}(z) = e^{z}\,\mathtt{ive}(\nu, z)$ and combining the
    exponentials via $-(x+\lambda)/2 + \sqrt{\lambda x}
    = -\tfrac12(\sqrt{x} - \sqrt{\lambda})^2$ gives the numerically
    stable form actually evaluated:

    $$f = \tfrac12\, e^{-\frac12(\sqrt{x} - \sqrt{\lambda})^2}
        \left(\tfrac{x}{\lambda}\right)^{\nu/2}
        \mathtt{ive}\!\left(\nu, \sqrt{\lambda x}\right).$$

    The location-scale family follows the usual convention
    $f(x; \text{loc}, \text{scale}) =
    \tfrac{1}{\text{scale}} f_{\text{std}}\!\left(
    \tfrac{x - \text{loc}}{\text{scale}}\right)$.

    ## Args

    - **x**: Quantile values. Scalar or array.
    - **df**: Degrees of freedom (must be positive). Scalar or array.
    - **nc**: Non-centrality parameter $\lambda \ge 0$. Scalar or array.
      ``nc == 0`` reduces to the central chi-squared distribution.
    - **loc**: Location parameter (default: 0). Scalar or array.
    - **scale**: Scale parameter (must be positive, default: 1). Scalar
      or array.

    ## Returns

    Log probability density values. Shape follows JAX broadcasting rules.

    ## Example

    ```python
    import jax.numpy as jnp
    import numerax

    # Single value
    lp = numerax.stats.ncx2.logpdf(4.0, df=3.0, nc=2.0)

    # Vectorized
    x_vals = jnp.array([0.5, 1.0, 2.0, 5.0])
    lps = numerax.stats.ncx2.logpdf(x_vals, df=3.0, nc=2.0)

    # Differentiable in x and nc
    grad_fn = jax.grad(numerax.stats.ncx2.logpdf, argnums=2)
    sensitivity = grad_fn(4.0, 3.0, 2.0)  # d logpdf / d nc
    ```

    ## Notes

    - **Differentiable**: w.r.t. ``x``, ``nc``, ``loc``, and ``scale``.
      **Not** differentiable w.r.t. ``df`` -- ``df`` feeds the order of
      the underlying [`ive`][numerax.special.ive], which has no
      order-derivative, so differentiating w.r.t. ``df`` raises
      ``TypeError`` rather than returning a silently-wrong value.
    - **Gradient at** $\text{nc} = 0$: the gradient w.r.t. ``nc`` is
      correct for every $\text{nc} > 0$, but at *exactly* $\text{nc} = 0$
      it returns ``0`` rather than the true one-sided score
      $\tfrac12\!\left(\tfrac{(x-\text{loc})/\text{scale}}{\text{df}}
      - 1\right)$. $\text{nc} = 0$ is the boundary of the parameter
      domain (where the derivative is one-sided and the closed form has a
      removable singularity), so this measure-zero point is not special
      cased. Differentiate at a small $\text{nc} > 0$ if the score at the
      null is needed.
    - **Broadcasting**: Supports JAX array broadcasting for all
      parameters.
    - **Accuracy**: inherits the ~1e-6 relative accuracy of ``ive``.
    - **Support boundary**: for $x \le \text{loc}$ the density is taken
      to be zero (``logpdf`` returns ``-inf``), matching
      ``scipy.stats.ncx2.pdf``. At the single point $x = \text{loc}$ the
      mathematical limit can diverge for $\text{df} < 2$; that
      measure-zero point is reported as zero density here.
    """
    x = jnp.asarray(x)
    df = jnp.asarray(df)
    nc = jnp.asarray(nc)
    loc = jnp.asarray(loc)
    scale = jnp.asarray(scale)

    y = (x - loc) / scale
    nu = df / 2.0 - 1.0

    # Safe substitutes keep the general branch finite (and its gradient
    # NaN-free) on the boundary so the jnp.where selections below do not
    # poison reverse-mode gradients at interior points.
    safe_nc = jnp.where(nc > 0, nc, 1.0)
    safe_y = jnp.where(y > 0, y, 1.0)

    z = jnp.sqrt(safe_nc * safe_y)
    log_std = (
        -jnp.log(2.0)
        - 0.5 * (jnp.sqrt(safe_y) - jnp.sqrt(safe_nc)) ** 2
        + (nu / 2.0) * jnp.log(safe_y / safe_nc)
        + jnp.log(ive(nu, z))
    )
    # x <= loc has zero density on the non-central branch.
    general = jnp.where(y > 0, log_std - jnp.log(scale), -jnp.inf)

    # nc == 0 reduces to the central chi-squared, which handles its own
    # support boundary (including the df < 2 divergence at x = loc).
    central = _chi2_logpdf(x, df, loc, scale)

    return jnp.where(nc > 0, general, central)


def pdf(
    x: ArrayLike,
    df: ArrayLike,
    nc: ArrayLike,
    loc: ArrayLike = 0,
    scale: ArrayLike = 1,
) -> ArrayLike:
    r"""
    Non-central chi-squared probability density function.

    Computes the probability density function of the non-central
    chi-squared distribution with $\text{df}$ degrees of freedom and
    non-centrality parameter $\lambda$ (``nc``), in the location-scale
    family. This is ``exp(logpdf(...))``; see
    [`logpdf`][numerax.stats.ncx2.logpdf] for the full mathematical
    background, conventions, and differentiability notes.

    ## Args

    - **x**: Quantile values. Scalar or array.
    - **df**: Degrees of freedom (must be positive). Scalar or array.
    - **nc**: Non-centrality parameter $\lambda \ge 0$. Scalar or array.
    - **loc**: Location parameter (default: 0). Scalar or array.
    - **scale**: Scale parameter (must be positive, default: 1). Scalar
      or array.

    ## Returns

    Probability density values. Shape follows JAX broadcasting rules.

    ## Example

    ```python
    import jax.numpy as jnp
    import numerax

    # Single value
    p = numerax.stats.ncx2.pdf(4.0, df=3.0, nc=2.0)

    # Vectorized
    x_vals = jnp.array([0.5, 1.0, 2.0, 5.0])
    ps = numerax.stats.ncx2.pdf(x_vals, df=3.0, nc=2.0)
    ```

    ## Notes

    - **Differentiable**: w.r.t. ``x``, ``nc``, ``loc``, and ``scale``;
      differentiating w.r.t. ``df`` raises ``TypeError``. The gradient
      w.r.t. ``nc`` returns ``0`` at exactly ``nc = 0`` (a domain
      boundary). See [`logpdf`][numerax.stats.ncx2.logpdf] for details.
    - **Accuracy**: inherits the ~1e-6 relative accuracy of ``ive``.
    """
    return jnp.exp(logpdf(x, df, nc, loc, scale))
