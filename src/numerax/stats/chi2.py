"""
Chi-squared distribution functions.

This module provides a complete interface for chi-squared distribution
computations, combining re-exported JAX's standard statistical functions
(pdf, cdf, etc.) with a custom high-precision percent point function (ppf).

All functions support location-scale parameterization and are fully
compatible with JAX transformations (JIT, grad, vmap).
"""

import jax
import jax.numpy as jnp
from jax.scipy.stats.chi2 import *  # noqa: F403 # Used to import names into module namespace
from jaxtyping import ArrayLike

from numerax.special import gammap_inverse


def _vectorized_gammap_inverse(q_flat, df_flat):
    """Helper function for vectorizing gammap_inverse over df parameter."""
    return gammap_inverse(q_flat, df_flat / 2.0)


def ppf(
    q: ArrayLike, df: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1
) -> ArrayLike:
    r"""
    Chi-squared percent point function (inverse CDF).

    ## Overview

    Computes the percent point function (quantile function) of the chi-squared
    distribution. This is the inverse of the cumulative distribution function,
    finding $x$ such that $P(X \leq x) = q$ for a chi-squared random
    variable $X$ with $\text{df}$ degrees of freedom.

    ## Mathematical Background

    The chi-squared distribution with $\text{df}$ degrees of freedom is a
    special case of the gamma distribution:

    $$X \sim \chi^2(\text{df}) \equiv \text{Gamma}\left(
    \frac{\text{df}}{2}, 2\right)$$

    For the location-scale family:

    $$Y = \text{loc} + \text{scale} \cdot X$$

    The percent point function is computed as:

    $$
    \text{ppf}(q, \text{df}, \text{loc}, \text{scale}) =
    \text{loc} + \text{scale} \cdot 2 \cdot
    \text{gammap\_inverse}\left(q, \frac{\text{df}}{2}\right)
    $$

    ## Args

    - **q**: Probability values in $[0, 1]$. Can be scalar or array.
    - **df**: Degrees of freedom (must be positive). Can be scalar or array.
    - **loc**: Location parameter (default: 0). Can be scalar or array.
    - **scale**: Scale parameter (must be positive, default: 1). Can be
      scalar or array.

    ## Returns

    Quantiles $x$ where $P(X \leq x) = q$. Shape follows JAX broadcasting
    rules.

    ## Example

    ```python
    import jax.numpy as jnp
    import numerax

    # Single quantile
    x = numerax.stats.chi2.ppf(0.5, df=2)  # Median of χ²(2)

    # Multiple quantiles
    q_vals = jnp.array([0.1, 0.25, 0.5, 0.75, 0.9])
    x_vals = numerax.stats.chi2.ppf(q_vals, df=3)

    # Location-scale family
    x_scaled = numerax.stats.chi2.ppf(0.5, df=2, loc=1, scale=2)

    # Differentiable for optimization
    grad_fn = jax.grad(numerax.stats.chi2.ppf)
    sensitivity = grad_fn(0.5, 2.0)  # ∂x/∂q at median
    ```

    ## Notes

    - **Differentiable**: Automatic differentiation through `gammap_inverse`
    - **Broadcasting**: Supports JAX array broadcasting for all parameters
    - **Performance**: JIT-compiled compatibility
    """
    # Convert inputs to JAX arrays to enable broadcasting
    q = jnp.asarray(q)
    degrees_freedom = jnp.asarray(df)
    loc = jnp.asarray(loc)
    scale = jnp.asarray(scale)

    # Flatten inputs for vectorization, then reshape back
    orig_shape = jnp.broadcast_shapes(
        q.shape, degrees_freedom.shape, loc.shape, scale.shape
    )

    q_flat = jnp.broadcast_to(q, orig_shape).flatten()
    df_flat = jnp.broadcast_to(degrees_freedom, orig_shape).flatten()
    loc_flat = jnp.broadcast_to(loc, orig_shape).flatten()
    scale_flat = jnp.broadcast_to(scale, orig_shape).flatten()

    # Apply vectorized computation
    result_flat = jax.vmap(_vectorized_gammap_inverse)(q_flat, df_flat)
    result = loc_flat + scale_flat * 2.0 * result_flat

    return result.reshape(orig_shape)
