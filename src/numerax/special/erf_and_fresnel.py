import jax.scipy.special
from jaxtyping import ArrayLike

"""
Error functions and Fresnel integrals.

This module provides error functions and Fresnel integrals that are missing
from jax.scipy.special but available in scipy.special, offering complete
compatibility with scipy's API while maintaining full JAX differentiability.
"""


def erfcinv(x: ArrayLike) -> ArrayLike:
    r"""
    Inverse complementary error function.

    ## Overview

    Computes the inverse of the complementary error function, finding $y$ such
    that $\text{erfc}(y) = x$ for given $x \in (0, 2)$.

    ## Mathematical Background

    The inverse complementary error function is related to the inverse error
    function by:

    $$\text{erfcinv}(x) = \text{erfinv}(1 - x)$$

    This relationship allows us to implement erfcinv as a simple wrapper around
    the existing JAX implementation of erfinv.

    ## Args

    - **x**: Input values in $(0, 2)$. Can be scalar or array.

    ## Returns

    Values $y$ where $\text{erfc}(y) = x$. Shape follows JAX broadcasting rules.

    ## Example

    ```python
    import jax.numpy as jnp
    import numerax

    # Single value
    y = numerax.special.erfcinv(0.5)  # ≈ 0.4769

    # Array input
    x_vals = jnp.array([0.1, 0.5, 1.0, 1.5, 1.9])
    y_vals = numerax.special.erfcinv(x_vals)

    # Differentiable for optimization
    grad_fn = jax.grad(numerax.special.erfcinv)
    sensitivity = grad_fn(0.5)
    ```

    ## Notes

    - **Differentiable**: Full automatic differentiation support through JAX
    - **Broadcasting**: Supports JAX array broadcasting
    - **Domain**: Input must be in $(0, 2)$ for real outputs
    - **Performance**: JIT-compiled compatibility
    """
    return jax.scipy.special.erfinv(1.0 - x)