"""Utility functions for the numerax package.

This module provides development utilities for creating JAX-compatible
functions and tools for working with PyTree structures, including parameter
counting for machine learning models.
"""

import functools
from collections.abc import Callable
from typing import TypeVar

F = TypeVar("F", bound=Callable)


def preserve_metadata(decorator):
    """
    Wrapper that ensures a decorator preserves function metadata for
    documentation tools.

    ## Overview

    This is particularly useful for JAX decorators like `@custom_jvp` that
    create special objects which may not preserve `__doc__` and other metadata
    properly for documentation generators like pdoc.

    ## Args

    - **decorator**: The decorator function to wrap

    ## Returns

    A new decorator that preserves metadata

    ## Example

    ```python
    import jax
    from numerax.utils import preserve_metadata

    @preserve_metadata(jax.custom_jvp)
    def my_function(x):
        \"\"\"This docstring will be preserved for automatic
        documentation generation.\"\"\"
        return x
    ```
    """

    def metadata_preserving_decorator(func: F) -> F:
        # Apply the original decorator
        decorated = decorator(func)
        # Ensure metadata is preserved using functools.wraps pattern
        return functools.wraps(func)(decorated)

    return metadata_preserving_decorator


def count_params(pytree, filter=None, verbose=True):
    """
    Count the total number of parameters in a PyTree structure.

    ## Overview

    This function counts parameters in PyTree-based models by filtering for
    array-like objects and summing their sizes. It is particularly useful
    for neural network models built with JAX frameworks like Equinox.

    The function traverses the PyTree structure, applies a filter to
    identify parameter arrays, and computes the total parameter count.

    ## Args

    - **pytree**: The PyTree structure to count parameters in (e.g., a
      model, dict of arrays, or nested structure)
    - **filter**: Optional filter function to identify parameters. If
      `None`, uses `equinox.is_array` as the default filter. Custom
      filters should accept a single argument and return `True` for
      objects that should be counted
    - **verbose**: If `True`, prints the parameter count in scientific
      notation. If `False`, only returns the count silently

    ## Returns

    The total number of parameters as an integer

    ## Requirements

    - **equinox**: Install with `pip install numerax[sciml]` or
      `pip install equinox`

    ## Example

    ```python
    import jax.numpy as jnp
    from numerax.utils import count_params

    # Simple dict-based model
    model = {"weights": jnp.ones((10, 5)), "bias": jnp.zeros(5)}
    count = count_params(model)
    # Prints: Number of parameters: 5.5e+01
    # Returns: 55

    # With custom filter
    count = count_params(
        model,
        filter=lambda x: hasattr(x, "ndim") and x.ndim > 1,
        verbose=False,
    )
    # Returns: 50 (only the weights matrix)

    # With Equinox model
    import equinox as eqx


    class MLP(eqx.Module):
        layers: list

        def __init__(self, key):
            self.layers = [
                eqx.nn.Linear(10, 64, key=key),
                eqx.nn.Linear(64, 1, key=key),
            ]


    model = MLP(jax.random.PRNGKey(0))
    count = count_params(model)
    # Counts all trainable parameters in the MLP
    ```

    ## Notes

    - The default filter (`equinox.is_array`) correctly identifies
      parameter arrays in Equinox modules and standard JAX PyTrees
    - For custom filtering logic, provide a function that returns
      `True` for leaves that should be counted as parameters
    - The function handles nested PyTree structures automatically
    """
    import equinox as eqx  # noqa: PLC0415
    import jax  # noqa: PLC0415

    if filter is None:
        filter = eqx.is_array

    # Use equinox.filter to extract only the parameter arrays
    num_params = sum(
        x.size for x in jax.tree_util.tree_leaves(eqx.filter(pytree, filter))
    )

    if verbose:
        print(f"Number of parameters: {num_params:.1e}")

    return num_params
