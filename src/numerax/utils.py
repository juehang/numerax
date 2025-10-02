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


def tree_summary(pytree, is_leaf=None, max_depth=3, verbose=True):
    """
    Pretty-print PyTree structure with shapes and parameter counts.

    ## Overview

    This function displays a hierarchical view of a PyTree structure
    (e.g., neural network models) showing the organization, array shapes,
    data types, and parameter counts at each level. The output is
    similar to Keras' `model.summary()` or torchinfo's summaries.

    ## Args

    - **pytree**: The PyTree structure to summarize (e.g., a model,
      dict of arrays, or nested structure)
    - **is_leaf**: Optional function to identify leaf nodes. If `None`,
      uses `equinox.is_array` as the default. Leaf nodes are displayed
      with shape, dtype, and parameter count details. Custom functions
      should accept a single argument and return `True` for leaves
    - **max_depth**: Maximum nesting depth to display. Nodes deeper
      than this level will not be shown. Defaults to 3
    - **verbose**: If `True`, prints the formatted summary. If `False`,
      only returns the total parameter count silently

    ## Returns

    The total number of parameters as an integer

    ## Requirements

    - **equinox**: Install with `pip install numerax[sciml]` or
      `pip install equinox` (required when using default `is_leaf`)

    ## Example

    ```python
    import jax.numpy as jnp
    from numerax.utils import tree_summary

    # Nested dict-based model
    model = {
        "encoder": {
            "weights": jnp.ones((10, 20)),
            "bias": jnp.zeros(20),
        },
        "decoder": {
            "weights": jnp.ones((20, 5)),
            "bias": jnp.zeros(5),
        },
    }

    count = tree_summary(model)
    # Prints formatted table showing structure
    # Returns: 325

    # With custom is_leaf function
    tree_summary(model, is_leaf=lambda x: hasattr(x, "shape"))

    # Limit depth
    tree_summary(model, max_depth=2)

    # Silent mode
    count = tree_summary(model, verbose=False)
    # Returns: 325 without printing
    ```

    ## Output Format

    ```
    ==================================================================
    PyTree Summary
    ==================================================================
    Name                  Shape           Dtype           Params
    ------------------------------------------------------------------
    encoder                                                   220
      - weights           [10,20]         f32               200
      - bias              [20]            f32                20
    decoder                                                   105
      - weights           [20,5]          f32               100
      - bias              [5]             f32                 5
    ==================================================================
    Total params: 325
    ==================================================================
    ```

    ## Notes

    - Container nodes (dicts, lists, modules) show total parameter
      counts for their entire subtree
    - Leaf nodes (arrays) show shape, dtype, and individual param count
    - Indentation shows nesting depth in the PyTree structure
    - Works with Equinox modules, nested dicts, lists, tuples, and
      custom PyTree nodes
    - Use custom `is_leaf` functions to control what counts as a leaf
      node (useful for custom PyTree registrations)
    """
    import equinox as eqx  # noqa: PLC0415

    if is_leaf is None:
        is_leaf = eqx.is_array

    # Column widths for alignment
    col_name = 22
    col_shape = 16
    col_dtype = 12
    col_params = 12
    total_width = 70

    lines = []

    # Calculate total params once using existing utility
    total_params = count_params(pytree, filter=is_leaf, verbose=False)

    def _traverse(pytree, name, depth):
        """Recursively traverse PyTree and collect formatted lines."""
        if depth > max_depth:
            return

        indent = "  " * depth

        if is_leaf(pytree):
            # Leaf node - format with shape, dtype, params
            shape_str = f"[{','.join(map(str, pytree.shape))}]"
            dtype_str = pytree.dtype.name
            # Use count_params for consistency
            params = count_params(pytree, filter=is_leaf, verbose=False)

            line = (
                f"{indent}- {name:<{col_name - len(indent) - 2}}"
                f"{shape_str:<{col_shape}}"
                f"{dtype_str:<{col_dtype}}{params:>{col_params},}"
            )
            lines.append(line)
        else:
            # Container node - use count_params for subtree total
            subtree_params = count_params(
                pytree, filter=is_leaf, verbose=False
            )

            header = (
                f"{indent}{name:<{col_name - len(indent)}}"
                f"{'':<{col_shape}}"
                f"{'':<{col_dtype}}{subtree_params:>{col_params},}"
            )
            lines.append(header)

            # Recurse into children based on container type
            if isinstance(pytree, dict):
                for key, value in pytree.items():
                    _traverse(value, str(key), depth + 1)
            elif isinstance(pytree, list | tuple):
                for i, value in enumerate(pytree):
                    _traverse(value, f"[{i}]", depth + 1)
            elif hasattr(pytree, "__dict__"):
                # Equinox modules, dataclasses, etc.
                for key, value in vars(pytree).items():
                    _traverse(value, key, depth + 1)

    # Traverse the tree
    _traverse(pytree, "root", 0)

    if verbose:
        # Print header
        print("=" * total_width)
        print("PyTree Summary")
        print("=" * total_width)

        # Print column headers
        header_line = (
            f"{'Name':<{col_name}}{'Shape':<{col_shape}}"
            f"{'Dtype':<{col_dtype}}{'Params':>{col_params}}"
        )
        print(header_line)
        print("-" * total_width)

        # Print tree structure
        for line in lines:
            print(line)

        # Print footer
        print("=" * total_width)
        print(f"Total params: {total_params:,}")
        print("=" * total_width)

    return total_params
