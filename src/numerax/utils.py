"""Utility functions for the numerax package."""

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
