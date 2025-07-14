"""
Statistical and numerical computation functions for JAX, focusing on tools
not available in the main JAX API.

## Overview

This package provides JAX-compatible implementations of specialized numerical
functions with full differentiability support. All functions are designed to
work seamlessly with JAX's transformations (JIT, grad, vmap, etc.) and follow
JAX's functional programming paradigms.

### Special Functions (`numerax.special`)

Mathematical special functions with custom derivative implementations.
Functions use numerically stable algorithms and provide exact gradients
through custom JVP rules where standard automatic differentiation would
be inefficient or unstable.

### Statistical Methods (`numerax.stats`)

Advanced statistical computation tools for inference problems. Implements
efficient algorithms for complex statistical models, with particular focus
on optimization-based methods that benefit from JAX's compilation and
differentiation capabilities.

### Utilities (`numerax.utils`)

Development utilities for creating JAX-compatible functions with proper
documentation support. Includes decorators and helpers for preserving
function metadata when using JAX's advanced features like custom derivatives.
"""

from . import special, stats, utils

__version__ = "0.2.0"

__all__ = ["special", "stats", "utils"]
