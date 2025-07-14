# numerax

[![tests](https://github.com/juehang/numerax/actions/workflows/test.yml/badge.svg)](https://github.com/juehang/numerax/actions/workflows/test.yml)
[![docs](https://github.com/juehang/numerax/actions/workflows/docs.yml/badge.svg)](https://juehang.github.io/numerax/)

Statistical and numerical computation functions for JAX, focusing on tools not available in the main JAX API.

**[📖 Documentation](https://juehang.github.io/numerax/)**

## Installation

```bash
pip install numerax
```

## Features

### Special Functions

Inverse regularized incomplete gamma function with differentiability support:

```python
import jax.numpy as jnp
import numerax

# Compute gamma quantiles (inverse CDF)
p = jnp.array([0.1, 0.5, 0.9])  # Probabilities
a = 2.0  # Shape parameter

x = numerax.special.gammap_inverse(p, a)
# Returns quantiles where gammainc(a, x) = p

# Fully differentiable with custom JVP
grad_fn = jax.grad(numerax.special.gammap_inverse)
dx_dp = grad_fn(0.5, 2.0)  # Gradient with respect to probability
```

**Key features:**
- Halley's method for fast convergence
- Custom JVP implementation for exact gradients
- Numerical stability with adaptive precision
- Equivalent to gamma distribution inverse CDF

### Profile Likelihood

Efficient profile likelihood computation for statistical inference with nuisance parameters:

```python
import jax.numpy as jnp
import numerax

# Example: Normal distribution with mean inference, variance profiling
def normal_llh(params, data):
    mu, log_sigma = params
    sigma = jnp.exp(log_sigma)
    return jnp.sum(-0.5 * jnp.log(2 * jnp.pi) - log_sigma 
                   - 0.5 * ((data - mu) / sigma) ** 2)

# Profile over log_sigma, infer mu
is_nuisance = [False, True]  # mu=inference, log_sigma=nuisance

def get_initial_log_sigma(data):
    return jnp.array([jnp.log(jnp.std(data))])

profile_llh = numerax.stats.make_profile_llh(
    normal_llh, is_nuisance, get_initial_log_sigma
)

# Evaluate profile likelihood
data = jnp.array([1.2, 0.8, 1.5, 0.9, 1.1])
llh_val, opt_nuisance, diff, n_iter = profile_llh(jnp.array([1.0]), data)
```

**Key features:**
- JIT-compiled for performance
- L-BFGS optimization with convergence diagnostics
- Configurable tolerance and initial values
- Handles parameter masking automatically

### Utilities

Development utilities for creating JAX functions with custom derivatives while ensuring proper documentation support. Includes decorators for preserving function metadata when using JAX's advanced features.

## Requirements

- Python ≥ 3.12
- JAX
- jaxtyping
- optax

## Acknowledgements
This work is supported by the Department of Energy AI4HEP program.