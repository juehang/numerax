# numerax

[![tests](https://github.com/juehang/numerax/actions/workflows/test.yml/badge.svg)](https://github.com/juehang/numerax/actions/workflows/test.yml)
[![Coverage Status](https://coveralls.io/repos/github/juehang/numerax/badge.svg?branch=main)](https://coveralls.io/github/juehang/numerax?branch=main)
[![docs](https://github.com/juehang/numerax/actions/workflows/docs.yml/badge.svg)](https://juehang.github.io/numerax/)
[![DOI](https://zenodo.org/badge/1018495069.svg)](https://zenodo.org/badge/latestdoi/1018495069)

Statistical and numerical computation functions for JAX, focusing on tools not available in the main JAX API.

**[📖 Documentation](https://juehang.github.io/numerax/)**

## Installation

```bash
pip install numerax

# With scientific ML dependencies like equinox
pip install numerax[sciml]
```

## Features

### Special Functions

Inverse functions for statistical distributions with differentiability support:

```python
import jax.numpy as jnp
import numerax

# Inverse functions for statistical distributions
x = numerax.special.gammap_inverse(p, a)  # Gamma quantiles
y = numerax.special.erfcinv(x)  # Inverse complementary error function

# Chi-squared distribution (includes JAX functions + custom ppf)
x = numerax.stats.chi2.ppf(q, df, loc=0, scale=1)
```

**Key features:**
- Inverse functions for statistical distributions missing from JAX
- Full differentiability and JAX transformation support

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
- Convergence diagnostics and configurable optimization parameters
- Automatic parameter masking for inference vs. nuisance parameters

### Utilities

Utilities for working with PyTree-based models, including parameter counting and model summaries.

```python
from numerax.utils import count_params, tree_summary
import jax.numpy as jnp

# Count parameters in PyTree-based models
model = {"weights": jnp.ones((10, 5)), "bias": jnp.zeros(5)}
num_params = count_params(model)  # 55 parameters

# Pretty-print model structure (similar to Keras model.summary())
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
tree_summary(model)
# ======================================================================
# PyTree Summary
# ======================================================================
# Name                  Shape           Dtype             Params
# ----------------------------------------------------------------------
# root                                                       325
#   encoder                                                  220
#     - weights         [10,20]         float32              200
#     - bias            [20]            float32               20
#   decoder                                                  105
#     - weights         [20,5]          float32              100
#     - bias            [5]             float32                5
# ======================================================================
# Total params: 325
# ======================================================================
```

**Key features:**
- Parameter counting for PyTree-based models including Equinox (requires `numerax[sciml]`)
- Model structure visualization with shapes, dtypes, and parameter counts
- Decorators for preserving function metadata when using JAX's advanced features

## Acknowledgements
This work is supported by the Department of Energy AI4HEP program.

## Citation
If you use `numerax` in your research, please cite it using the citation information from Zenodo (click the DOI badge at the top of the README) to ensure you get the correct DOI for the version you used.