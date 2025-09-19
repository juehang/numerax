# numerax 1.0.0 Release Notes

**Release Date**: September 19, 2025

We are excited to announce the first stable release of **numerax** 1.0.0! This milestone release provides a comprehensive suite of statistical and numerical computation functions for JAX, focusing on specialized tools not available in the main JAX API.

## 🎉 Major Features

### Special Functions (`numerax.special`)

Mathematical special functions with custom derivative implementations, using numerically stable algorithms and providing exact gradients through custom JVP rules:

#### **Inverse Gamma Functions**
- **`gammap_inverse(p, a)`** - Inverse of the regularized incomplete gamma function
  - Computes quantiles of the Gamma(a, 1) distribution
  - Uses Halley's method for fast quadratic convergence
  - Custom JVP implementation for exact gradients
  - Numerical stability with adaptive precision
  - Based on methods from Numerical Recipes (Press et al., 2007)

#### **Error Functions** 
- **`erfcinv(x)`** - Inverse complementary error function
  - Finds y such that erfc(y) = x for x ∈ (0, 2)
  - Fully differentiable with JAX transformations
  - Vectorized input support

### Statistical Methods (`numerax.stats`)

Advanced statistical computation tools for inference problems with JAX optimization capabilities:

#### **Chi-squared Distribution**
- **Complete chi-squared interface** (`numerax.stats.chi2`)
  - All standard statistical functions (pdf, cdf, sf, logpdf, etc.)
  - **Custom high-precision percent point function (ppf)**
  - Location-scale parameterization support
  - Full JAX transformation compatibility

#### **Profile Likelihood**
- **`make_profile_llh()`** - Factory for creating profile likelihood functions
  - Optimizes over nuisance parameters using L-BFGS
  - JIT-compiled for high performance
  - Configurable tolerance and optimization settings
  - Handles parameter masking automatically
  - Returns optimized values, convergence diagnostics, and iteration counts

### Utilities (`numerax.utils`)

Development utilities for creating JAX-compatible functions:

#### **Metadata Preservation**
- **`preserve_metadata(decorator)`** - Ensures decorators preserve function metadata
  - Critical for documentation tools like pdoc
  - Works seamlessly with JAX decorators like `@custom_jvp`
  - Maintains docstrings and function signatures

## 🔧 Technical Highlights

### **JAX Integration**
- **Full JAX compatibility** - All functions work with JIT, grad, vmap, and other JAX transformations
- **Custom JVP implementations** - Exact gradients where automatic differentiation would be inefficient
- **Numerical stability** - Adaptive precision and robust algorithms throughout

### **Performance**
- **JIT compilation support** - All functions can be compiled for optimal performance
- **Vectorized operations** - Efficient handling of array inputs
- **Optimized algorithms** - State-of-the-art numerical methods

### **Quality & Testing**
- **Comprehensive test suite** - Validates against SciPy reference implementations
- **Gradient testing** - Ensures differentiability works correctly
- **Edge case handling** - Robust behavior at distribution boundaries

## 📦 Installation

```bash
pip install numerax
```

## 🚀 Quick Start

```python
import jax.numpy as jnp
import numerax

# Special functions with gradients
x = numerax.special.gammap_inverse(0.5, 2.0)  # Gamma quantiles
y = numerax.special.erfcinv(0.5)              # Inverse complementary error

# Chi-squared distribution
quantile = numerax.stats.chi2.ppf(0.95, df=1)  # 95th percentile

# Profile likelihood for statistical inference
profile_llh = numerax.stats.make_profile_llh(
    llh_fn=my_likelihood,
    is_nuisance=[False, True, True],  # mask for parameters
    get_initial_nuisance=lambda params: jnp.array([1.0, 0.0])
)

# All functions work with JAX transformations
grad_fn = jax.grad(numerax.special.gammap_inverse)
sensitivity = grad_fn(0.5, 2.0)
```

## 📋 Requirements

- Python ≥ 3.12
- JAX
- jaxtyping  
- optax

## 🔗 Documentation

- **[📖 Full Documentation](https://juehang.github.io/numerax/)**
- **[API Reference](https://juehang.github.io/numerax/api/)**
- **[GitHub Repository](https://github.com/juehang/numerax)**

## 🏗️ Architecture

numerax follows JAX's functional programming paradigms and is organized into three main modules:

- **`numerax.special`** - Mathematical special functions with custom derivatives
- **`numerax.stats`** - Statistical computation tools for inference problems  
- **`numerax.utils`** - Development utilities for JAX function creation

All functions are designed for:
- **Differentiability** - Full gradient support through custom JVP rules
- **Performance** - JIT compilation and vectorized operations
- **Stability** - Numerically robust algorithms with adaptive precision
- **Compatibility** - Seamless integration with JAX's ecosystem

## 🙏 Acknowledgements

This work is supported by the Department of Energy AI4HEP program.

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Full Changelog**: https://github.com/juehang/numerax/compare/0.3.0...1.0.0