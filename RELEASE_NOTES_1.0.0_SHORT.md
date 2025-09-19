# numerax 1.0.0 - First Stable Release! 🎉

The first stable release of **numerax** is here! A comprehensive suite of statistical and numerical computation functions for JAX.

## ✨ What's New

### Special Functions
- **`gammap_inverse(p, a)`** - Inverse gamma function with custom gradients and Halley's method
- **`erfcinv(x)`** - Inverse complementary error function, fully differentiable

### Statistical Tools  
- **Chi-squared distribution** - Complete interface with high-precision PPF
- **Profile likelihood** - Factory function with L-BFGS optimization for statistical inference

### Utilities
- **`preserve_metadata()`** - Decorator wrapper for maintaining documentation with JAX

## 🚀 Key Features

- **Full JAX compatibility** - JIT, grad, vmap support for all functions
- **Custom derivatives** - Exact gradients via JVP rules for numerical stability  
- **High performance** - Vectorized operations with adaptive precision
- **Comprehensive testing** - Validated against SciPy references

## 📦 Installation

```bash
pip install numerax
```

## 💡 Quick Example

```python
import jax.numpy as jnp
import numerax

# Special functions with gradients
x = numerax.special.gammap_inverse(0.5, 2.0)
y = numerax.special.erfcinv(0.5)

# Statistical distributions
quantile = numerax.stats.chi2.ppf(0.95, df=1)

# Profile likelihood for inference
profile_llh = numerax.stats.make_profile_llh(
    llh_fn=my_likelihood,
    is_nuisance=[False, True, True]
)
```

**[📖 Documentation](https://juehang.github.io/numerax/)** | **[Full Release Notes](RELEASE_NOTES_1.0.0.md)**

**Full Changelog**: https://github.com/juehang/numerax/compare/0.3.0...1.0.0