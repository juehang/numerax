# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-09-19

### Added

#### Special Functions
- `gammap_inverse(p, a)` - Inverse of the regularized incomplete gamma function
  - Halley's method for fast quadratic convergence  
  - Custom JVP implementation for exact gradients
  - Numerical stability with adaptive precision
  - Based on Numerical Recipes methods
- `erfcinv(x)` - Inverse complementary error function
  - Fully differentiable with JAX transformations
  - Vectorized input support

#### Statistical Methods  
- Complete chi-squared distribution interface (`numerax.stats.chi2`)
  - All standard functions (pdf, cdf, sf, logpdf, logcdf, logsf)
  - Custom high-precision percent point function (ppf)
  - Location-scale parameterization support
- `make_profile_llh()` - Factory for profile likelihood functions
  - L-BFGS optimization over nuisance parameters
  - JIT compilation support
  - Configurable tolerance and optimizer settings
  - Parameter masking capabilities

#### Utilities
- `preserve_metadata(decorator)` - Wrapper for preserving function metadata
  - Ensures docstrings survive JAX decorators
  - Critical for documentation generation tools

#### Infrastructure
- Comprehensive test suite with SciPy validation
- Full JAX transformation compatibility (JIT, grad, vmap)
- Documentation with MkDocs and API reference
- CI/CD with GitHub Actions
- MIT license

### Technical Details
- Python ≥ 3.12 requirement
- JAX, jaxtyping, optax dependencies
- Modular architecture with three main modules:
  - `numerax.special` - Mathematical special functions
  - `numerax.stats` - Statistical computation tools  
  - `numerax.utils` - Development utilities

## [0.3.0] - 2025-08-26

### Changed
- Internal refactoring and improvements

## [0.2.0] - 2025-07-14

### Added
- Optimizer parameter for profile likelihood functions

## [0.1.0] - 2025-07-14

### Added
- Initial release with basic functionality

[1.0.0]: https://github.com/juehang/numerax/compare/0.3.0...1.0.0
[0.3.0]: https://github.com/juehang/numerax/compare/0.2.0...0.3.0
[0.2.0]: https://github.com/juehang/numerax/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/juehang/numerax/releases/tag/0.1.0