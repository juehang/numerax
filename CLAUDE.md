# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Numerax is a Python package that provides useful statistics and numerical computation functions based on JAX that are not part of the main JAX API. The project uses modern Python packaging with Hatch as the build system.

## Development Commands

### Testing
- Run tests: `hatch run test`
- Run tests with coverage: `hatch run test-cov`
- Generate coverage report: `hatch run cov-report`
- Run full coverage workflow: `hatch run cov`

### Code Quality
- Check code formatting and linting: `hatch run lint:check`
- Auto-format and fix issues: `hatch run lint:fmt`

### Documentation
- Build documentation: `pdoc --math -o docs/ numerax`
- Documentation is auto-deployed via GitHub Pages at https://juehang.github.io/numerax/

### Build System
- The project uses Hatch with `hatchling` as the build backend
- Version is managed in `src/numerax/__init__.py`
- Runtime dependencies: JAX, optax, and pdoc for numerical computations, optimization, and documentation

### CI/CD
- GitHub Actions workflow for tests: `.github/workflows/test.yml`
- GitHub Actions workflow for docs: `.github/workflows/docs.yml`
- Uses official PyPA Hatch action for reliable environment setup
- Automated testing on push/PR to main branch
- Automatic documentation deployment to GitHub Pages

## Code Architecture

### Package Structure
- `src/numerax/` - Main package directory
- `src/numerax/__init__.py` - Package initialization with version info and stats import
- `src/numerax/stats/` - Statistics module with profile likelihood functionality
- `src/numerax/stats/profile.py` - Profile likelihood implementation (`make_profile_llh`)
- `tests/` - Test directory with analytical validation tests
- `tests/test_profile_gaussian.py` - Gaussian profile likelihood tests with analytical validation

### Key Technical Details
- Python 3.12+ required
- JAX-based numerical computations with JIT compilation
- Uses optax for L-BFGS optimization in profile likelihood
- Uses Ruff for linting and formatting with strict rules
- Line length: 79 characters
- Code coverage tracking enabled with branch coverage

### Code Style
- Ruff configuration includes extensive rule sets (pycodestyle, pyflakes, pyupgrade, flake8-bugbear, etc.)
- Double quotes preferred for strings
- Space-based indentation
- Docstring code formatting enabled

## Development Notes

- Profile likelihood functionality implemented in `numerax.stats.make_profile_llh`
- Uses JAX for JIT compilation and numerical computations
- L-BFGS optimization via optax for maximizing over nuisance parameters
- Comprehensive LaTeX documentation with raw docstrings for pdoc compatibility
- Tests include analytical validation against known mathematical solutions
- Uses relative tolerance for convergence matching internal optimization logic
- Documentation includes mathematical equations rendered with LaTeX
- GitHub Pages deployment provides professional documentation site