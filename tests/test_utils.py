"""Tests for utility functions in numerax.utils."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from numerax.utils import count_params


@pytest.mark.parametrize(
    ("model", "expected_count", "description"),
    [
        # Simple structures
        (
            {"weights": jnp.ones((10, 5)), "bias": jnp.zeros(5)},
            55,
            "simple_dict",
        ),
        (
            {
                "layer1": {
                    "weights": jnp.ones((10, 20)),
                    "bias": jnp.zeros(20),
                },
                "layer2": {
                    "weights": jnp.ones((20, 5)),
                    "bias": jnp.zeros(5),
                },
            },
            325,
            "nested_dict",
        ),
        # Empty and edge cases
        ({}, 0, "empty_dict"),
        (
            {"metadata": "info", "config": {"lr": 0.01}},
            0,
            "no_arrays",
        ),
        # Mixed types
        (
            {
                "arrays": {"a": jnp.ones(10), "b": jnp.zeros(5)},
                "scalars": {"x": 1.0, "y": 2.0},
                "strings": {"name": "model"},
            },
            15,
            "mixed_types",
        ),
        # List structures
        (
            [jnp.ones((5, 5)), jnp.zeros((3, 3)), jnp.ones(10)],
            44,
            "list_of_arrays",
        ),
    ],
)
def test_count_params_structures(model, expected_count, description):
    """Test parameter counting on various PyTree structures."""
    count = count_params(model, verbose=False)
    assert count == expected_count, f"Failed for {description}"


def test_count_params_custom_filter():
    """Test parameter counting with custom filter function."""
    model = {
        "weights": jnp.ones((10, 5)),
        "bias": jnp.zeros(5),
        "metadata": "not an array",
    }
    # Only count 2D arrays
    count = count_params(
        model,
        filter=lambda x: hasattr(x, "ndim") and x.ndim > 1,
        verbose=False,
    )
    assert count == 50  # Only the weights matrix


@pytest.mark.parametrize("verbose", [True, False])
def test_count_params_verbose_modes(verbose, capsys):
    """Test verbose and non-verbose output modes."""
    model = {"weights": jnp.ones((10, 5))}
    count = count_params(model, verbose=verbose)

    captured = capsys.readouterr()
    if verbose:
        assert "Number of parameters: 5.0e+01" in captured.out
    else:
        assert captured.out == ""
    assert count == 50


@pytest.mark.parametrize(
    ("has_non_arrays", "expected_count"),
    [(False, 55), (True, 55)],
)
def test_count_params_equinox_modules(has_non_arrays, expected_count):
    """Test parameter counting with Equinox modules."""
    if has_non_arrays:

        class Model(eqx.Module):
            weight: jax.Array
            bias: jax.Array
            name: str

            def __init__(self, key):
                wkey, bkey = jax.random.split(key)
                self.weight = jax.random.normal(wkey, (10, 5))
                self.bias = jax.random.normal(bkey, (5,))
                self.name = "test_model"

    else:

        class Model(eqx.Module):
            weight: jax.Array
            bias: jax.Array

            def __init__(self, key):
                wkey, bkey = jax.random.split(key)
                self.weight = jax.random.normal(wkey, (10, 5))
                self.bias = jax.random.normal(bkey, (5,))

    model = Model(jax.random.PRNGKey(0))
    count = count_params(model, verbose=False)
    assert count == expected_count
