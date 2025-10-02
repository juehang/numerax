"""Tests for utility functions in numerax.utils."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from numerax.utils import count_params, tree_summary


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


# Tests for tree_summary


def _make_equinox_model():
    """Helper to create Equinox model for testing."""

    class Model(eqx.Module):
        weight: jax.Array
        bias: jax.Array
        name: str

        def __init__(self, key):
            wkey, bkey = jax.random.split(key)
            self.weight = jax.random.normal(wkey, (10, 5))
            self.bias = jax.random.normal(bkey, (5,))
            self.name = "test_model"

    return Model(jax.random.PRNGKey(0))


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
        # List structures
        (
            [jnp.ones((5, 5)), jnp.zeros((3, 3)), jnp.ones(10)],
            44,
            "list_of_arrays",
        ),
        # Deeply nested
        (
            {
                "encoder": {
                    "layer1": {
                        "weights": jnp.ones((10, 20)),
                        "bias": jnp.zeros(20),
                    },
                    "layer2": {
                        "weights": jnp.ones((20, 30)),
                        "bias": jnp.zeros(30),
                    },
                },
                "decoder": {
                    "layer1": {
                        "weights": jnp.ones((30, 20)),
                        "bias": jnp.zeros(20),
                    },
                    "layer2": {
                        "weights": jnp.ones((20, 10)),
                        "bias": jnp.zeros(10),
                    },
                },
            },
            1680,
            "deeply_nested",
        ),
        # Equinox module
        (_make_equinox_model(), 55, "equinox_module"),
    ],
)
def test_tree_summary_structures(model, expected_count, description):
    """Test tree_summary on various PyTree structures."""
    count = tree_summary(model, verbose=False)
    assert count == expected_count, f"Failed for {description}"


def test_tree_summary_verbose_output(capsys):
    """Test that verbose mode produces formatted output."""
    model = {
        "layer1": {
            "weights": jnp.ones((10, 20)),
            "bias": jnp.zeros(20),
        }
    }

    count = tree_summary(model, verbose=True)

    captured = capsys.readouterr()
    output = captured.out

    # Check for essential components
    assert "PyTree Summary" in output
    assert "Name" in output
    assert "Shape" in output
    assert "Dtype" in output
    assert "Params" in output
    assert "layer1" in output
    assert "weights" in output
    assert "bias" in output
    assert "[10,20]" in output
    assert "[20]" in output
    assert "Total params:" in output
    assert "220" in output  # Total params

    # Verify return value
    assert count == 220


def test_tree_summary_silent_mode(capsys):
    """Test that verbose=False produces no output."""
    model = {"weights": jnp.ones((10, 5))}

    count = tree_summary(model, verbose=False)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert count == 50


def test_tree_summary_custom_is_leaf():
    """Test tree_summary with custom is_leaf treating strings as leaves."""
    # Create a pytree with mixed types including strings
    model = {
        "config": {"name": "model1", "version": "v1.0"},
        "params": {"weights": jnp.ones((10, 5))},
    }

    # Custom is_leaf that treats strings as countable leaves
    def custom_is_leaf(x):
        if isinstance(x, str):
            return True
        return eqx.is_array(x)

    # With custom is_leaf, strings should be treated as leaves
    # This will try to access .shape on strings and fail,
    # demonstrating the custom is_leaf is actually being used
    with pytest.raises(AttributeError):
        tree_summary(model, is_leaf=custom_is_leaf, verbose=False)


def test_tree_summary_max_depth():
    """Test max_depth parameter limits display depth."""
    model = {
        "level1": {
            "level2": {
                "level3": {"weights": jnp.ones((5, 5))},
            }
        }
    }

    # With max_depth=1, should only see level1
    count = tree_summary(model, max_depth=1, verbose=False)
    assert count == 25  # Still counts all params

    # With max_depth=2, should see level1 and level2
    count = tree_summary(model, max_depth=2, verbose=False)
    assert count == 25


def test_tree_summary_list_structure(capsys):
    """Test tree_summary with list/tuple structures."""
    model = [jnp.ones((3, 3)), jnp.zeros((2, 2))]

    count = tree_summary(model, verbose=True)

    captured = capsys.readouterr()
    output = captured.out

    # Check for list index notation
    assert "[0]" in output
    assert "[1]" in output
    assert count == 13  # 9 + 4
