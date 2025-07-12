"""Test profile likelihood with Gaussian data and analytical validation."""

import jax
import jax.numpy as jnp

from numerax.stats import make_profile_llh


def test_gaussian_profile_analytical_validation():
    r"""Test profile likelihood against analytical Gaussian MLE solution.

    For Gaussian data with fixed $\mu$, the MLE of $\sigma$ is:
    $\hat{\sigma}(\mu) = \sqrt{\Sigma(x_i-\mu)^2/N}$
    This test validates that our profile likelihood optimization finds the same
    result as this analytical solution.
    """
    # Set random seed for reproducible test
    key = jax.random.PRNGKey(42)

    # Generate synthetic Gaussian data
    true_mu = 2.0
    true_sigma = 1.5
    n_samples = 50

    data = jax.random.normal(key, (n_samples,)) * true_sigma + true_mu

    # Define normal log likelihood function
    def normal_llh(params, data):
        mu, log_sigma = params
        sigma = jnp.exp(log_sigma)
        return jnp.sum(
            -0.5 * jnp.log(2 * jnp.pi)
            - log_sigma
            - 0.5 * ((data - mu) / sigma) ** 2
        )

    # Set up profile likelihood: mu=inference, log_sigma=nuisance
    is_nuisance = [False, True]  # mu=inference, log_sigma=nuisance

    def get_initial_log_sigma(data):
        # Initialize with log of sample standard deviation
        return jnp.array([jnp.log(jnp.std(data))])

    # Create profile likelihood function
    profile_llh = make_profile_llh(
        normal_llh, is_nuisance, get_initial_log_sigma
    )

    # Test at a fixed mu value
    test_mu = 1.8
    llh_val, opt_log_sigma, diff, n_iter = profile_llh(
        jnp.array([test_mu]), data
    )

    # Extract optimized sigma
    sigma_opt = jnp.exp(opt_log_sigma[0])

    # Calculate analytical solution: sigma_hat(mu) = sqrt(sum((x_i-mu)^2)/N)
    sigma_analytical = jnp.sqrt(jnp.mean((data - test_mu) ** 2))

    # Validate convergence
    assert n_iter > 0, "Optimization should have run at least one iteration"
    assert jnp.abs(diff) <= jnp.abs(llh_val * 1e-6), (
        f"Should have converged, but diff={diff}"
    )

    # Validate against analytical solution
    assert jnp.allclose(sigma_opt, sigma_analytical, atol=1e-3), (
        f"Optimized sigma={sigma_opt:.6f} should match "
        f"analytical sigma={sigma_analytical:.6f}"
    )

    # Validate likelihood value is finite and reasonable
    assert jnp.isfinite(llh_val), "Profile likelihood value should be finite"
    assert llh_val < 0, "Log likelihood should be negative"


def test_gaussian_profile_at_true_mean():
    """Test profile likelihood when mu equals the true data-generating mean."""
    key = jax.random.PRNGKey(123)

    # Generate data
    true_mu = 3.0
    true_sigma = 0.8
    n_samples = 30

    data = jax.random.normal(key, (n_samples,)) * true_sigma + true_mu

    def normal_llh(params, data):
        mu, log_sigma = params
        sigma = jnp.exp(log_sigma)
        return jnp.sum(
            -0.5 * jnp.log(2 * jnp.pi)
            - log_sigma
            - 0.5 * ((data - mu) / sigma) ** 2
        )

    is_nuisance = [False, True]

    def get_initial_log_sigma(data):
        return jnp.array([jnp.log(jnp.std(data))])

    profile_llh = make_profile_llh(
        normal_llh, is_nuisance, get_initial_log_sigma
    )

    # Test at sample mean (should be close to optimal)
    sample_mean = jnp.mean(data)
    llh_val, opt_log_sigma, _, _ = profile_llh(jnp.array([sample_mean]), data)

    sigma_opt = jnp.exp(opt_log_sigma[0])
    sigma_analytical = jnp.sqrt(jnp.mean((data - sample_mean) ** 2))

    # Should converge quickly and match analytical solution well
    assert jnp.allclose(sigma_opt, sigma_analytical, atol=1e-4)
    assert jnp.isfinite(llh_val)
