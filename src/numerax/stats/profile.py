"""Profile likelihood functions for statistical inference."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from optax import lbfgs

_DEFAULT_OPTIMIZER = lbfgs()


def make_profile_llh(
    llh_fn: Callable,
    is_nuisance: list[bool] | jnp.ndarray,
    get_initial_nuisance: Callable,
    tol: float = 1e-6,
    initial_value: float = 1e-9,
    initial_diff: float = 1e9,
    optimizer: optax.GradientTransformation = _DEFAULT_OPTIMIZER,
) -> Callable:
    r"""
    Factory function for creating profile likelihood functions.

    ## Overview

    Profile likelihood is a statistical technique used when dealing with
    nuisance parameters that are not of primary interest but are necessary
    for the model. This function creates an optimized profile likelihood
    that maximizes over nuisance parameters while keeping inference
    parameters fixed.

    ## Mathematical Background

    Given a likelihood function $L(\boldsymbol{\theta}, \boldsymbol{\lambda})$
    where $\boldsymbol{\theta}$ are parameters of interest and
    $\boldsymbol{\lambda}$ are nuisance parameters, the profile likelihood is:

    $$L_p(\boldsymbol{\theta}) = \max_{\boldsymbol{\lambda}}
    L(\boldsymbol{\theta}, \boldsymbol{\lambda})$$

    In practice, we work with the log-likelihood
    $\ell(\boldsymbol{\theta}, \boldsymbol{\lambda}) =
    \log L(\boldsymbol{\theta}, \boldsymbol{\lambda})$:

    $$\ell_p(\boldsymbol{\theta}) = \max_{\boldsymbol{\lambda}}
    \ell(\boldsymbol{\theta}, \boldsymbol{\lambda})$$

    This function uses L-BFGS optimization to find the maximum likelihood
    estimates of nuisance parameters for each fixed value of inference
    parameters.

    ## Args

    - **llh_fn**: Log likelihood function taking (params, *args) and
      returning scalar log likelihood value
    - **is_nuisance**: Boolean array where True indicates nuisance
      parameters and False indicates inference parameters
    - **get_initial_nuisance**: Function taking (*args) and returning
      initial values for nuisance parameters
    - **tol**: Convergence tolerance for the optimization (default: 1e-6)
    - **initial_value**: Initial objective value for convergence tracking
      (default: 1e-9)
    - **initial_diff**: Initial difference for convergence tracking
      (default: 1e9)
    - **optimizer**: Optax optimizer to use for maximization
      (default: lbfgs()). Currently tested only with the default L-BFGS
      optimizer

    ## Returns

    Profile likelihood function with signature:
    `(inference_values, *args) -> (profile_llh_value, optimal_nuisance,
    convergence_diff, num_iterations)`

    ## Example

    Consider fitting a normal distribution where we want to infer the mean
    $\mu$ but treat the variance $\sigma^2$ as a nuisance parameter:

    ```python
    import jax.numpy as jnp
    import numerax

    # Sample data
    data = jnp.array([1.2, 0.8, 1.5, 0.9, 1.1, 1.3, 0.7, 1.4])


    # Log likelihood for normal distribution
    def normal_llh(params, data):
        mu, log_sigma = params  # Use log(sigma) for numerical stability
        sigma = jnp.exp(log_sigma)
        return jnp.sum(
            -0.5 * jnp.log(2 * jnp.pi)
            - log_sigma
            - 0.5 * ((data - mu) / sigma) ** 2
        )


    # Profile over log_sigma (nuisance), infer mu
    is_nuisance = [False, True]  # mu=inference, log_sigma=nuisance


    def get_initial_log_sigma(data):
        # Initialize with log of sample standard deviation
        return jnp.array([jnp.log(jnp.std(data))])


    profile_llh = numerax.stats.make_profile_llh(
        normal_llh, is_nuisance, get_initial_log_sigma
    )

    # Evaluate profile likelihood at different mu values
    mu_test = 1.0
    llh_val, opt_log_sigma, diff, n_iter = profile_llh(
        jnp.array([mu_test]), data
    )
    ```

    ## Notes

    - The function is JIT-compiled for performance
    - Uses L-BFGS optimization which is well-suited for smooth likelihood
      surfaces
    - Returns convergence information for diagnostics
    - Handles parameter masking automatically
    - Consider using log-parameterization for positive parameters
      (e.g., $\log \sigma$) to improve numerical stability
    - This function might not work well if the likelihood surface has
      multiple local maxima; in such cases, consider ensuring that
      initial guesses are close to the global maximum.
    """
    nuisance_mask = jnp.array(is_nuisance)
    inference_mask = ~nuisance_mask

    @jax.jit
    def profile_llh(inference_values, *args):
        solver = optimizer
        initial_nuisance = get_initial_nuisance(*args)
        opt_state = solver.init(initial_nuisance)

        def objective(nuisance_params):
            # Reconstruct full parameter vector
            full_params = jnp.zeros(len(nuisance_mask))
            full_params = full_params.at[inference_mask].set(inference_values)
            full_params = full_params.at[nuisance_mask].set(nuisance_params)
            return -llh_fn(full_params, *args)

        value_and_grad = optax.value_and_grad_from_state(objective)

        def profile_llh_loopfun(var):
            params, last_value, opt_state, _, n = var
            value, grad = value_and_grad(params, state=opt_state)
            updates, opt_state = solver.update(
                grad,
                opt_state,
                params,
                value=value,
                grad=grad,
                value_fn=objective,
            )
            params = optax.apply_updates(params, updates)
            diff = last_value - value
            return params, value, opt_state, diff, n + 1

        params, value, opt_state, diff, n = jax.lax.while_loop(
            lambda var: jnp.abs(var[-2]) > jnp.abs(var[1] * tol),
            profile_llh_loopfun,
            (initial_nuisance, initial_value, opt_state, initial_diff, 0),
        )

        return -value, params, diff, n

    return profile_llh
