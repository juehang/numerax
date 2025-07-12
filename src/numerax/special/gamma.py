import jax
import jax.numpy as jnp
import jax.scipy.special as special
from jaxtyping import ArrayLike

# Global constants for numerical stability - adapt to JAX precision setting
_DTYPE = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
TINY = jnp.finfo(_DTYPE).smallest_normal  # For preventing underflow
EPS = jnp.finfo(_DTYPE).eps  # For convergence tolerance (machine epsilon)


@jax.custom_jvp
def gammap_inverse(p: ArrayLike, a: float) -> ArrayLike:
    """
    Inverse of regularized incomplete gamma function using Halley's method.

    Finds x such that gammainc(a, x) = p.
    This is equivalent to the inverse CDF of Gamma(a, 1) distribution.
    Initial guess based on Numerical Recipes (Press et al., 2007).

    Args:
        p: Probability values in [0, 1]
        a: Shape parameter (positive)

    Returns:
        Quantiles where gammainc(a, x) = p
    """

    def objective(x):
        """F(x) = gammainc(a, x) - p"""
        return special.gammainc(a, x) - p

    # Initial guess from Numerical Recipes
    def initial_guess(u_val, a_val):
        # a = dof/2 for chi-squared

        def large_a_guess():
            # For a > 1: use Wilson-Hilferty approximation
            pp = jnp.where(u_val < 0.5, u_val, 1.0 - u_val)
            t = jnp.sqrt(-2.0 * jnp.log(pp))
            x = (2.30753 + t * 0.27061) / (
                1.0 + t * (0.99229 + t * 0.04481)
            ) - t
            x = jnp.where(u_val < 0.5, -x, x)
            return jnp.fmax(
                1e-3,
                a_val
                * (1.0 - 1.0 / (9.0 * a_val) - x / (3.0 * jnp.sqrt(a_val)))
                ** 3,
            )

        def small_a_guess():
            # For a <= 1: use equations (6.2.8) and (6.2.9)
            t = 1.0 - a_val * (0.253 + a_val * 0.12)
            return jnp.where(
                u_val < t,
                (u_val / t) ** (1.0 / a_val),
                1.0 - jnp.log(1.0 - (u_val - t) / (1.0 - t)),
            )

        return jnp.real(
            jnp.where(a_val > 1.0, large_a_guess(), small_a_guess())
        )

    # Derivatives for Halley's method
    f = objective
    df_dx = jax.grad(objective)
    d2f_dx2 = jax.grad(df_dx)

    x = initial_guess(p, a)

    # Use while_loop for dynamic convergence
    def cond_fn(state):
        x, step, iteration = state
        # Continue while step is large and we haven't exceeded max iterations
        return (jnp.abs(step) > EPS * jnp.abs(x)) & (iteration < 12)

    def body_fn(state):
        x, _, iteration = state

        f_val = f(x)
        df_val = df_dx(x)
        d2f_val = d2f_dx2(x)

        # Halley's method: x_{n+1} = x_n - 2*f*f' / (2*f'^2 - f*f'')
        numerator = 2 * f_val * df_val
        denominator = 2 * df_val**2 - f_val * d2f_val

        # Avoid division by zero and ensure step is reasonable
        denominator = jnp.where(
            jnp.abs(denominator) < TINY,
            jnp.sign(denominator) * TINY,
            denominator,
        )

        step = numerator / denominator
        x_new = x - step

        # Ensure x stays positive
        x_new = jnp.fmax(x_new, TINY)

        return (x_new, step, iteration + 1)

    # Initial state: (x, step, iteration)
    initial_state = (x, jnp.inf, 0)
    final_state = jax.lax.while_loop(cond_fn, body_fn, initial_state)
    x = final_state[0]

    return x


@gammap_inverse.defjvp
def gammap_inverse_jvp(primals, tangents):
    """
    Custom JVP for gammap_inverse using implicit function theorem.

    For F(x, p) = gammainc(a, x) - p = 0:
    dx/dp = -∂F/∂p / ∂F/∂x = 1 / (∂/∂x gammainc(a, x))
    """
    p, a = primals
    p_dot, _ = tangents

    # Forward pass
    x = gammap_inverse(p, a)

    # Compute derivative: dx/dp = 1 / (d/dx gammainc(a, x))
    def gammainc_x(x_val):
        return special.gammainc(a, x_val)

    dgammainc_dx = jax.grad(gammainc_x)(x)

    # Avoid division by zero
    dgammainc_dx = jnp.where(
        jnp.abs(dgammainc_dx) < TINY,
        jnp.sign(dgammainc_dx) * TINY,
        dgammainc_dx,
    )

    dx_dp = 1.0 / dgammainc_dx

    # For now, ignore a derivatives (could be added if needed)
    x_dot = dx_dp * p_dot

    return x, x_dot
