# exact_solutions.py

import jax.numpy as jnp

def kdv_two_soliton(x, t):
    """
    Exact two-soliton solution for the Korteweg-de Vries (KdV) equation.
    Function taken from https://github.com/pehersto/ng/solvers/exactKdV.py
    
    Args:
        x: jnp.array, points in space. Can be a vector.
        t: float or jnp.array, a single point in time.
    """
    # Parameters for the two-soliton solution
    k = jnp.asarray([jnp.sqrt(5.), 1.]) # Note: Swapped order to match common visualizations (taller soliton on left)
    eta_0 = jnp.asarray([10.73, 0.])
    
    # Ensure inputs are correctly shaped for broadcasting
    x = jnp.asarray(x)
    t = jnp.asarray(t)
    
    # Calculate phases
    eta1 = k[0] * x.reshape(-1, 1) - k[0]**3 * t.reshape(1, -1) + eta_0[0]
    eta2 = k[1] * x.reshape(-1, 1) - k[1]**3 * t.reshape(1, -1) + eta_0[1]
    
    # Interaction term
    c = ((k[0] - k[1]) / (k[0] + k[1]))**2
    
    # Solution components
    f = 1. + jnp.exp(eta1) + jnp.exp(eta2) + c * jnp.exp(eta1 + eta2)
    df_dx = k[0] * jnp.exp(eta1) + k[1] * jnp.exp(eta2) + c * (k[0] + k[1]) * jnp.exp(eta1 + eta2)
    ddf_dxx = k[0]**2 * jnp.exp(eta1) + k[1]**2 * jnp.exp(eta2) + c * (k[0] + k[1])**2 * jnp.exp(eta1 + eta2)
    
    # Full solution formula
    u = 2 * (f * ddf_dxx - df_dx**2) / f**2
    
    # Handle potential numerical issues and return squeezed result
    u = jnp.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
    return jnp.squeeze(u)