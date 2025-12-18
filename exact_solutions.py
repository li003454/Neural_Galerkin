# exact_solutions.py

import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp


# KdV equation

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


# Allen-Cahn equation

def exactAC():
    """
    Compute the exact solution for the Allen-Cahn equation using numerical integration.
    
    Returns:
        t: numpy array, time points
        u: numpy array, solution at each time point (shape: [N, len(t)])
    """
    N = 1024  # 2048
    x = np.linspace(0, 2 * np.pi, N)
    L = 2 * np.pi
    h = L / N
    T = 12
    
    epsilon = 5e-2
    a = lambda x, t: (1.05 + t * np.sin(x))
    phi = lambda x, w, b: np.exp(- w ** 2 * np.abs(np.sin(np.pi * (x - b) / L)) ** 2)
    u0 = phi(x, np.sqrt(10), 0.5) - phi(x, np.sqrt(10), 4.4)
    
    A = - 2 * np.eye(N) + np.eye(N, k=1) + np.eye(N, k=-1)
    # periodic boundary conditions
    A[0, -2] = 1
    A[-1, 1] = 1
    
    rhs = lambda t, u: epsilon * A @ u / (h ** 2) + a(x, t) * (u - u ** 3)
    
    res = solve_ivp(rhs, [0, T], u0, method='BDF', max_step=0.01)
    
    return res.t, res.y  # t, u


# Cache the exact solution to avoid recomputing
_ac_t_full = None
_ac_u_full = None
_ac_x_full = None

def _get_ac_exact_solution():
    """Get or compute the cached exact AC solution."""
    global _ac_t_full, _ac_u_full, _ac_x_full
    if _ac_t_full is None:
        _ac_t_full, _ac_u_full = exactAC()
        _ac_x_full = np.linspace(0, 2 * np.pi, _ac_u_full.shape[0])
    return _ac_t_full, _ac_u_full, _ac_x_full

def ac_solution(x, t):
    """
    Allen-Cahn equation solution at given spatial points and time.
    This function interpolates from the precomputed exact solution.
    
    Args:
        x: jnp.array or np.array, points in space. Can be a vector.
        t: float or jnp.array, a single point in time.
    
    Returns:
        u: jnp.array, solution values at the given (x, t) points.
    """
    # Convert to numpy for interpolation
    x = np.asarray(x)
    t = np.asarray(t)
    
    # Get the cached full solution
    t_full, u_full, x_full = _get_ac_exact_solution()
    
    # Find the closest time index
    if t.ndim == 0:  # scalar time
        t_idx = np.argmin(np.abs(t_full - t))
        u_at_t = u_full[:, t_idx]
    else:  # array of times
        u_at_t = np.zeros((len(x), len(t)))
        for i, t_val in enumerate(t):
            t_idx = np.argmin(np.abs(t_full - t_val))
            u_at_t[:, i] = u_full[:, t_idx]
    
    # Interpolate in space
    if t.ndim == 0:  # scalar time
        u_interp = np.interp(x, x_full, u_at_t)
    else:  # array of times
        u_interp = np.zeros((len(x), len(t)))
        for i in range(len(t)):
            u_interp[:, i] = np.interp(x, x_full, u_at_t[:, i])
    
    # Convert back to JAX array
    return jnp.asarray(jnp.squeeze(u_interp))