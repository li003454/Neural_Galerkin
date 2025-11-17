# physics.py (Hardened Version)

import jax
import jax.numpy as jnp
from functools import partial

from nn import ShallowNetKdV 

@partial(jax.jit, static_argnames=('model_apply_fn',))
def _scalar_u(model_apply_fn, params, x):
    """Helper function to get a scalar output from the network."""
    # x should be a single scalar value
    x_input = jnp.array([[x]]) if jnp.ndim(x) == 0 else jnp.atleast_2d(x)
    result = model_apply_fn({'params': params}, x_input)
    return jnp.squeeze(result)

@partial(jax.jit, static_argnames=('model_apply_fn',))
def dx_forward(model_apply_fn, params, xs):
    """Computes u_x for a batch of points xs with NaN protection."""
    def u_fn(x_scalar):
        return _scalar_u(model_apply_fn, params, x_scalar)
    
    # Ensure xs is at least 1D for vmap
    xs_1d = jnp.atleast_1d(xs.squeeze())
    ux_vals = jax.vmap(jax.grad(u_fn))(xs_1d)
    # +++ 安全网 +++
    return jnp.where(jnp.isnan(ux_vals), 0.0, ux_vals)

@partial(jax.jit, static_argnames=('model_apply_fn',))
def d3x_forward(model_apply_fn, params, xs):
    """Computes u_xxx for a batch of points xs with NaN protection."""
    def u_fn(x_scalar):
        return _scalar_u(model_apply_fn, params, x_scalar)
    
    d1_fn = jax.grad(u_fn)
    d2_fn = jax.grad(d1_fn)
    d3_fn = jax.grad(d2_fn)
    
    # Ensure xs is at least 1D for vmap
    xs_1d = jnp.atleast_1d(xs.squeeze())
    uxxx_vals = jax.vmap(d3_fn)(xs_1d)
    # +++ 安全网 +++
    return jnp.where(jnp.isnan(uxxx_vals), 0.0, uxxx_vals)

@partial(jax.jit, static_argnames=('model_apply_fn',))
def kdv_spatial_residual(model_apply_fn, params, xs):
    """
    Computes the spatial part of the KdV residual f(u) = 6*u*u_x + u_xxx
    with NaN protection at every step.
    """
    xs = jnp.atleast_2d(xs)
    
    # --- 计算 u ---
    # vmap a simplified forward pass for u_vals
    u_vals = jax.vmap(partial(_scalar_u, model_apply_fn), in_axes=(None, 0))(params, xs)
    u_vals = jnp.where(jnp.isnan(u_vals), 0.0, u_vals) # <-- 安全网 1

    # --- 计算 ux ---
    ux_vals = dx_forward(model_apply_fn, params, xs) # <-- dx_forward 内部已经有安全网

    # --- 计算 uxxx ---
    uxxx_vals = d3x_forward(model_apply_fn, params, xs) # <-- d3x_forward 内部已经有安全网

    # --- 计算最终残差 ---
    residual = 6.0 * u_vals * ux_vals + uxxx_vals
    # +++ 最终安全网 +++
    return jnp.where(jnp.isnan(residual), 0.0, residual).squeeze()