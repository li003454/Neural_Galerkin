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

@partial(jax.jit, static_argnames=('model_apply_fn',))
def d2x_forward(model_apply_fn, params, xs):
    """Computes u_xx for a batch of points xs with NaN protection."""
    def u_fn(x_scalar):
        return _scalar_u(model_apply_fn, params, x_scalar)
    
    d1_fn = jax.grad(u_fn)
    d2_fn = jax.grad(d1_fn)
    
    # Ensure xs is at least 1D for vmap
    xs_1d = jnp.atleast_1d(xs.squeeze())
    uxx_vals = jax.vmap(d2_fn)(xs_1d)
    # +++ 安全网 +++
    return jnp.where(jnp.isnan(uxx_vals), 0.0, uxx_vals)

@partial(jax.jit, static_argnames=('model_apply_fn',))
def ac_spatial_residual(model_apply_fn, params, xs, t):
    """
    Computes the spatial part of the Allen-Cahn residual.
    AC equation: u_t = ε * u_xx + a(x,t) * (u - u^3)
    where ε = 5e-2 and a(x,t) = 1.05 + t * sin(x)
    
    Spatial residual: f(u) = ε * u_xx + a(x,t) * (u - u^3)
    with NaN protection at every step.
    """
    xs = jnp.atleast_2d(xs)
    epsilon = 5e-2
    
    # Compute a(x, t) = 1.05 + t * sin(x)
    xs_1d = jnp.squeeze(xs)
    a_vals = 1.05 + t * jnp.sin(xs_1d)
    
    # --- 计算 u ---
    u_vals = jax.vmap(partial(_scalar_u, model_apply_fn), in_axes=(None, 0))(params, xs)
    u_vals = jnp.where(jnp.isnan(u_vals), 0.0, u_vals)  # 安全网 1
    u_vals = jnp.where(jnp.isinf(u_vals), jnp.sign(u_vals) * 10.0, u_vals)  # 限制 u 的范围
    # 进一步限制 u 的值域，防止 u^3 爆炸
    u_vals = jnp.clip(u_vals, -2.0, 2.0)

    # --- 计算 uxx ---
    uxx_vals = d2x_forward(model_apply_fn, params, xs)  # 内部已有安全网
    uxx_vals = jnp.where(jnp.isinf(uxx_vals), jnp.sign(uxx_vals) * 100.0, uxx_vals)  # 限制 uxx

    # --- 计算最终残差 ---
    # f(u) = ε * u_xx + a(x,t) * (u - u^3)
    nonlinear_term = u_vals - u_vals ** 3
    # 限制非线性项，防止爆炸
    nonlinear_term = jnp.clip(nonlinear_term, -10.0, 10.0)
    
    residual = epsilon * uxx_vals + a_vals * nonlinear_term
    
    # +++ 最终安全网 +++
    residual = jnp.where(jnp.isnan(residual), 0.0, residual)
    residual = jnp.where(jnp.isinf(residual), jnp.sign(residual) * 100.0, residual)
    # 限制残差的绝对值
    residual = jnp.clip(residual, -100.0, 100.0)
    
    return residual.squeeze()