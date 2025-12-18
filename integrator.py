# integrators.py

import jax
import jax.numpy as jnp
from functools import partial
from jax.flatten_util import ravel_pytree

# Import from our own modules
from utils import assemble_M, assemble_F

def estimate_theta_dot(model_apply_fn, params, particles, ridge_lambda, spatial_residual_fn, t):
    """
    Estimates the current time derivative of theta (θ̇) by solving M θ̇ = F.
    Supports both KdV (spatial_residual_fn with 3 args) and AC (4 args including t) equations.
    """
    M = assemble_M(model_apply_fn, params, particles, ridge_lambda)
    # 注意：我们的assemble_F已经包含了负号，所以 F = -<J, f>
    # assemble_F will automatically detect if spatial_residual_fn needs time parameter
    F = assemble_F(model_apply_fn, params, spatial_residual_fn, particles, t)
    
    # 使用 jnp.linalg.solve 因为 M 应该被正则化得很好
    # 如果仍然担心，可以换回 lstsq
    theta_dot_flat = jnp.linalg.lstsq(M, F)[0]
    
    return theta_dot_flat

def rk4_step(model_apply_fn, params, particles, dt, ridge_lambda, spatial_residual_fn, t):
    """
    Performs a single time step using the 4th-order Runge-Kutta method.
    Supports both KdV (spatial_residual_fn with 3 args) and AC (4 args including t) equations.
    """
    # 压平参数以进行计算
    params_flat, unravel_fn = ravel_pytree(params)

    def theta_dynamics(p_flat, current_t):
        p_tree = unravel_fn(p_flat)
        # 估算 θ̇ (压平的)
        # estimate_theta_dot will handle both KdV and AC residual functions
        return estimate_theta_dot(
            model_apply_fn, p_tree, particles, ridge_lambda, spatial_residual_fn, current_t
        )

    # RK4 stages
    k1 = theta_dynamics(params_flat, t)
    k2 = theta_dynamics(params_flat + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = theta_dynamics(params_flat + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = theta_dynamics(params_flat + dt * k3, t + dt)
    
    # Gradient clipping for RK4 stages - 更严格的限制
    max_grad_norm = 50.0  # 从 100.0 降低到 50.0
    k1 = jnp.clip(k1, -max_grad_norm, max_grad_norm)
    k2 = jnp.clip(k2, -max_grad_norm, max_grad_norm)
    k3 = jnp.clip(k3, -max_grad_norm, max_grad_norm)
    k4 = jnp.clip(k4, -max_grad_norm, max_grad_norm)
    
    # 检查每个 k 是否有 NaN/Inf
    k1 = jnp.where(jnp.isnan(k1) | jnp.isinf(k1), 0.0, k1)
    k2 = jnp.where(jnp.isnan(k2) | jnp.isinf(k2), 0.0, k2)
    k3 = jnp.where(jnp.isnan(k3) | jnp.isinf(k3), 0.0, k3)
    k4 = jnp.where(jnp.isnan(k4) | jnp.isinf(k4), 0.0, k4)
    
    # Update flattened parameters
    new_params_flat = params_flat + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    
    # Parameter regularization: prevent parameters from growing too large
    param_norm = jnp.linalg.norm(new_params_flat)
    max_param_norm = 500.0  # 从 1000.0 降低到 500.0
    if param_norm > max_param_norm:
        new_params_flat = new_params_flat * max_param_norm / param_norm
    
    # NaN detection and recovery for parameters
    if jnp.any(jnp.isnan(new_params_flat)) or jnp.any(jnp.isinf(new_params_flat)):
        print("Warning: NaN/Inf detected in parameters, reverting to previous state")
        new_params_flat = params_flat
    
    # 返回更新后的 PyTree
    return unravel_fn(new_params_flat)