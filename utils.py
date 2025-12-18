# galerkin.py (Corrected Version)

import jax
import jax.numpy as jnp
from functools import partial
from jax.flatten_util import ravel_pytree

# Import from our own modules
# (确保这些导入路径正确)
from nn import ShallowNetKdV 
# from physics import kdv_spatial_residual

def get_unravel_fn(model, dummy_input):
    """Get the unravel function for converting flat parameters to PyTree."""
    _, unravel_fn = ravel_pytree(model.init(jax.random.PRNGKey(0), dummy_input)['params'])
    return unravel_fn

def compute_jacobian_matrix(model_apply_fn, params, particles):
    """
    Computes the Jacobian of the network output with respect to the parameters,
    and returns it as a dense matrix of shape (num_particles, num_params).
    """
    # 1. 定义一个只接受参数（作为PyTree）和单个粒子的函数
    def single_forward(p, x):
        return model_apply_fn({'params': p}, x.reshape(1, -1)).squeeze()

    # 2. 对这个函数求梯度，得到一个返回PyTree梯度的函数
    grad_fn_tree = jax.grad(single_forward)

    # 3. 使用vmap计算所有粒子的梯度PyTree
    J_tree = jax.vmap(grad_fn_tree, in_axes=(None, 0))(params, particles)

    # 4. 将梯度的PyTree压平成一个 (num_particles, num_params) 的矩阵
    #    我们先将每个叶子节点压平，然后拼接起来
    leaves, _ = jax.tree_util.tree_flatten(J_tree)
    J_matrix = jnp.concatenate([leaf.reshape(particles.shape[0], -1) for leaf in leaves], axis=1)
    
    # 5. 安全网
    J_matrix = jnp.where(jnp.isnan(J_matrix), 0.0, J_matrix)
    return J_matrix

def assemble_M(model_apply_fn, params, particles, ridge_lambda=0.0):
    """Assembles the mass matrix M = <J, J>."""
    J = compute_jacobian_matrix(model_apply_fn, params, particles)
    num_particles = particles.shape[0]
    
    M = jnp.dot(J.T, J) / num_particles
    M_reg = M + ridge_lambda * jnp.eye(M.shape[1])
    return M_reg

def assemble_F(model_apply_fn, params, spatial_residual_fn, particles, t):
    """Assembles the force vector F = -<J, f>."""
    J = compute_jacobian_matrix(model_apply_fn, params, particles)
    num_particles = particles.shape[0]

    # `spatial_residual_fn` expects (model_apply_fn, params, xs) or (model_apply_fn, params, xs, t)
    # Check if spatial_residual_fn needs time parameter by checking its signature
    import inspect
    sig = inspect.signature(spatial_residual_fn)
    num_params = len(sig.parameters)
    
    if num_params == 4:  # AC equation: (model_apply_fn, params, xs, t)
        def single_residual_fn(p, x):
            return spatial_residual_fn(model_apply_fn, p, x, t)
    else:  # KdV equation: (model_apply_fn, params, xs)
        def single_residual_fn(p, x):
            return spatial_residual_fn(model_apply_fn, p, x)

    f_vals = jax.vmap(single_residual_fn, in_axes=(None, 0))(params, particles.squeeze())
    f_vals = jnp.where(jnp.isnan(f_vals), 0.0, f_vals)
    
    F = jnp.dot(J.T, f_vals) / num_particles
    return -F

def compute_residual_for_sampling(model_apply_fn, params, spatial_residual_fn, theta_dot_flat, particles, t):
    """Computes the full PDE residual R = u_t + f(u) for SVGD sampling."""
    J = compute_jacobian_matrix(model_apply_fn, params, particles)
    
    # Add numerical stability checks
    J = jnp.where(jnp.isnan(J), 0.0, J)
    J = jnp.where(jnp.isinf(J), jnp.sign(J) * 1e6, J)
    
    u_t = jnp.dot(J, theta_dot_flat)
    
    # Check if spatial_residual_fn needs time parameter
    import inspect
    sig = inspect.signature(spatial_residual_fn)
    num_params = len(sig.parameters)
    
    if num_params == 4:  # AC equation: (model_apply_fn, params, xs, t)
        def single_residual_fn(p, x):
            result = spatial_residual_fn(model_apply_fn, p, x, t)
            # Numerical stability
            result = jnp.where(jnp.isnan(result), 0.0, result)
            result = jnp.where(jnp.isinf(result), jnp.sign(result) * 1e6, result)
            return result
    else:  # KdV equation: (model_apply_fn, params, xs)
        def single_residual_fn(p, x):
            result = spatial_residual_fn(model_apply_fn, p, x)
            # Numerical stability
            result = jnp.where(jnp.isnan(result), 0.0, result)
            result = jnp.where(jnp.isinf(result), jnp.sign(result) * 1e6, result)
            return result

    # Ensure particles is at least 1D for vmap
    particles_1d = jnp.atleast_1d(particles.squeeze())
    f_vals = jax.vmap(single_residual_fn, in_axes=(None, 0))(params, particles_1d)
    
    residual = u_t + f_vals
    
    # Final numerical stability check
    residual = jnp.where(jnp.isnan(residual), 0.0, residual)
    residual = jnp.where(jnp.isinf(residual), jnp.sign(residual) * 1e6, residual)
    
    return residual