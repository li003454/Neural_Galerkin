# sampler.py

import jax
import jax.numpy as jnp
from functools import partial

# Import from our own modules
from utils import compute_residual_for_sampling
from physics import kdv_spatial_residual

@jax.jit
def svgd_kernel(z, h):
    """
    Computes the RBF kernel matrix K and its gradient dxK.
    z shape: (num_particles, dim)
    """
    z_norm_sq = jnp.sum(z**2, axis=1, keepdims=True)
    pairwise_dists_sq = z_norm_sq + z_norm_sq.T - 2 * jnp.dot(z, z.T)
    
    K = jnp.exp(-pairwise_dists_sq / (2 * h**2))
    
    # Gradient of K w.r.t. the first argument
    dxK = -jnp.matmul(K, z)
    sum_K = jnp.sum(K, axis=1, keepdims=True)
    dxK += z * sum_K
    dxK /= (h**2)
    
    return K, dxK

def svgd_update_scan(
    z0, key, model_apply_fn, params, theta_dot_flat, t,
    spatial_residual_fn, steps, epsilon, gamma, h, corrected, config
):
    """
    Performs multiple SVGD steps using a JIT-compiled jax.lax.scan loop.
    """
    num_particles = z0.shape[0]

    def log_mu(particle):
        """Target log-probability density function."""
        # We need to reshape for the residual function
        particle = particle.reshape(1, -1)
        # Using the squared residual, as in the reference paper
        r = compute_residual_for_sampling(
            model_apply_fn, params, spatial_residual_fn, theta_dot_flat, particle, t
        )
        # Ensure r is a scalar
        r_scalar = jnp.squeeze(r)
        # Adding a small epsilon for numerical stability of log
        r_sq = jnp.square(r_scalar) + 1e-8
        log_prob = gamma * jnp.log(r_sq)
        
        # Add boundary penalty if enabled
        if config['SVGD_PARAMS'].get('boundary_penalty', False):
            domain = config['PROBLEM_DATA']['domain']
            boundary_strength = config['SVGD_PARAMS'].get('boundary_strength', 10.0)
            x_val = jnp.squeeze(particle)
            
            # Soft boundary penalty using exponential decay (JAX-compatible)
            left_penalty = jnp.where(x_val < domain[0], 
                                   jnp.exp(boundary_strength * (domain[0] - x_val)), 0.0)
            right_penalty = jnp.where(x_val > domain[1], 
                                    jnp.exp(boundary_strength * (x_val - domain[1])), 0.0)
            boundary_penalty = left_penalty + right_penalty
            
            log_prob -= boundary_penalty
        
        return log_prob

    # Pre-compute the vmapped gradient function
    grad_log_mu_vmap = jax.vmap(jax.grad(log_mu))

    def body(carry, _):
        z, key = carry
        
        # ∇_x log μ at each particle, shape (n, d)
        grad_log = grad_log_mu_vmap(z)
        
        # Gradient clipping to prevent explosion
        grad_norm = jnp.linalg.norm(grad_log, axis=1, keepdims=True)
        max_grad_norm = 10.0  # Maximum allowed gradient norm
        grad_log = jnp.where(grad_norm > max_grad_norm, 
                            grad_log * max_grad_norm / grad_norm, 
                            grad_log)
        
        # Kernel and its gradient, shapes (n, n) and (n, d)
        K, dxK = svgd_kernel(z, h)
        
        # Deterministic SVGD force (repulsive + attractive)
        phi = (jnp.matmul(K, grad_log) + dxK) / num_particles

        phi_magnitude = jnp.linalg.norm(phi, axis=1, keepdims=True)
        adaptive_epsilon = epsilon / (1.0 + phi_magnitude)  # Reduce step size when forces are large
        
        # Update step
        z_next = z + adaptive_epsilon * phi
        
        if corrected:
            key, subkey = jax.random.split(key)
            # Add Langevin-style noise, using robust SVD method
            U, S, _ = jnp.linalg.svd(K)
            noise_matrix = U @ jnp.diag(jnp.sqrt(S))
            noise = jnp.sqrt(2 * epsilon / num_particles) * noise_matrix @ jax.random.normal(subkey, z.shape)
            z_next += noise
        
        return (z_next, key), None

    # Run the scan loop
    (z_final, final_key), _ = jax.lax.scan(body, (z0, key), jnp.arange(steps))
    
    return z_final, final_key

def create_particle_stepper(model, config):
    """
    Factory function that creates a stepper for updating particle positions.
    This is the main interface for the runner.
    """
    svgd_cfg = config['SVGD_PARAMS']
    
    if not svgd_cfg['enabled']:
        # If adaptive sampling is disabled, return a dummy function that does nothing
        @jax.jit
        def dummy_stepper(particles, key, params, theta_dot_flat, t):
            return particles, key
        return dummy_stepper

    # Create a stepper function without using partial to avoid parameter conflicts
    def stepper(particles, key, params, theta_dot_flat, t):
        """
        The actual function that will be called in the main loop.
        """
        new_particles, new_key = svgd_update_scan(
            particles, key, model.apply, params, theta_dot_flat, t,
            kdv_spatial_residual, svgd_cfg['steps'],
            svgd_cfg['epsilon'], svgd_cfg['gamma'], svgd_cfg['h'], svgd_cfg['corrected'], config
        )
        
        # Constrain particles to the domain (clipping)
        domain = config['PROBLEM_DATA']['domain']
        new_particles = jnp.clip(new_particles, domain[0], domain[1])
        
        return new_particles, new_key
        
    return stepper