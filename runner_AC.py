# runner_AC.py

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from tqdm import trange
from time import time

# Import all our custom modules
from config import (
    AC_PROBLEM_DATA, AC_NETWORK_PARAMS, AC_SVGD_PARAMS, AC_EVOLUTION_PARAMS,
    AC_OUTPUT_PATHS, AC_TRAINING_DATA
)
from nn import DeepNetAC
from physics import ac_spatial_residual
from utils import get_unravel_fn, compute_residual_for_sampling
from integrator import rk4_step, estimate_theta_dot
from exact_solutions import ac_solution
from jax.flatten_util import ravel_pytree

def create_particle_stepper_AC(model, config):
    """Create particle stepper for AC equation."""
    from sampler import svgd_update_scan
    
    svgd_cfg = config['SVGD_PARAMS']
    
    if not svgd_cfg['enabled']:
        @jax.jit
        def dummy_stepper(particles, key, params, theta_dot_flat, t):
            return particles, key
        return dummy_stepper

    def stepper(particles, key, params, theta_dot_flat, t):
        """The actual function that will be called in the main loop."""
        new_particles, new_key = svgd_update_scan(
            particles, key, model.apply, params, theta_dot_flat, t,
            ac_spatial_residual, svgd_cfg['steps'],
            svgd_cfg['epsilon'], svgd_cfg['gamma'], svgd_cfg['h'], 
            svgd_cfg['corrected'], config
        )
        
        # Constrain particles to the domain (clipping)
        domain = config['PROBLEM_DATA']['domain']
        new_particles = jnp.clip(new_particles, domain[0], domain[1])
        
        return new_particles, new_key
        
    return stepper

class CoupledNGSVGDRunnerAC:
    """
    Main runner class for the Coupled Neural Galerkin-SVGD algorithm for Allen-Cahn equation.
    """
    def __init__(self, config):
        self.cfg = config
        
        # --- Setup Model and Parameters ---
        self.model = DeepNetAC(
            m=config['NETWORK_PARAMS']['m'],
            l=config['NETWORK_PARAMS']['l'],
            L=config['NETWORK_PARAMS']['L']
        )
        self.params = self._load_initial_params()
        self.unravel_fn = get_unravel_fn(self.model, jnp.ones((1, config['PROBLEM_DATA']['d'])))

        # --- Setup Particles ---
        self.particles = jnp.load('data/particle0_AC.npy')
        num_anchors = config['EVOLUTION_PARAMS'].get('num_anchor_particles', 0)
        domain = config['PROBLEM_DATA']['domain']
        
        # Create adaptive anchor particles
        self.base_anchor_particles = jnp.linspace(domain[0], domain[1], num_anchors).reshape(-1, 1)
        self.adaptive_anchors_enabled = config['EVOLUTION_PARAMS'].get('adaptive_anchors', True)
        self.extra_anchor_density = config['EVOLUTION_PARAMS'].get('extra_anchor_density', 2.0)
        
        # Initialize with base anchors
        self.anchor_particles = self.base_anchor_particles.copy()
        
        # --- Setup Sampler and Integrator ---
        self.particle_stepper = create_particle_stepper_AC(self.model, config)
        
        # --- JAX PRNG Key ---
        self.key = jax.random.PRNGKey(config['TRAINING_DATA']['seed'])

    def _load_initial_params(self):
        """Loads and unravels the initial network parameters."""
        theta0_flat = jnp.load(self.cfg['OUTPUT_PATHS']['initial_theta'])
        dummy_input = jnp.ones((1, self.cfg['PROBLEM_DATA']['d']))
        _, unravel_fn = ravel_pytree(self.model.init(jax.random.PRNGKey(0), dummy_input)['params'])
        return unravel_fn(theta0_flat)

    def _update_adaptive_anchors(self, current_t):
        """Update anchor particles based on dynamic particle distribution."""
        if not self.adaptive_anchors_enabled:
            return
            
        domain = self.cfg['PROBLEM_DATA']['domain']
        domain_length = domain[1] - domain[0]
        
        dynamic_particles_1d = self.particles.squeeze()
        
        analysis_grid = jnp.linspace(domain[0], domain[1], 200)
        grid_spacing = analysis_grid[1] - analysis_grid[0]
        
        window_size = domain_length / 20
        densities = []
        
        for x in analysis_grid:
            distances = jnp.abs(dynamic_particles_1d - x)
            particles_in_window = jnp.sum(distances <= window_size)
            density = particles_in_window / (2 * window_size + 1e-8)
            densities.append(density)
        
        densities = jnp.array(densities)
        
        mean_density = jnp.mean(densities)
        sparse_threshold = mean_density * 0.3
        sparse_mask = densities < sparse_threshold
        
        sparse_regions = analysis_grid[sparse_mask]
        
        if len(sparse_regions) > 0:
            extra_anchors_per_region = max(1, int(self.extra_anchor_density))
            extra_anchors = []
            
            sparse_groups = []
            current_group = [sparse_regions[0]]
            
            for i in range(1, len(sparse_regions)):
                if sparse_regions[i] - sparse_regions[i-1] <= 2 * grid_spacing:
                    current_group.append(sparse_regions[i])
                else:
                    sparse_groups.append(current_group)
                    current_group = [sparse_regions[i]]
            sparse_groups.append(current_group)
            
            for group in sparse_groups:
                if len(group) >= 3:
                    group_start, group_end = min(group), max(group)
                    group_anchors = jnp.linspace(group_start, group_end, extra_anchors_per_region)
                    extra_anchors.extend(group_anchors)
            
            if extra_anchors:
                extra_anchors = jnp.array(extra_anchors).reshape(-1, 1)
                self.anchor_particles = jnp.concatenate([self.base_anchor_particles, extra_anchors])
            else:
                self.anchor_particles = self.base_anchor_particles.copy()
        else:
            self.anchor_particles = self.base_anchor_particles.copy()

    def run(self):
        """Executes the main time-stepping loop."""
        evo_cfg = self.cfg['EVOLUTION_PARAMS']
        dt = evo_cfg['dt']
        n_steps = int(evo_cfg['t_final'] / dt)
        
        history = {'t': [], 'theta': [], 'particles': []}
        
        current_t = 0.0
        
        print("🚀 Starting time evolution for Allen-Cahn equation...")
        start_time = time()
        
        for k in trange(n_steps, desc="Neural Galerkin Evolution (AC)"):
            # Store current state
            history['t'].append(current_t)
            history['theta'].append(self.params)
            history['particles'].append(self.particles)
            
            # Update adaptive anchor particles every 10 steps
            if k % 10 == 0:
                self._update_adaptive_anchors(current_t)
            
            # 1. Combine dynamic and anchor particles for Galerkin method
            particles_for_galerkin = jnp.concatenate([self.particles, self.anchor_particles])

            # 2. Estimate current theta_dot for SVGD
            theta_dot_flat = estimate_theta_dot(
                self.model.apply, self.params, particles_for_galerkin,
                evo_cfg['ridge_lambda'], ac_spatial_residual, current_t
            )
            
            # Check for NaN/Inf in theta_dot
            if jnp.any(jnp.isnan(theta_dot_flat)) or jnp.any(jnp.isinf(theta_dot_flat)):
                print(f"Warning: NaN/Inf in theta_dot at step {k}, t={current_t:.4f}")
                theta_dot_flat = jnp.where(jnp.isnan(theta_dot_flat) | jnp.isinf(theta_dot_flat), 
                                          0.0, theta_dot_flat)

            # 3. Update dynamic particle positions using SVGD
            self.particles, self.key = self.particle_stepper(
                self.particles, self.key, self.params, theta_dot_flat, current_t
            )
            
            # Check particles for NaN/Inf
            if jnp.any(jnp.isnan(self.particles)) or jnp.any(jnp.isinf(self.particles)):
                print(f"Warning: NaN/Inf in particles at step {k}, t={current_t:.4f}")
                self.particles = jnp.where(jnp.isnan(self.particles) | jnp.isinf(self.particles),
                                         0.0, self.particles)
            
            # 4. Perform one RK4 time step for theta
            new_particles_for_galerkin = jnp.concatenate([self.particles, self.anchor_particles])
            self.params = rk4_step(
                self.model.apply, self.params, new_particles_for_galerkin, dt,
                evo_cfg['ridge_lambda'], ac_spatial_residual, current_t
            )
            
            # Check parameters for NaN/Inf
            params_flat_check, _ = ravel_pytree(self.params)
            if jnp.any(jnp.isnan(params_flat_check)) or jnp.any(jnp.isinf(params_flat_check)):
                print(f"Error: NaN/Inf in parameters at step {k}, t={current_t:.4f}. Stopping.")
                break
            
            current_t += dt
            
        print(f"✅ Evolution complete. Total time: {time() - start_time:.2f}s")
        self._save_and_report(history)

    def _save_and_report(self, history):
        """Saves the simulation results and generates plots."""
        print("\n💾 Saving results...")
        
        out_dir = "run_output_AC"
        os.makedirs(out_dir, exist_ok=True)
        
        # Save final state
        final_theta_flat, _ = ravel_pytree(history['theta'][-1])
        np.save(os.path.join(out_dir, "theta_final_AC.npy"), final_theta_flat)
        np.save(os.path.join(out_dir, "particles_final_AC.npy"), history['particles'][-1])
        
        print("📊 Generating final plots...")
        # Plotting at specified time points
        plot_times = [0.0, self.cfg['EVOLUTION_PARAMS']['t_final'] / 2, self.cfg['EVOLUTION_PARAMS']['t_final']]
        
        fig, axes = plt.subplots(1, len(plot_times), figsize=(18, 5))
        
        for ax, t_plot in zip(axes, plot_times):
            # Find the closest time step in history
            idx = np.abs(np.array(history['t']) - t_plot).argmin()
            
            params_at_t = history['theta'][idx]
            particles_at_t = history['particles'][idx]
            
            x_plot = jnp.linspace(
                self.cfg['PROBLEM_DATA']['domain'][0],
                self.cfg['PROBLEM_DATA']['domain'][1],
                512
            )
            current_t = history['t'][idx]
            u_true = ac_solution(x_plot, current_t)
            u_pred = self.model.apply({'params': params_at_t}, x_plot.reshape(-1, 1))
            
            # Debug: print solution statistics
            print(f"  t={current_t:.2f}: Truth range=[{jnp.min(u_true):.4f}, {jnp.max(u_true):.4f}], "
                  f"Pred range=[{jnp.min(u_pred):.4f}, {jnp.max(u_pred):.4f}]")
            
            ax.plot(x_plot, u_true, 'k--', linewidth=2, label='Truth')
            ax.plot(x_plot, u_pred, 'darkviolet', linewidth=2, label='Neural Galerkin')
            
            # Plot particles at a position relative to the data range
            particles_1d = particles_at_t.squeeze()
            y_range = max(np.max(u_true), np.max(u_pred)) - min(np.min(u_true), np.min(u_pred))
            particle_y = min(np.min(u_true), np.min(u_pred)) - 0.1 * y_range
            ax.scatter(particles_1d, particle_y * np.ones_like(particles_1d),
                       c='mediumseagreen', marker='x', s=30, label='Particles', zorder=5)
            
            ax.set_title(f't = {history["t"][idx]:.2f}')
            ax.set_xlabel('x')
            ax.set_ylabel('u(x, t)')
            ax.legend()
            ax.grid(True)
            
            # Set appropriate y limits for AC equation (include particles)
            y_max = max(np.max(u_true), np.max(u_pred))
            y_min = min(np.min(u_true), np.min(u_pred))
            ax.set_ylim(particle_y - 0.05 * y_range, y_max * 1.1)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "evolution_plot_AC.png"), dpi=300)
        print(f"💾 Evolution plot saved to '{out_dir}/evolution_plot_AC.png'")
        plt.show()

if __name__ == "__main__":
    # Combine all configs into one dictionary for the runner
    config = {
        'PROBLEM_DATA': AC_PROBLEM_DATA,
        'NETWORK_PARAMS': AC_NETWORK_PARAMS,
        'SVGD_PARAMS': AC_SVGD_PARAMS,
        'EVOLUTION_PARAMS': AC_EVOLUTION_PARAMS,
        'OUTPUT_PATHS': AC_OUTPUT_PATHS,
        'TRAINING_DATA': AC_TRAINING_DATA
    }
    
    runner = CoupledNGSVGDRunnerAC(config)
    runner.run()

