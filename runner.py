# runner.py

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from tqdm import trange
from time import time

# Import all our custom modules
from config import PROBLEM_DATA, NETWORK_PARAMS, SVGD_PARAMS, EVOLUTION_PARAMS, OUTPUT_PATHS, TRAINING_DATA
from nn import ShallowNetKdV
from physics import kdv_spatial_residual
from utils import get_unravel_fn, compute_residual_for_sampling
from sampler import create_particle_stepper
from integrator import rk4_step, estimate_theta_dot
from jax.flatten_util import ravel_pytree

class CoupledNGSVGDRunner:
    """
    Main runner class for the Coupled Neural Galerkin-SVGD algorithm.
    This class orchestrates the entire simulation flow.
    """
    def __init__(self, config):
        self.cfg = config
        
        # --- Setup Model and Parameters ---
        self.model = ShallowNetKdV(m=config['NETWORK_PARAMS']['m'], L=config['NETWORK_PARAMS']['L'])
        self.params = self._load_initial_params()
        self.unravel_fn = get_unravel_fn(self.model, jnp.ones((1, config['PROBLEM_DATA']['d'])))

        # --- Setup Particles ---
        self.particles = jnp.load(config['OUTPUT_PATHS']['initial_particles'])
        num_anchors = config['EVOLUTION_PARAMS'].get('num_anchor_particles', 0)
        domain = config['PROBLEM_DATA']['domain']
        
        # Create adaptive anchor particles
        self.base_anchor_particles = jnp.linspace(domain[0], domain[1], num_anchors).reshape(-1, 1)
        self.adaptive_anchors_enabled = config['EVOLUTION_PARAMS'].get('adaptive_anchors', True)
        self.extra_anchor_density = config['EVOLUTION_PARAMS'].get('extra_anchor_density', 2.0)
        
        # Initialize with base anchors
        self.anchor_particles = self.base_anchor_particles.copy()
        
        # --- Setup Sampler and Integrator ---
        self.particle_stepper = create_particle_stepper(self.model, config)
        
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
        
        # Find sparse regions (areas with few dynamic particles)
        dynamic_particles_1d = self.particles.squeeze()
        
        # Create a fine grid to analyze particle density
        analysis_grid = jnp.linspace(domain[0], domain[1], 200)
        grid_spacing = analysis_grid[1] - analysis_grid[0]
        
        # Calculate local particle density using a sliding window
        window_size = domain_length / 20  # Adaptive window size
        densities = []
        
        for x in analysis_grid:
            # Count particles within window around x
            distances = jnp.abs(dynamic_particles_1d - x)
            particles_in_window = jnp.sum(distances <= window_size)
            density = particles_in_window / (2 * window_size + 1e-8)  # Normalize by window size
            densities.append(density)
        
        densities = jnp.array(densities)
        
        # Identify sparse regions (density below threshold)
        mean_density = jnp.mean(densities)
        sparse_threshold = mean_density * 0.3  # Regions with < 30% of mean density
        sparse_mask = densities < sparse_threshold
        
        # Find sparse regions that need extra anchors
        sparse_regions = analysis_grid[sparse_mask]
        
        if len(sparse_regions) > 0:
            # Add extra anchors in sparse regions with higher density
            extra_anchors_per_region = max(1, int(self.extra_anchor_density))
            extra_anchors = []
            
            # Group consecutive sparse points into regions
            sparse_groups = []
            current_group = [sparse_regions[0]]
            
            for i in range(1, len(sparse_regions)):
                if sparse_regions[i] - sparse_regions[i-1] <= 2 * grid_spacing:
                    current_group.append(sparse_regions[i])
                else:
                    sparse_groups.append(current_group)
                    current_group = [sparse_regions[i]]
            sparse_groups.append(current_group)
            
            # Add extra anchors in each sparse group
            for group in sparse_groups:
                if len(group) >= 3:  # Only for significant sparse regions
                    group_start, group_end = min(group), max(group)
                    group_anchors = jnp.linspace(group_start, group_end, extra_anchors_per_region)
                    extra_anchors.extend(group_anchors)
            
            if extra_anchors:
                extra_anchors = jnp.array(extra_anchors).reshape(-1, 1)
                # Combine base anchors with extra anchors
                self.anchor_particles = jnp.concatenate([self.base_anchor_particles, extra_anchors])
            else:
                self.anchor_particles = self.base_anchor_particles.copy()
        else:
            self.anchor_particles = self.base_anchor_particles.copy()

    def run(self):
        """Executes the main time-stepping loop."""
        # --- Initialization ---
        evo_cfg = self.cfg['EVOLUTION_PARAMS']
        dt = evo_cfg['dt']
        n_steps = int(evo_cfg['t_final'] / dt)
        
        history = {'t': [], 'theta': [], 'particles': []}
        
        current_t = 0.0
        
        print("🚀 Starting time evolution...")
        start_time = time()
        
        for k in trange(n_steps, desc="Neural Galerkin Evolution"):
            # Store current state
            history['t'].append(current_t)
            history['theta'].append(self.params)
            history['particles'].append(self.particles)
            
            # --- Debug Information (every 50 steps) ---
            if k % 50 == 0:
                print(f"\n=== DEBUG INFO at step {k}, t={current_t:.4f} ===")
                
                # Check particles for NaN/Inf
                particles_stats = {
                    'min': jnp.min(self.particles),
                    'max': jnp.max(self.particles),
                    'mean': jnp.mean(self.particles),
                    'std': jnp.std(self.particles),
                    'has_nan': jnp.any(jnp.isnan(self.particles)),
                    'has_inf': jnp.any(jnp.isinf(self.particles)),
                    'count': len(self.particles)
                }
                print(f"Particles - Min: {particles_stats['min']:.4f}, Max: {particles_stats['max']:.4f}")
                print(f"Particles - Mean: {particles_stats['mean']:.4f}, Std: {particles_stats['std']:.4f}")
                print(f"Particles - Has NaN: {particles_stats['has_nan']}, Has Inf: {particles_stats['has_inf']}")
                print(f"Particles - Count: {particles_stats['count']}")
                
                # Check boundary violations
                domain = self.cfg['PROBLEM_DATA']['domain']
                particles_1d = self.particles.squeeze()
                out_of_bounds = jnp.sum((particles_1d < domain[0]) | (particles_1d > domain[1]))
                print(f"Particles - Out of bounds [{domain[0]}, {domain[1]}]: {out_of_bounds}")
                
                # Check anchor particle coverage
                print(f"Anchor particles - Count: {len(self.anchor_particles)}")
                base_count = len(self.base_anchor_particles)
                extra_count = len(self.anchor_particles) - base_count
                print(f"Anchor particles - Base: {base_count}, Extra: {extra_count}")
                anchor_min, anchor_max = jnp.min(self.anchor_particles), jnp.max(self.anchor_particles)
                print(f"Anchor particles - Range: [{anchor_min:.1f}, {anchor_max:.1f}]")
                
                # Check coverage gaps (areas with sparse particle coverage)
                all_particles = jnp.concatenate([self.particles.squeeze(), self.anchor_particles.squeeze()])
                all_particles_sorted = jnp.sort(all_particles)
                max_gap = jnp.max(jnp.diff(all_particles_sorted))
                print(f"Max particle gap: {max_gap:.2f}")
                
                # Check network parameters
                params_flat, _ = ravel_pytree(self.params)
                params_stats = {
                    'min': jnp.min(params_flat),
                    'max': jnp.max(params_flat),
                    'mean': jnp.mean(params_flat),
                    'has_nan': jnp.any(jnp.isnan(params_flat)),
                    'has_inf': jnp.any(jnp.isinf(params_flat))
                }
                print(f"Network Params - Min: {params_stats['min']:.4f}, Max: {params_stats['max']:.4f}")
                print(f"Network Params - Mean: {params_stats['mean']:.4f}")
                print(f"Network Params - Has NaN: {params_stats['has_nan']}, Has Inf: {params_stats['has_inf']}")
                
                # Test network output at particles
                try:
                    u_vals = self.model.apply({'params': self.params}, self.particles)
                    u_stats = {
                        'min': jnp.min(u_vals),
                        'max': jnp.max(u_vals),
                        'mean': jnp.mean(u_vals),
                        'has_nan': jnp.any(jnp.isnan(u_vals)),
                        'has_inf': jnp.any(jnp.isinf(u_vals))
                    }
                    print(f"Network Output - Min: {u_stats['min']:.4f}, Max: {u_stats['max']:.4f}")
                    print(f"Network Output - Mean: {u_stats['mean']:.4f}")
                    print(f"Network Output - Has NaN: {u_stats['has_nan']}, Has Inf: {u_stats['has_inf']}")
                except Exception as e:
                    print(f"Error evaluating network: {e}")
                
                # Check network behavior in boundary regions
                try:
                    # Test network output in left and right "vacuum" regions
                    left_boundary = jnp.linspace(domain[0], domain[0] + 5, 10).reshape(-1, 1)
                    right_boundary = jnp.linspace(domain[1] - 5, domain[1], 10).reshape(-1, 1)
                    
                    u_left = self.model.apply({'params': self.params}, left_boundary)
                    u_right = self.model.apply({'params': self.params}, right_boundary)
                    
                    print(f"Left boundary region - Max: {jnp.max(jnp.abs(u_left)):.4f}")
                    print(f"Right boundary region - Max: {jnp.max(jnp.abs(u_right)):.4f}")
                except Exception as e:
                    print(f"Error evaluating boundary regions: {e}")
                
                # Test SVGD-related quantities if we have theta_dot
                try:
                    # Estimate theta_dot for debugging
                    particles_for_galerkin = jnp.concatenate([self.particles, self.anchor_particles])
                    debug_theta_dot = estimate_theta_dot(
                        self.model.apply, self.params, particles_for_galerkin,
                        evo_cfg['ridge_lambda'], kdv_spatial_residual, current_t
                    )
                    
                    theta_dot_stats = {
                        'min': jnp.min(debug_theta_dot),
                        'max': jnp.max(debug_theta_dot),
                        'mean': jnp.mean(debug_theta_dot),
                        'has_nan': jnp.any(jnp.isnan(debug_theta_dot)),
                        'has_inf': jnp.any(jnp.isinf(debug_theta_dot))
                    }
                    print(f"Theta_dot - Min: {theta_dot_stats['min']:.4f}, Max: {theta_dot_stats['max']:.4f}")
                    print(f"Theta_dot - Mean: {theta_dot_stats['mean']:.4f}")
                    print(f"Theta_dot - Has NaN: {theta_dot_stats['has_nan']}, Has Inf: {theta_dot_stats['has_inf']}")
                    
                    # Test residual computation for a few particles
                    from utils import compute_residual_for_sampling
                    test_particles = self.particles[:3] if len(self.particles) >= 3 else self.particles
                    for i, particle in enumerate(test_particles):
                        try:
                            residual = compute_residual_for_sampling(
                                self.model.apply, self.params, kdv_spatial_residual, 
                                debug_theta_dot, particle.reshape(1, -1), current_t
                            )
                            log_val = 0.25 * jnp.log(jnp.square(jnp.squeeze(residual)) + 1e-8)  # gamma=0.25 from config
                            print(f"Particle {i} - x: {particle[0]:.4f}, residual: {jnp.squeeze(residual):.4f}, log_val: {log_val:.4f}")
                        except Exception as e:
                            print(f"Error computing residual for particle {i}: {e}")
                            
                except Exception as e:
                    print(f"Error in SVGD debugging: {e}")
                
                print("=" * 50)
            
            # --- Main Algorithm Loop (Decoupled) ---
            
            # Update adaptive anchor particles every 10 steps
            if k % 10 == 0:
                self._update_adaptive_anchors(current_t)
            
            # 1. Combine dynamic and anchor particles for Galerkin method
            particles_for_galerkin = jnp.concatenate([self.particles, self.anchor_particles])

            # 2. Estimate current theta_dot for SVGD
            theta_dot_flat = estimate_theta_dot(
                self.model.apply, self.params, particles_for_galerkin,
                evo_cfg['ridge_lambda'], kdv_spatial_residual, current_t
            )

            # 3. Update dynamic particle positions using SVGD
            self.particles, self.key = self.particle_stepper(
                self.particles, self.key, self.params, theta_dot_flat, current_t
            )
            
            # 4. Perform one RK4 time step for theta
            # Re-combine with the *new* dynamic particles
            new_particles_for_galerkin = jnp.concatenate([self.particles, self.anchor_particles])
            self.params = rk4_step(
                self.model.apply, self.params, new_particles_for_galerkin, dt,
                evo_cfg['ridge_lambda'], kdv_spatial_residual, current_t
            )
            
            current_t += dt
            
        print(f"✅ Evolution complete. Total time: {time() - start_time:.2f}s")
        self._save_and_report(history)

    def _generate_l2_error_plots(self, history, out_dir):
        """Generate L2 error plots for quantitative analysis."""
        print("📈 Generating L2 error analysis...")
        
        # Calculate L2 errors over time
        l2_errors = []
        times = []
        spatial_errors_snapshots = {}  # Store spatial error for selected time points
        
        x_eval = jnp.linspace(self.cfg['PROBLEM_DATA']['domain'][0], 
                             self.cfg['PROBLEM_DATA']['domain'][1], 512)
        
        for i, t in enumerate(history['t']):
            # Get true solution
            u_true = self.cfg['PROBLEM_DATA']['exact_sol'](x_eval, t)
            
            # Get neural network prediction
            params_at_t = history['theta'][i]
            u_pred = self.model.apply({'params': params_at_t}, x_eval.reshape(-1, 1))
            
            # Calculate L2 error
            error = u_pred - u_true
            l2_error = jnp.sqrt(jnp.mean(error**2))
            
            l2_errors.append(l2_error)
            times.append(t)
            
            # Store spatial error for specific time points
            if i % (len(history['t']) // 4) == 0 or i == len(history['t']) - 1:
                spatial_errors_snapshots[t] = {
                    'x': x_eval,
                    'error': error,
                    'u_true': u_true,
                    'u_pred': u_pred
                }
        
        # Create L2 error plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: L2 error over time
        axes[0, 0].semilogy(times, l2_errors, 'b-', linewidth=2, marker='o', markersize=3)
        axes[0, 0].set_xlabel('Time t')
        axes[0, 0].set_ylabel('L2 Error (log scale)')
        axes[0, 0].set_title('L2 Error Evolution Over Time')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add statistics
        min_error = jnp.min(jnp.array(l2_errors))
        max_error = jnp.max(jnp.array(l2_errors))
        mean_error = jnp.mean(jnp.array(l2_errors))
        final_error = l2_errors[-1]
        
        stats_text = f'Min: {min_error:.2e}\nMax: {max_error:.2e}\nMean: {mean_error:.2e}\nFinal: {final_error:.2e}'
        axes[0, 0].text(0.02, 0.98, stats_text, transform=axes[0, 0].transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Plot 2: Linear scale L2 error
        axes[0, 1].plot(times, l2_errors, 'r-', linewidth=2, marker='s', markersize=3)
        axes[0, 1].set_xlabel('Time t')
        axes[0, 1].set_ylabel('L2 Error (linear scale)')
        axes[0, 1].set_title('L2 Error Evolution (Linear Scale)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Spatial error distribution at selected times
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        for i, (t, data) in enumerate(spatial_errors_snapshots.items()):
            if i < len(colors):
                axes[1, 0].plot(data['x'], jnp.abs(data['error']), 
                               color=colors[i], linewidth=2, label=f't = {t:.2f}')
        
        axes[1, 0].set_xlabel('Spatial coordinate x')
        axes[1, 0].set_ylabel('|Error| = |u_pred - u_true|')
        axes[1, 0].set_title('Spatial Error Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
        
        # Plot 4: Error statistics vs particle distribution
        # Show how error correlates with particle density
        final_time_data = list(spatial_errors_snapshots.values())[-1]
        final_particles = history['particles'][-1].squeeze()
        
        # Create histogram of particle density
        hist, bin_edges = jnp.histogram(final_particles, bins=50, 
                                      range=(self.cfg['PROBLEM_DATA']['domain'][0], 
                                            self.cfg['PROBLEM_DATA']['domain'][1]))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Interpolate error to bin centers
        error_at_bins = jnp.interp(bin_centers, final_time_data['x'], jnp.abs(final_time_data['error']))
        
        ax4_twin = axes[1, 1].twinx()
        
        # Plot particle density (histogram)
        axes[1, 1].bar(bin_centers, hist, width=(bin_edges[1]-bin_edges[0])*0.8, 
                      alpha=0.6, color='lightblue', label='Particle density')
        axes[1, 1].set_ylabel('Particle count', color='blue')
        axes[1, 1].tick_params(axis='y', labelcolor='blue')
        
        # Plot error
        ax4_twin.semilogy(bin_centers, error_at_bins, 'r-', linewidth=2, 
                         marker='o', markersize=4, label='|Error|')
        ax4_twin.set_ylabel('|Error| (log scale)', color='red')
        ax4_twin.tick_params(axis='y', labelcolor='red')
        
        axes[1, 1].set_xlabel('Spatial coordinate x')
        axes[1, 1].set_title('Error vs Particle Density (Final Time)')
        
        # Add legends
        lines1, labels1 = axes[1, 1].get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        axes[1, 1].legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "l2_error_analysis.png"), dpi=300)
        plt.show()
        
        # Save L2 error data
        error_data = {
            'times': [float(t) for t in times],
            'l2_errors': [float(e) for e in l2_errors],
            'statistics': {
                'min_error': float(min_error),
                'max_error': float(max_error),
                'mean_error': float(mean_error),
                'final_error': float(final_error)
            }
        }
        
        import json
        with open(os.path.join(out_dir, "l2_error_data.json"), 'w') as f:
            json.dump(error_data, f, indent=2)
        
        print(f"📊 L2 Error Statistics:")
        print(f"   Min Error: {min_error:.2e}")
        print(f"   Max Error: {max_error:.2e}")
        print(f"   Mean Error: {mean_error:.2e}")
        print(f"   Final Error: {final_error:.2e}")
        print(f"💾 L2 error analysis saved to '{out_dir}/l2_error_analysis.png'")

    def _generate_spacetime_plot(self, out_dir):
        """Generate spacetime plot showing the complete solution evolution."""
        print("🌌 Generating spacetime plot...")
        
        # Create spacetime grid
        domain = self.cfg['PROBLEM_DATA']['domain']
        t_final = self.cfg['EVOLUTION_PARAMS']['t_final']
        
        # High resolution grid for smooth visualization
        x_grid = jnp.linspace(domain[0], domain[1], 300)
        t_grid = jnp.linspace(0, t_final, 200)
        X, T = jnp.meshgrid(x_grid, t_grid)
        
        # Compute true solution over the entire spacetime grid
        print("Computing spacetime solution...")
        U_spacetime = jnp.zeros_like(X)
        
        for i, t in enumerate(t_grid):
            if i % 20 == 0:
                print(f"  Progress: {i/len(t_grid)*100:.1f}%")
            
            # Get true solution at this time
            u_true_t = self.cfg['PROBLEM_DATA']['exact_sol'](x_grid, t)
            U_spacetime = U_spacetime.at[i, :].set(u_true_t)
        
        # Create the spacetime plot
        plt.figure(figsize=(12, 8))
        
        # Plot with proper orientation (time on x-axis, space on y-axis to match your reference)
        im = plt.imshow(U_spacetime.T, extent=[0, t_final, domain[0], domain[1]], 
                       aspect='auto', origin='lower', cmap='plasma', 
                       vmin=0, vmax=jnp.max(U_spacetime))
        
        # Add colorbar
        cbar = plt.colorbar(im, shrink=0.8)
        cbar.set_label('u(x, t)', fontsize=14)
        
        # Labels and title
        plt.xlabel('Time (t)', fontsize=14)
        plt.ylabel('Space (x)', fontsize=14)
        plt.title('Spacetime Solution of the KdV Equation (Two-Soliton)\nNeural Galerkin Method', fontsize=16)
        
        # Add grid for better readability
        plt.grid(True, alpha=0.3, color='white', linewidth=0.5)
        
        # Tight layout and save
        plt.tight_layout()
        spacetime_path = os.path.join(out_dir, "spacetime_solution.png")
        plt.savefig(spacetime_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"🌌 Spacetime plot saved to '{spacetime_path}'")

    def _save_and_report(self, history):
        """Saves the simulation results and generates plots."""
        print("\n💾 Saving results...")
        
        # Create output directory
        out_dir = "run_output"
        os.makedirs(out_dir, exist_ok=True)
        
        # Save history
        # (Converting PyTrees to arrays for saving can be complex, for now we save the final state)
        final_theta_flat, _ = ravel_pytree(history['theta'][-1])
        np.save(os.path.join(out_dir, "theta_final.npy"), final_theta_flat)
        np.save(os.path.join(out_dir, "particles_final.npy"), history['particles'][-1])
        
        # Save config for reproducibility
        # (Need to handle non-serializable items like functions)
        
        print("📊 Generating final plots...")
        # Plotting at specified time points
        plot_times = [0.0, self.cfg['EVOLUTION_PARAMS']['t_final'] / 2, self.cfg['EVOLUTION_PARAMS']['t_final']]
        
        fig, axes = plt.subplots(1, len(plot_times), figsize=(18, 5))
        
        for ax, t_plot in zip(axes, plot_times):
            # Find the closest time step in history
            idx = np.abs(np.array(history['t']) - t_plot).argmin()
            
            params_at_t = history['theta'][idx]
            particles_at_t = history['particles'][idx]
            
            x_plot = jnp.linspace(self.cfg['PROBLEM_DATA']['domain'][0], self.cfg['PROBLEM_DATA']['domain'][1], 512)
            u_true = self.cfg['PROBLEM_DATA']['exact_sol'](x_plot, history['t'][idx])
            u_pred = self.model.apply({'params': params_at_t}, x_plot.reshape(-1, 1))
            
            ax.plot(x_plot, u_true, 'k--', linewidth=2, label='Truth')
            ax.plot(x_plot, u_pred, 'darkviolet', linewidth=2, label='Neural Galerkin')
            ax.scatter(particles_at_t, -0.2 * np.ones_like(particles_at_t),
                       c='mediumseagreen', marker='x', s=30, label='Particles')
            ax.set_title(f't = {history["t"][idx]:.2f}')
            ax.set_xlabel('x')
            ax.set_ylabel('u(x, t)')
            ax.legend()
            ax.grid(True)
            ax.set_ylim(-0.5, np.max(u_true) * 1.1)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "evolution_plot.png"), dpi=300)
        plt.show()
        
        # Generate L2 error analysis
        self._generate_l2_error_plots(history, out_dir)
        
        # Generate spacetime plot
        self._generate_spacetime_plot(out_dir)

if __name__ == "__main__":
    # Combine all configs into one dictionary for the runner
    config = {
        'PROBLEM_DATA': PROBLEM_DATA,
        'NETWORK_PARAMS': NETWORK_PARAMS,
        'SVGD_PARAMS': SVGD_PARAMS,
        'EVOLUTION_PARAMS': EVOLUTION_PARAMS,
        'OUTPUT_PATHS': OUTPUT_PATHS,
        'TRAINING_DATA': TRAINING_DATA # For seed
    }
    
    runner = CoupledNGSVGDRunner(config)
    runner.run()