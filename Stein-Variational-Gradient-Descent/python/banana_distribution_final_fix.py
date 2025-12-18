"""
Final Fixed Banana Distribution Experiment
Key fixes:
1. Adaptive bandwidth with upper bound to prevent explosion
2. Smaller step size with adaptive scheduling
3. Better initialization
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import os

class BananaDistribution:
    def __init__(self, b=0.03, sigma1=10.0, sigma2=2.0):
        # sigma1: x1 方向基底高斯的 std
        # sigma2: x2 方向基底高斯的 std
        self.b = b
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def log_prob(self, theta):
        x1, x2 = theta[:, 0], theta[:, 1]
        g = x2 - self.b * (x1**2 - self.sigma1**2)
        U = 0.5 * (x1**2 / self.sigma1**2 + g**2 / self.sigma2**2)
        return -U

    def dlnprob(self, theta):
        """
        Gradient of log probability (self-consistent with log_prob).
        
        For U = 0.5 * (x1^2/sigma1^2 + g^2/sigma2^2) where g = x2 - b*(x1^2 - sigma1^2):
        dU/dx1 = x1/sigma1^2 - (2*b*x1*g)/sigma2^2
        dU/dx2 = g/sigma2^2
        
        Since log_prob = -U, we return -dU/dtheta = dlnprob/dtheta
        """
        x1, x2 = theta[:, 0], theta[:, 1]
        g = x2 - self.b * (x1**2 - self.sigma1**2)

        # dU/dx1: derivative of U w.r.t. x1
        dU_dx1 = (x1 / self.sigma1**2) - (2 * self.b * x1 * g) / (self.sigma2**2)
        # dU/dx2: derivative of U w.r.t. x2
        dU_dx2 = g / (self.sigma2**2)

        # Return gradient of log_prob = -U, so dlnprob/dtheta = -dU/dtheta
        return -np.column_stack([dU_dx1, dU_dx2])



class SVGD_Adaptive:
    """SVGD with adaptive bandwidth that doesn't explode"""
    
    def __init__(self, initial_h=None, max_h=None):
        self.initial_h = initial_h
        self.max_h = max_h or 2.0  # Cap bandwidth
    
    def svgd_kernel(self, theta, h=-1):
        sq_dist = pdist(theta)
        pairwise_dists = squareform(sq_dist)**2
        
        if h < 0:  # Adaptive bandwidth
            median_dist = np.median(pairwise_dists[pairwise_dists > 0])
            
            # Calculate bandwidth
            if self.initial_h is None:
                h_calc = np.sqrt(0.5 * median_dist / np.log(theta.shape[0] + 1))
            else:
                # Use initial h as reference, but adapt slightly
                current_scale = np.sqrt(median_dist)
                initial_scale = self.initial_h * np.sqrt(2 * np.log(theta.shape[0] + 1) / 0.5)
                h_calc = self.initial_h * np.sqrt(current_scale / initial_scale)
                h_calc = min(h_calc, self.max_h)  # Cap it
            
            h = min(h_calc, self.max_h)
        
        Kxy = np.exp(-pairwise_dists / (h**2 * 2))
        
        dxkxy = -np.matmul(Kxy, theta)
        sumkxy = np.sum(Kxy, axis=1)
        for i in range(theta.shape[1]):
            dxkxy[:, i] = dxkxy[:, i] + np.multiply(theta[:, i], sumkxy)
        dxkxy = dxkxy / (h**2)
        
        return (Kxy, dxkxy), h


def plot_comparison(banana_model, particles_history, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    
    # Expand plotting range to show full banana shape
    x1_range = np.linspace(-20, 20, 300)
    x2_range = np.linspace(-10, 10, 300)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    grid_points = np.column_stack([X1.ravel(), X2.ravel()])
    
    # Use the SAME log_prob function as SVGD uses
    log_probs = banana_model.log_prob(grid_points)
    # Normalize for visualization (subtract max to avoid overflow)
    log_probs_max = np.max(log_probs)
    probs = np.exp(log_probs - log_probs_max)
    probs = probs.reshape(X1.shape)
    
    # Verify: check banana curve has high probability
    x1_test = np.array([0, 5, -5, 10, -10])
    x2_test = banana_model.b * (x1_test**2 - banana_model.sigma1**2)  # Theoretical banana curve
    test_points = np.column_stack([x1_test, x2_test])
    test_log_probs = banana_model.log_prob(test_points)
    test_probs = np.exp(test_log_probs - log_probs_max)
    print("  Verification: Banana curve points probability: {}".format(test_probs))
    
    # Ensure we're using the correct probability values
    # The contour should follow the banana shape, not be elliptical
    
    # Adjust layout based on number of checkpoints
    n_plots = len(particles_history)
    if n_plots <= 6:
        n_rows, n_cols = 2, 3
    elif n_plots <= 8:
        n_rows, n_cols = 2, 4
    else:
        n_rows, n_cols = 3, 3
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Hide extra subplots if any
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    for idx, (iter_num, particles) in enumerate(particles_history):
        ax = axes[idx]
        
        # Use more levels starting from very small values to show full banana shape
        levels = np.linspace(1e-5, 1.0, 40)  # Start from very small to show full banana
        ax.contour(X1, X2, probs, levels=levels, colors='gray', alpha=0.4, linewidths=0.8)
        ax.contourf(X1, X2, probs, levels=levels, cmap='Reds', alpha=0.25)
        
        # Banana curve: x2 = b * (x1^2 - sigma1^2)
        x1_curve = np.linspace(-20, 20, 200)
        x2_curve = banana_model.b * (x1_curve**2 - banana_model.sigma1**2)
        ax.plot(x1_curve, x2_curve, 'b--', linewidth=1.5, alpha=0.6, label='Banana Curve')
        
        ax.scatter(particles[:, 0], particles[:, 1], 
                  c='green', s=40, alpha=0.7, edgecolors='darkgreen', linewidths=0.5,
                  label='Particles')
        
        log_probs_particles = banana_model.log_prob(particles)
        probs_particles = np.exp(log_probs_particles - np.max(log_probs))
        high_prob_ratio = np.mean(probs_particles > 0.1)
        
        mean = np.mean(particles, axis=0)
        std = np.std(particles, axis=0)
        
        ax.set_xlim(-20, 20)
        ax.set_ylim(-10, 10)
        ax.set_xlabel('$x_1$', fontsize=11)
        ax.set_ylabel('$x_2$', fontsize=11)
        ax.set_title('Iter {}: High prob={:.0%}\nMean=({:.2f},{:.2f}), Std=({:.2f},{:.2f})'.format(
            iter_num, high_prob_ratio, mean[0], mean[1], std[0], std[1]), 
            fontsize=11, fontweight='bold')
        if idx == 0:
            ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.suptitle('SVGD: Banana Distribution Evolution ({} iterations)'.format(particles_history[-1][0]), 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'banana_svgd_evolution_extended.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print("✅ Evolution plot saved to: " + save_path)
    plt.close()
    
    # Also create a high-quality final state plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    levels = np.linspace(1e-5, 1.0, 40)  # Start from very small to show full banana
    ax.contour(X1, X2, probs, levels=levels, colors='black', alpha=0.5, linewidths=1.2)
    ax.contourf(X1, X2, probs, levels=levels, cmap='Reds', alpha=0.3)
    
    # Banana curve: x2 = b * (x1^2 - sigma1^2)
    x1_curve = np.linspace(-20, 20, 300)
    x2_curve = banana_model.b * (x1_curve**2 - banana_model.sigma1**2)
    ax.plot(x1_curve, x2_curve, 'b--', linewidth=2.5, alpha=0.8, 
            label='Theoretical Banana Curve', zorder=3)
    
    # Final particles
    final_particles = particles_history[-1][1]
    ax.scatter(final_particles[:, 0], final_particles[:, 1], 
              c='mediumseagreen', s=60, alpha=0.85, edgecolors='darkgreen', 
              linewidths=1.2, label='SVGD Particles ({} iterations)'.format(particles_history[-1][0]),
              zorder=4)
    
    # Statistics
    mean = np.mean(final_particles, axis=0)
    std = np.std(final_particles, axis=0)
    log_probs_final = banana_model.log_prob(final_particles)
    probs_final = np.exp(log_probs_final - log_probs_max)
    high_prob_ratio = np.mean(probs_final > 0.1)
    
    stats_text = 'Mean: ({:.2f}, {:.2f})\nStd: ({:.2f}, {:.2f})\nHigh Prob: {:.1%}'.format(
        mean[0], mean[1], std[0], std[1], high_prob_ratio)
    ax.text(0.02, 0.98, stats_text,
           transform=ax.transAxes, fontsize=12,
           verticalalignment='top', 
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=1.5))
    
    ax.set_xlim(-20, 20)
    ax.set_ylim(-10, 10)
    ax.set_xlabel('$x_1$', fontsize=16)
    ax.set_ylabel('$x_2$', fontsize=16)
    ax.set_title('SVGD: Final Particle Distribution\nBanana Distribution ({} iterations)'.format(particles_history[-1][0]), 
                fontsize=18, fontweight='bold', pad=20)
    ax.legend(fontsize=13, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    save_path_final = os.path.join(save_dir, 'banana_svgd_final_extended.png')
    plt.savefig(save_path_final, dpi=300, bbox_inches='tight')
    print("✅ Final state plot saved to: " + save_path_final)
    plt.close()


def run_banana_final_fix():
    print("🍌 Running Banana Distribution (Final Fix)...")
    print("=" * 60)
    
    # Standard banana distribution: sigma1=10.0 (variance=100) for x1, sigma2=2.0 for x2
    banana_model = BananaDistribution(b=0.03, sigma1=10.0, sigma2=2.0)
    n_particles = 100
    np.random.seed(42)
    
    # Initialize particles in a small cluster near origin to show dramatic evolution
    # This will create a "from circular blob to banana" visual effect
    x0 = np.random.normal([0, 0], [3.0, 3.0], [n_particles, 2])
    
    print("Initialization: {} particles".format(n_particles))
    print("Initial mean: {}, std: {}".format(np.mean(x0, axis=0), np.std(x0, axis=0)))
    
    # Theoretical values for verification
    # x1 ~ N(0, sigma1^2), so mean=0, std=sigma1
    # x2 has mean=0, variance = sigma2^2 + 2*b^2*sigma1^4
    theoretical_x1_std = banana_model.sigma1
    theoretical_x2_var = banana_model.sigma2**2 + 2 * banana_model.b**2 * banana_model.sigma1**4
    theoretical_x2_std = np.sqrt(theoretical_x2_var)
    print("Theoretical: x1 std={:.2f}, x2 std={:.2f}".format(
        theoretical_x1_std, theoretical_x2_std))
    
    # Calculate initial bandwidth
    sq_dist_init = pdist(x0)
    pairwise_dists_init = squareform(sq_dist_init)**2
    median_init = np.median(pairwise_dists_init[pairwise_dists_init > 0])
    h_init = np.sqrt(0.5 * median_init / np.log(n_particles + 1))
    print("Initial bandwidth h: {:.4f}".format(h_init))
    
    svgd = SVGD_Adaptive(initial_h=h_init, max_h=1.5)  # Cap at 1.5
    
    n_iter = 10000  # Increased iterations
    stepsize = 0.003  # Very small step size
    
    print("\nRunning SVGD: {} iterations, stepsize={}, max_h={}".format(
        n_iter, stepsize, svgd.max_h))
    
    particles_history = []
    checkpoints = [0, 100, 500, 1000, 2500, 5000, 7500, 10000]  # More checkpoints
    
    theta = np.copy(x0)
    particles_history.append((0, np.copy(theta)))
    
    fudge_factor = 1e-6
    historical_grad = 0
    alpha = 0.9
    
    bandwidth_history = []
    
    for iter in range(n_iter):
        lnpgrad = banana_model.dlnprob(theta)
        (kxy, dxkxy), h_current = svgd.svgd_kernel(theta, h=-1)
        bandwidth_history.append(h_current)
        
        grad_theta = (np.matmul(kxy, lnpgrad) + dxkxy) / n_particles
        
        if iter == 0:
            historical_grad = historical_grad + grad_theta ** 2
        else:
            historical_grad = alpha * historical_grad + (1 - alpha) * (grad_theta ** 2)
        adj_grad = np.divide(grad_theta, fudge_factor + np.sqrt(historical_grad))
        theta = theta + stepsize * adj_grad
        
        if (iter + 1) in checkpoints:
            particles_history.append((iter + 1, np.copy(theta)))
            mean = np.mean(theta, axis=0)
            std = np.std(theta, axis=0)
            
            log_probs = banana_model.log_prob(theta)
            probs = np.exp(log_probs - np.max(log_probs))
            high_prob_ratio = np.mean(probs > 0.1)
            
            # Also compute g direction statistics
            g = theta[:, 1] - banana_model.b * (theta[:, 0]**2 - banana_model.sigma1**2)
            g_std = np.std(g)
            
            print("  Iter {}: mean={}, std={}, g_std={:.2f}, h={:.4f}, high_prob={:.1%}".format(
                iter + 1, mean, std, g_std, h_current, high_prob_ratio))
    
    print("\n✅ SVGD completed!")
    final_mean = np.mean(theta, axis=0)
    final_std = np.std(theta, axis=0)
    print("Final mean: {}".format(final_mean))
    print("Final std: {}".format(final_std))
    
    # Theoretical comparison
    theoretical_x1_std = banana_model.sigma1
    theoretical_x2_var = banana_model.sigma2**2 + 2 * banana_model.b**2 * banana_model.sigma1**4
    theoretical_x2_std = np.sqrt(theoretical_x2_var)
    print("\n📊 Statistical Verification:")
    print("  x1: sample std={:.2f}, theoretical={:.2f}, error={:.2%}".format(
        final_std[0], theoretical_x1_std, abs(final_std[0] - theoretical_x1_std) / theoretical_x1_std))
    print("  x2: sample std={:.2f}, theoretical={:.2f}, error={:.2%}".format(
        final_std[1], theoretical_x2_std, abs(final_std[1] - theoretical_x2_std) / theoretical_x2_std))
    
    # Check g direction: g = x2 - b*(x1^2 - sigma1^2) should be ~N(0, sigma2^2)
    g_final = theta[:, 1] - banana_model.b * (theta[:, 0]**2 - banana_model.sigma1**2)
    g_mean = np.mean(g_final)
    g_std = np.std(g_final)
    print("  g direction: mean={:.3f} (theoretical=0.0), std={:.2f} (theoretical={:.2f})".format(
        g_mean, g_std, banana_model.sigma2))
    
    print("\nBandwidth range: [{:.4f}, {:.4f}]".format(
        min(bandwidth_history), max(bandwidth_history)))
    
    log_probs_final = banana_model.log_prob(theta)
    probs_final = np.exp(log_probs_final - np.max(log_probs_final))
    high_prob_ratio_final = np.mean(probs_final > 0.1)
    print("Final high probability ratio: {:.2%}".format(high_prob_ratio_final))
    
    print("\n📊 Generating plots...")
    plot_comparison(banana_model, particles_history, save_dir='results')
    
    return theta, particles_history


if __name__ == '__main__':
    np.random.seed(42)
    final_particles, history = run_banana_final_fix()
    print("\n" + "=" * 60)
    print("✅ Final Fix Experiment completed!")
    print("=" * 60)

