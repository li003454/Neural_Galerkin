"""
Extended visualization for Banana Distribution with more iterations
Creates additional analysis plots
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from banana_distribution_final_fix import BananaDistribution, SVGD_Adaptive, run_banana_final_fix
from scipy.spatial.distance import pdist, squareform
import os

def create_convergence_analysis():
    """Create convergence analysis plots"""
    print("📊 Creating convergence analysis plots...")
    
    banana_model = BananaDistribution(b=0.1, sigma=10.0)
    n_particles = 100
    np.random.seed(42)
    
    x0 = np.random.normal([0, 10], [0.5, 1], [n_particles, 2])
    
    sq_dist_init = pdist(x0)
    pairwise_dists_init = squareform(sq_dist_init)**2
    median_init = np.median(pairwise_dists_init[pairwise_dists_init > 0])
    h_init = np.sqrt(0.5 * median_init / np.log(n_particles + 1))
    
    svgd = SVGD_Adaptive(initial_h=h_init, max_h=1.5)
    
    n_iter = 10000
    stepsize = 0.003
    
    theta = np.copy(x0)
    
    # Track metrics over iterations
    metrics = {
        'iter': [],
        'mean_x1': [],
        'mean_x2': [],
        'std_x1': [],
        'std_x2': [],
        'high_prob_ratio': [],
        'bandwidth': [],
        'mean_log_prob': []
    }
    
    fudge_factor = 1e-6
    historical_grad = 0
    alpha = 0.9
    
    log_probs_max = None
    
    print("Running extended simulation for analysis...")
    for iter in range(n_iter):
        lnpgrad = banana_model.dlnprob(theta)
        (kxy, dxkxy), h_current = svgd.svgd_kernel(theta, h=-1)
        
        grad_theta = (np.matmul(kxy, lnpgrad) + dxkxy) / n_particles
        
        if iter == 0:
            historical_grad = historical_grad + grad_theta ** 2
        else:
            historical_grad = alpha * historical_grad + (1 - alpha) * (grad_theta ** 2)
        adj_grad = np.divide(grad_theta, fudge_factor + np.sqrt(historical_grad))
        theta = theta + stepsize * adj_grad
        
        # Record metrics every 100 iterations
        if (iter + 1) % 100 == 0 or iter == 0:
            mean = np.mean(theta, axis=0)
            std = np.std(theta, axis=0)
            
            log_probs = banana_model.log_prob(theta)
            if log_probs_max is None:
                log_probs_max = np.max(log_probs)
            probs = np.exp(log_probs - log_probs_max)
            high_prob_ratio = np.mean(probs > 0.1)
            mean_log_prob = np.mean(log_probs)
            
            metrics['iter'].append(iter + 1)
            metrics['mean_x1'].append(mean[0])
            metrics['mean_x2'].append(mean[1])
            metrics['std_x1'].append(std[0])
            metrics['std_x2'].append(std[1])
            metrics['high_prob_ratio'].append(high_prob_ratio)
            metrics['bandwidth'].append(h_current)
            metrics['mean_log_prob'].append(mean_log_prob)
    
    # Create convergence plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Mean position
    ax = axes[0, 0]
    ax.plot(metrics['iter'], metrics['mean_x1'], 'o-', label='Mean $x_1$', 
            color='steelblue', markersize=3, linewidth=1.5)
    ax.plot(metrics['iter'], metrics['mean_x2'], 's-', label='Mean $x_2$', 
            color='coral', markersize=3, linewidth=1.5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=10, color='gray', linestyle='--', alpha=0.5, label='Target $x_2$')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Mean Position', fontsize=11)
    ax.set_title('Mean Position Convergence', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Standard deviation
    ax = axes[0, 1]
    ax.plot(metrics['iter'], metrics['std_x1'], 'o-', label='Std $x_1$', 
            color='steelblue', markersize=3, linewidth=1.5)
    ax.plot(metrics['iter'], metrics['std_x2'], 's-', label='Std $x_2$', 
            color='coral', markersize=3, linewidth=1.5)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Standard Deviation', fontsize=11)
    ax.set_title('Particle Spread Evolution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: High probability ratio
    ax = axes[0, 2]
    ax.plot(metrics['iter'], np.array(metrics['high_prob_ratio']) * 100, 'o-', 
            color='mediumseagreen', markersize=3, linewidth=1.5)
    ax.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Target (70%)')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('High Probability Ratio (%)', fontsize=11)
    ax.set_title('Convergence Quality', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Bandwidth evolution
    ax = axes[1, 0]
    ax.plot(metrics['iter'], metrics['bandwidth'], 'o-', 
            color='purple', markersize=3, linewidth=1.5)
    ax.axhline(y=svgd.max_h, color='red', linestyle='--', alpha=0.5, 
               label='Max Bandwidth ({})'.format(svgd.max_h))
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Bandwidth $h$', fontsize=11)
    ax.set_title('Bandwidth Evolution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Mean log probability
    ax = axes[1, 1]
    ax.plot(metrics['iter'], metrics['mean_log_prob'], 'o-', 
            color='darkorange', markersize=3, linewidth=1.5)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Mean Log Probability', fontsize=11)
    ax.set_title('Average Log Probability', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Combined view
    ax = axes[1, 2]
    ax2 = ax.twinx()
    
    line1 = ax.plot(metrics['iter'], np.array(metrics['high_prob_ratio']) * 100, 
                    'o-', color='mediumseagreen', markersize=3, linewidth=1.5, 
                    label='High Prob Ratio (%)')
    line2 = ax2.plot(metrics['iter'], metrics['bandwidth'], 's-', 
                     color='purple', markersize=3, linewidth=1.5, 
                     label='Bandwidth')
    
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('High Prob Ratio (%)', fontsize=11, color='mediumseagreen')
    ax2.set_ylabel('Bandwidth', fontsize=11, color='purple')
    ax.set_title('Convergence Metrics', fontsize=12, fontweight='bold')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('SVGD Convergence Analysis ({} iterations)'.format(n_iter), 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    os.makedirs('results', exist_ok=True)
    save_path = 'results/banana_convergence_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print("✅ Convergence analysis saved to: " + save_path)
    plt.close()
    
    return metrics


if __name__ == '__main__':
    print("=" * 60)
    print("Creating Extended Visualizations")
    print("=" * 60)
    
    np.random.seed(42)
    metrics = create_convergence_analysis()
    
    print("\n" + "=" * 60)
    print("✅ Extended visualizations completed!")
    print("=" * 60)

