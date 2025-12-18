"""
Bayesian Logistic Regression Experiment for SVGD
This is Experiment 2: Quantitative Analysis

Compare SVGD with baselines (SGD/MAP, SGLD) on Bayesian Logistic Regression
"""

import numpy as np
import scipy.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy.matlib as nm
from svgd import SVGD
from bayesian_logistic_regression import BayesianLR
import os
import sys

# Add compatibility for Python 2/3
if sys.version_info[0] < 3:
    from sklearn.cross_validation import train_test_split
else:
    from sklearn.model_selection import train_test_split


class SGD_MAP:
    """SGD for Maximum A Posteriori (MAP) estimation - single particle baseline"""
    
    def __init__(self, model, stepsize=0.01, n_iter=1000):
        self.model = model
        self.stepsize = stepsize
        self.n_iter = n_iter
    
    def fit(self, x0):
        """Fit using SGD"""
        theta = np.copy(x0)
        
        for iter in range(self.n_iter):
            grad = self.model.dlnprob(theta.reshape(1, -1))
            theta = theta + self.stepsize * grad[0]
            
            if (iter + 1) % 1000 == 0:
                print("  SGD iteration {}".format(iter + 1))
        
        return theta.reshape(1, -1)


class SGLD:
    """Stochastic Gradient Langevin Dynamics - MCMC baseline"""
    
    def __init__(self, model, stepsize=0.01, n_iter=1000):
        self.model = model
        self.stepsize = stepsize
        self.n_iter = n_iter
    
    def fit(self, x0):
        """Fit using SGLD"""
        theta = np.copy(x0)
        samples = []
        
        for iter in range(self.n_iter):
            grad = self.model.dlnprob(theta.reshape(1, -1))
            # Add noise for Langevin dynamics
            noise = np.random.normal(0, np.sqrt(2 * self.stepsize), theta.shape)
            theta = theta + self.stepsize * grad[0] + noise
            
            # Collect samples (after burn-in)
            if iter > self.n_iter // 2:
                samples.append(theta.copy())
            
            if (iter + 1) % 1000 == 0:
                print("  SGLD iteration {}".format(iter + 1))
        
        return np.array(samples)


def run_bayesian_lr_experiment():
    """Run Bayesian Logistic Regression experiment comparing SVGD, SGD, SGLD"""
    print("=" * 60)
    print("📊 Running Bayesian Logistic Regression Experiment")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    data_path = '../data/covertype.mat'
    if not os.path.exists(data_path):
        print("Error: Data file not found at {}".format(data_path))
        print("Please ensure covertype.mat is in the data/ directory")
        return
    
    data = scipy.io.loadmat(data_path)
    X_input = data['covtype'][:, 1:]
    y_input = data['covtype'][:, 0]
    y_input[y_input == 2] = -1
    
    N = X_input.shape[0]
    X_input = np.hstack([X_input, np.ones([N, 1])])
    d = X_input.shape[1]
    D = d + 1  # +1 for alpha parameter
    
    print("  Dataset size: {} samples, {} features".format(N, d))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_input, y_input, test_size=0.2, random_state=42)
    
    print("  Train: {} samples, Test: {} samples".format(
        len(X_train), len(y_test)))
    
    # Initialize model
    a0, b0 = 1, 0.01
    model = BayesianLR(X_train, y_train, batchsize=100, a0=a0, b0=b0)
    
    # Initialize particles (same for all methods)
    M = 100  # Number of particles for SVGD
    np.random.seed(42)
    alpha0 = np.random.gamma(a0, b0, M)
    theta0 = np.zeros([M, D])
    for i in range(M):
        theta0[i, :] = np.hstack([
            np.random.normal(0, np.sqrt(1 / alpha0[i]), d), 
            np.log(alpha0[i])
        ])
    
    results = {}
    
    # ===== Method 1: SVGD =====
    print("\n2. Running SVGD...")
    print("   Particles: {}, Iterations: 6000".format(M))
    svgd = SVGD()
    theta_svgd = svgd.update(
        x0=theta0, 
        lnprob=model.dlnprob, 
        bandwidth=-1, 
        n_iter=6000, 
        stepsize=0.05, 
        alpha=0.9, 
        debug=True
    )
    acc_svgd, llh_svgd = model.evaluation(theta_svgd, X_test, y_test)
    results['SVGD'] = {'accuracy': acc_svgd, 'log_likelihood': llh_svgd, 'n_particles': M}
    print("   ✅ SVGD Results:")
    print("      Accuracy: {:.4f}".format(acc_svgd))
    print("      Log-Likelihood: {:.4f}".format(llh_svgd))
    
    # ===== Method 2: SGD (MAP) =====
    print("\n3. Running SGD (MAP)...")
    print("   Single particle, Iterations: 6000")
    sgd_map = SGD_MAP(model, stepsize=0.01, n_iter=6000)
    # Initialize from mean of SVGD initialization
    theta0_map = np.mean(theta0, axis=0)
    theta_sgd = sgd_map.fit(theta0_map)
    acc_sgd, llh_sgd = model.evaluation(theta_sgd, X_test, y_test)
    results['SGD (MAP)'] = {'accuracy': acc_sgd, 'log_likelihood': llh_sgd, 'n_particles': 1}
    print("   ✅ SGD Results:")
    print("      Accuracy: {:.4f}".format(acc_sgd))
    print("      Log-Likelihood: {:.4f}".format(llh_sgd))
    
    # ===== Method 3: SGLD =====
    print("\n4. Running SGLD...")
    print("   Iterations: 6000 (collecting last 3000 samples)")
    sgld = SGLD(model, stepsize=0.01, n_iter=6000)
    theta0_sgld = np.mean(theta0, axis=0)
    theta_sgld_samples = sgld.fit(theta0_sgld)
    # Use all samples for evaluation
    acc_sgld, llh_sgld = model.evaluation(theta_sgld_samples, X_test, y_test)
    results['SGLD'] = {'accuracy': acc_sgld, 'log_likelihood': llh_sgld, 
                      'n_particles': len(theta_sgld_samples)}
    print("   ✅ SGLD Results:")
    print("      Accuracy: {:.4f}".format(acc_sgld))
    print("      Log-Likelihood: {:.4f}".format(llh_sgld))
    
    # ===== Print Summary Table =====
    print("\n" + "=" * 60)
    print("📊 RESULTS SUMMARY")
    print("=" * 60)
    print("{:<15} {:>12} {:>18} {:>12}".format(
        "Method", "Accuracy", "Log-Likelihood", "Particles"))
    print("-" * 60)
    for method, res in results.items():
        print("{:<15} {:>12.4f} {:>18.4f} {:>12}".format(
            method, res['accuracy'], res['log_likelihood'], res['n_particles']))
    print("=" * 60)
    
    # ===== Create Visualization =====
    print("\n5. Generating comparison plot...")
    os.makedirs('results', exist_ok=True)
    
    methods = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in methods]
    log_likelihoods = [results[m]['log_likelihood'] for m in methods]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy comparison
    bars1 = ax1.bar(methods, accuracies, color=['mediumseagreen', 'steelblue', 'coral'], alpha=0.7)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim([min(accuracies) - 0.01, max(accuracies) + 0.01])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                '{:.4f}'.format(acc),
                ha='center', va='bottom', fontsize=10)
    
    # Log-likelihood comparison (filter out inf values for plotting)
    log_likelihoods_plot = [llh if np.isfinite(llh) else -10 for llh in log_likelihoods]
    bars2 = ax2.bar(methods, log_likelihoods_plot, color=['mediumseagreen', 'steelblue', 'coral'], alpha=0.7)
    ax2.set_ylabel('Test Log-Likelihood', fontsize=12)
    ax2.set_title('Test Log-Likelihood Comparison', fontsize=14, fontweight='bold')
    valid_llh = [llh for llh in log_likelihoods if np.isfinite(llh)]
    if valid_llh:
        ax2.set_ylim([min(valid_llh) - 0.5, max(valid_llh) + 0.5])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, llh in zip(bars2, log_likelihoods):
        height = bar.get_height()
        label = '{:.4f}'.format(llh) if np.isfinite(llh) else '-inf'
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                label,
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    save_path = 'results/bayesian_lr_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print("   ✅ Comparison plot saved to: {}".format(save_path))
    plt.close()
    
    # Save results to file
    result_file = 'results/bayesian_lr_results.txt'
    with open(result_file, 'w') as f:
        f.write("Bayesian Logistic Regression Results\n")
        f.write("=" * 60 + "\n\n")
        f.write("{:<15} {:>12} {:>18} {:>12}\n".format(
            "Method", "Accuracy", "Log-Likelihood", "Particles"))
        f.write("-" * 60 + "\n")
        for method, res in results.items():
            f.write("{:<15} {:>12.4f} {:>18.4f} {:>12}\n".format(
                method, res['accuracy'], res['log_likelihood'], res['n_particles']))
    print("   ✅ Results saved to: {}".format(result_file))
    
    print("\n" + "=" * 60)
    print("✅ Experiment 2 (Bayesian Logistic Regression) completed!")
    print("=" * 60)
    
    return results


if __name__ == '__main__':
    # Fix random seed for reproducibility
    np.random.seed(42)
    
    # Run experiment
    results = run_bayesian_lr_experiment()

