# step2_initial_sampling.py

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm, uniform
from jax.flatten_util import ravel_pytree

# Import our custom modules
from config import PROBLEM_DATA, NETWORK_PARAMS, SAMPLING_DATA, OUTPUT_PATHS
from nn import ShallowNetKdV
from exact_solutions import kdv_two_soliton

def acceptance_rejection_sampling():
    """
    Generates initial particle distribution by sampling from |u(x, 0)|
    using the Acceptance-Rejection method.
    """
    print("🚀 Starting initial particle sampling using Acceptance-Rejection...")
    
    # --- Setup ---
    # 设定随机种子以保证可复现性 (注意：scipy/numpy使用不同的随机状态)
    np.random.seed(SAMPLING_DATA['seed'])
    
    # 定义目标分布 p(x) 正比于 |u(x,0)|
    # 我们需要一个能在标量上工作的版本
    def target_dist_unnormalized(x):
        # 我们从解析解采样，而不是拟合的网络，这样更精确
        u0 = kdv_two_soliton(jnp.array([x]), t=0.0)
        return jnp.abs(u0)

    # 定义提议分布 q(x)，这里是一个高斯分布
    proposal_dist = norm(
        loc=SAMPLING_DATA['proposal_dist_mean'],
        scale=SAMPLING_DATA['proposal_dist_std']
    )
    
    # 包络函数的常数 M，使得 M*q(x) >= p(x)
    C = SAMPLING_DATA['scaling_constant_C']
    
    particles = []
    
    # --- Sampling Loop ---
    print(f"Generating {SAMPLING_DATA['num_particles']} samples...")
    for _ in range(SAMPLING_DATA['num_particles']):
        it = 0
        while it < SAMPLING_DATA['max_rejection_iters']:
            # 1. 从提议分布 q(x) 中采样一个点 y
            y = proposal_dist.rvs()
            
            # 2. 从 [0, 1] 的均匀分布中采样一个点 u
            u = uniform.rvs()
            
            # 3. 计算接受概率 alpha = p(y) / (C * q(y))
            acceptance_ratio = target_dist_unnormalized(y) / (C * proposal_dist.pdf(y))
            
            # 4. 接受或拒绝
            if u <= acceptance_ratio:
                particles.append(y)
                break # 采样成功，跳出内层循环
            
            it += 1
            
        if it >= SAMPLING_DATA['max_rejection_iters']:
            raise RuntimeError(f"Acceptance-Rejection sampling did not converge after {SAMPLING_DATA['max_rejection_iters']} iterations.")

    print(f"✅ Sampling complete. Generated {len(particles)} particles.")
    
    # --- Save Results ---
    particles_array = jnp.array(particles).reshape(-1, 1)
    jnp.save(OUTPUT_PATHS['initial_particles'], particles_array)
    print(f"💾 Initial particles saved to '{OUTPUT_PATHS['initial_particles']}'")
    
    return particles_array

def visualize_initial_state():
    """
    Loads the fitted network and sampled particles, then generates a comparison plot.
    """
    print("\n📊 Visualizing initial state...")
    
    # --- Load Data ---
    # 加载拟合好的网络参数
    try:
        theta_flat = jnp.load(OUTPUT_PATHS['initial_theta'])
    except FileNotFoundError:
        print(f"Error: '{OUTPUT_PATHS['initial_theta']}' not found. Please run step1_fit_initial_condition.py first.")
        return
        
    # 加载采样好的粒子
    try:
        particles = jnp.load(OUTPUT_PATHS['initial_particles'])
    except FileNotFoundError:
        print(f"Error: '{OUTPUT_PATHS['initial_particles']}' not found. Did the sampling step run correctly?")
        return

    # --- Reconstruct Network ---
    net = ShallowNetKdV(m=NETWORK_PARAMS['m'], L=NETWORK_PARAMS['L'])
    # 我们需要一个虚拟输入来获取参数的 PyTree 结构
    dummy_x = jnp.ones((1, PROBLEM_DATA['d']))
    # Unravel the flat parameters into a PyTree
    _, unravel_fn = ravel_pytree(net.init(jax.random.PRNGKey(0), dummy_x)['params'])

    # --- Plotting ---
    x_plot = jnp.linspace(PROBLEM_DATA['domain'][0], PROBLEM_DATA['domain'][1], PROBLEM_DATA['N'])
    u_true_plot = PROBLEM_DATA['initial_fn'](x_plot)
    u_fitted_plot = net.apply({'params': unravel_fn(theta_flat)}, x_plot.reshape(-1, 1))
    
    plt.style.use('seaborn-v0_8-whitegrid') # 使用一个好看的绘图风格
    plt.figure(figsize=(12, 7))
    
    # 绘制曲线
    plt.plot(x_plot, u_true_plot, 'k--', linewidth=2.5, label='Truth')
    plt.plot(x_plot, u_fitted_plot, color='darkviolet', linewidth=2.5, label='Neural Galerkin (Fitted)')
    
    # 绘制粒子
    # 在y=-0.2的位置绘制，使其清晰可见
    plt.scatter(particles, -0.2 * np.ones_like(particles), c='mediumseagreen', marker='x', s=50, label='Particles')
    
    # 美化图表
    plt.title('Initial State: Fitted Solution and Sampled Particles', fontsize=16)
    plt.xlabel('x (Spatial Domain)', fontsize=12)
    plt.ylabel('u(x, 0) (Numerical Solution)', fontsize=12)
    plt.xlim(PROBLEM_DATA['domain'])
    plt.ylim(-0.5, max(np.max(u_true_plot), np.max(u_fitted_plot)) * 1.1)
    plt.legend(fontsize=12, loc='upper right', frameon=True, shadow=True)
    plt.tight_layout()
    
    # 保存图像
    output_fig_path = os.path.join(os.path.dirname(OUTPUT_PATHS['initial_particles']), 'initial_state.png')
    plt.savefig(output_fig_path, dpi=300)
    print(f"🖼️ Plot saved to '{output_fig_path}'")
    
    plt.show()

if __name__ == "__main__":
    acceptance_rejection_sampling()
    visualize_initial_state()