# step2_initial_sampling_AC.py

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm, uniform
from jax.flatten_util import ravel_pytree

# Import our custom modules
from config import (
    AC_PROBLEM_DATA, AC_NETWORK_PARAMS, AC_OUTPUT_PATHS, AC_SAMPLING_DATA,
    ac_initial_condition
)
from nn import DeepNetAC

def acceptance_rejection_sampling_AC():
    """
    Generates initial particle distribution by sampling from |u(x, 0)|
    using the Acceptance-Rejection method for Allen-Cahn equation.
    """
    print("🚀 Starting initial particle sampling for AC equation using Acceptance-Rejection...")
    
    # --- Setup ---
    # 设定随机种子以保证可复现性
    np.random.seed(AC_SAMPLING_DATA['seed'])
    
    # 定义目标分布 p(x) 正比于 |u(x,0)|
    def target_dist_unnormalized(x):
        # 使用预计算的初始条件
        u0 = AC_PROBLEM_DATA['initial_fn'](np.array([x]))
        return np.abs(u0)
    
    # 定义提议分布 q(x)，这里是一个高斯分布
    proposal_dist = norm(
        loc=AC_SAMPLING_DATA['proposal_dist_mean'],
        scale=AC_SAMPLING_DATA['proposal_dist_std']
    )
    
    # 包络函数的常数 M，使得 M*q(x) >= p(x)
    C = AC_SAMPLING_DATA['scaling_constant_C']
    
    particles = []
    
    # --- Sampling Loop ---
    print(f"Generating {AC_SAMPLING_DATA['num_particles']} samples...")
    for i in range(AC_SAMPLING_DATA['num_particles']):
        it = 0
        while it < AC_SAMPLING_DATA['max_rejection_iters']:
            # 1. 从提议分布 q(x) 中采样一个点 y
            y = proposal_dist.rvs()
            
            # 确保 y 在域内（周期性边界条件，可以wrap）
            # 但为了简单，我们限制在 [0, 2π]
            while y < AC_PROBLEM_DATA['domain'][0] or y > AC_PROBLEM_DATA['domain'][1]:
                y = proposal_dist.rvs()
            
            # 2. 从 [0, 1] 的均匀分布中采样一个点 u
            u = uniform.rvs()
            
            # 3. 计算接受概率 alpha = p(y) / (C * q(y))
            try:
                acceptance_ratio = target_dist_unnormalized(y) / (C * proposal_dist.pdf(y))
                # 确保接受概率在 [0, 1] 范围内
                acceptance_ratio = min(1.0, max(0.0, acceptance_ratio))
            except:
                acceptance_ratio = 0.0
            
            # 4. 接受或拒绝
            if u <= acceptance_ratio:
                particles.append(y)
                break  # 采样成功，跳出内层循环
            
            it += 1
        
        if it >= AC_SAMPLING_DATA['max_rejection_iters']:
            # 如果拒绝采样失败，使用均匀采样作为后备
            y = np.random.uniform(
                AC_PROBLEM_DATA['domain'][0],
                AC_PROBLEM_DATA['domain'][1]
            )
            particles.append(y)
            if (i + 1) % 10 == 0:
                print(f"  Warning: Used uniform sampling for particle {i+1}")
    
    print(f"✅ Sampling complete. Generated {len(particles)} particles.")
    
    # --- Save Results ---
    particles_array = jnp.array(particles).reshape(-1, 1)
    output_path = 'data/particle0_AC.npy'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    jnp.save(output_path, particles_array)
    print(f"💾 Initial particles saved to '{output_path}'")
    
    return particles_array

def visualize_initial_state_AC():
    """
    Loads the fitted network and sampled particles, then generates a comparison plot.
    """
    print("\n📊 Visualizing initial state for AC equation...")
    
    # --- Load Data ---
    # 加载拟合好的网络参数
    try:
        theta_flat = jnp.load(AC_OUTPUT_PATHS['initial_theta'])
    except FileNotFoundError:
        print(f"Error: '{AC_OUTPUT_PATHS['initial_theta']}' not found. Please run step1_fit_initial_condition_AC.py first.")
        return
        
    # 加载采样好的粒子
    try:
        particles = jnp.load('data/particle0_AC.npy')
    except FileNotFoundError:
        print(f"Error: 'data/particle0_AC.npy' not found. Did the sampling step run correctly?")
        return

    # --- Reconstruct Network ---
    net = DeepNetAC(
        m=AC_NETWORK_PARAMS['m'],
        l=AC_NETWORK_PARAMS['l'],
        L=AC_NETWORK_PARAMS['L']
    )
    # 我们需要一个虚拟输入来获取参数的 PyTree 结构
    dummy_x = jnp.ones((1, AC_PROBLEM_DATA['d']))
    # Unravel the flat parameters into a PyTree
    _, unravel_fn = ravel_pytree(net.init(jax.random.PRNGKey(0), dummy_x)['params'])

    # --- Plotting ---
    x_plot = jnp.linspace(
        AC_PROBLEM_DATA['domain'][0],
        AC_PROBLEM_DATA['domain'][1],
        AC_PROBLEM_DATA['N']
    )
    u_true_plot = AC_PROBLEM_DATA['initial_fn'](x_plot)
    u_fitted_plot = net.apply({'params': unravel_fn(theta_flat)}, x_plot.reshape(-1, 1))
    
    plt.style.use('seaborn-v0_8-whitegrid')  # 使用一个好看的绘图风格
    plt.figure(figsize=(12, 7))
    
    # 绘制曲线
    plt.plot(x_plot, u_true_plot, 'k--', linewidth=2.5, label='Truth')
    plt.plot(x_plot, u_fitted_plot, color='darkviolet', linewidth=2.5, label='Neural Galerkin (Fitted)')
    
    # 绘制粒子
    # 在y=-1.2的位置绘制，使其清晰可见（AC方程的值域在[-1, 1]附近）
    particles_1d = particles.squeeze()
    plt.scatter(particles_1d, -1.2 * np.ones_like(particles_1d), 
                c='mediumseagreen', marker='x', s=50, label='Particles', zorder=5)
    
    # 美化图表
    plt.title('Initial State: Fitted Solution and Sampled Particles (Allen-Cahn)', fontsize=16)
    plt.xlabel('x (Spatial Domain)', fontsize=12)
    plt.ylabel('u(x, 0) (Numerical Solution)', fontsize=12)
    plt.xlim(AC_PROBLEM_DATA['domain'])
    
    # 设置 y 轴范围
    y_max = max(np.max(u_true_plot), np.max(u_fitted_plot))
    y_min = min(np.min(u_true_plot), np.min(u_fitted_plot))
    plt.ylim(y_min - 0.3, y_max * 1.1)
    
    plt.legend(fontsize=12, loc='upper right', frameon=True, shadow=True)
    plt.tight_layout()
    
    # 保存图像
    output_fig_path = os.path.join('data', 'initial_state_AC.png')
    os.makedirs(os.path.dirname(output_fig_path), exist_ok=True)
    plt.savefig(output_fig_path, dpi=300)
    print(f"🖼️ Plot saved to '{output_fig_path}'")
    
    plt.show()
    
    # 打印统计信息
    print("\n📊 Sampling Statistics:")
    print(f"   Number of particles: {len(particles_1d)}")
    print(f"   Particle range: [{np.min(particles_1d):.4f}, {np.max(particles_1d):.4f}]")
    print(f"   Particle mean: {np.mean(particles_1d):.4f}")
    print(f"   Particle std: {np.std(particles_1d):.4f}")

if __name__ == "__main__":
    acceptance_rejection_sampling_AC()
    visualize_initial_state_AC()

