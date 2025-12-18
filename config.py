# config.py

from exact_solutions import kdv_two_soliton

# --- Problem Definition ---
PROBLEM_DATA = {
    'd': 1,  # Spatial dimension
    'domain': [-20.0, 40.0],
    'L': 60.0, # Length of the domain (40 - (-20))
    'N': 512,  # Number of points for plotting
    'initial_fn': lambda x: kdv_two_soliton(x, t=0.0),
    'exact_sol': kdv_two_soliton,
}

# --- Initial Particle Sampling ---
SAMPLING_DATA = {
    'seed': 1, # 使用一个不同的种子
    'num_particles': 100,
    
    # --- Acceptance-Rejection Sampling Config ---
    'proposal_dist_mean': -5.0,   # 提议分布（高斯）的均值
    'proposal_dist_std': 4.5,     # 提议分布（高斯）的标准差
    'scaling_constant_C': 29.0,   # 用于包络函数的常数
    'max_rejection_iters': 1000,  # 防止无限循环的最大尝试次数
}


SVGD_PARAMS = {
    'enabled': True,            # 是否启用自适应采样
    'corrected': False,         # 是否使用带噪声的修正版SVGD
    'steps': 250,               # 每个主时间步内的SVGD迭代次数 (适中)
    'epsilon': 1e-2,            # SVGD的步长 (svgd_dt) (适中)
    'gamma': 0.25,              # 目标分布的温度参数
    'h': 0.05,                   # RBF核的宽度 (bandwidth) (适中)
    'boundary_penalty': True,   # 是否启用边界惩罚
    'boundary_strength': 1e4,  # 边界惩罚强度
    # 'noise_strength': 1e-3,   # (可选) 如果你想用解耦的噪声，可以启用这个
}

# --- Time Evolution Configuration ---
EVOLUTION_PARAMS = {
    't_final': 4.0,
    'dt': 1e-2,  # 主时间步长
    'ridge_lambda': 1e-4,  # 增强正则化，抑制外推区域的振荡
    'num_anchor_particles': 80,  # 增加锚点粒子数量，为"信息真空区"提供更密集约束
    'theta_weight_decay': 3e-3, # 增强参数衰减，促进平滑解
    
    # 自适应锚点粒子配置
    'adaptive_anchors': True,      # 启用自适应锚点粒子
    'extra_anchor_density': 3.0,   # 在稀疏区域每个区域添加的额外锚点数量
}

# --- Network Architecture ---
NETWORK_PARAMS = {
    'm': 20, # Number of neurons in the feature layer
    'L': PROBLEM_DATA['L']
}

# --- Initial Condition Fitting ---
TRAINING_DATA = {
    'seed': 0,
    'batch_size': 2000,
    'epochs': 200000,
    'gamma': 1e-3, # Adam learning rate
    'scheduler': None, # Optional: optax learning rate scheduler
}

OUTPUT_PATHS = {
    'initial_theta': 'data/theta0.npy',
    'initial_particles': 'data/particle0.npy' # <-- 确保这一行存在！
}

# ============================================================================
# --- Allen-Cahn (AC) Equation Configuration ---
# ============================================================================

from exact_solutions import exactAC
import numpy as np
import jax.numpy as jnp

# Precompute initial condition for efficiency
_t_full_AC, _u_full_AC = exactAC()
_x_full_AC = np.linspace(0, 2 * np.pi, _u_full_AC.shape[0])
_u0_full_AC = _u_full_AC[:, 0]  # Initial condition at t=0

def ac_initial_condition(x):
    """Efficient initial condition for Allen-Cahn equation."""
    x = np.asarray(x)
    if x.ndim == 2:
        x = x.squeeze()
    u_interp = np.interp(x, _x_full_AC, _u0_full_AC)
    return jnp.asarray(u_interp)

# --- AC Problem Definition ---
AC_PROBLEM_DATA = {
    'd': 1,  # Spatial dimension
    'domain': [0.0, 2 * jnp.pi],
    'L': 2 * jnp.pi,  # Length of the domain
    'N': 512,  # Number of points for plotting
    'initial_fn': ac_initial_condition,
    'exact_sol': None,  # Will use ac_solution function directly
}

# --- AC Initial Particle Sampling ---
AC_SAMPLING_DATA = {
    'seed': 1,
    'num_particles': 100,
    'proposal_dist_mean': np.pi,  # 提议分布（高斯）的均值，设为域的中点
    'proposal_dist_std': 1.5,     # 提议分布（高斯）的标准差
    'scaling_constant_C': 2.0,    # 用于包络函数的常数（AC 方程的值域较小）
    'max_rejection_iters': 1000,  # 防止无限循环的最大尝试次数
}

# --- AC SVGD Parameters ---
AC_SVGD_PARAMS = {
    'enabled': True,
    'corrected': False,
    'steps': 150,  # 减少 SVGD 步数以加快速度（从 250 降到 150）
    'epsilon': 1e-2,
    'gamma': 0.25,
    'h': 0.05,
    'boundary_penalty': True,
    'boundary_strength': 1e4,
}

# --- AC Time Evolution Configuration ---
AC_EVOLUTION_PARAMS = {
    't_final': 12.0,  # Match the exact solution time range
    'dt': 1e-2,  # 减小时间步长以提高数值稳定性（AC 方程需要更小的步长）
    'ridge_lambda': 1e-3,  # 增强正则化（从 1e-4 增加到 1e-3）以提高稳定性
    'num_anchor_particles': 60,  # 减少锚点粒子数量以加快速度（从 80 到 60）
    'theta_weight_decay': 5e-3,  # 增强参数衰减（从 3e-3 增加到 5e-3）以促进平滑解
    'adaptive_anchors': True,  # 启用自适应锚点粒子
    'extra_anchor_density': 2.0,  # 减少额外锚点密度（从 3.0 到 2.0）
}

# --- AC Network Architecture ---
AC_NETWORK_PARAMS = {
    'm': 20,  # Number of neurons in the feature layer (reduced from 40 for speed)
    'l': 2,   # Number of hidden layers (reduced from 3 for speed)
    'L': AC_PROBLEM_DATA['L']
}

# --- AC Initial Condition Fitting ---
AC_TRAINING_DATA = {
    'seed': 0,
    'batch_size': 2000,
    'epochs': 200000,
    'gamma': 5e-3,  # Adam learning rate (increased for faster convergence)
    'scheduler': None,  # Optional: optax learning rate scheduler
}

# --- AC Output Paths ---
AC_OUTPUT_PATHS = {
    'initial_theta': 'data/theta0_AC.npy',
    'initial_particles': 'data/particle0_AC.npy',
}