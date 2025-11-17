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