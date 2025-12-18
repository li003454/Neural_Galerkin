# Neural Galerkin with SVGD: Solving PDEs with Adaptive Sampling

This repository implements a **Coupled Neural Galerkin-SVGD** method for solving partial differential equations (PDEs). The method combines neural network parameterization with Stein Variational Gradient Descent (SVGD) for adaptive particle sampling, achieving high accuracy in solving nonlinear PDEs.

## Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Detailed Workflow](#detailed-workflow)
- [Configuration](#configuration)
- [Results](#results)
- [Baseline Comparison](#baseline-comparison)

---

## 🎯 Overview

This project solves two PDEs:

1. **Korteweg-de Vries (KdV) Equation**: 
   - `u_t + 6uu_x + u_{xxx} = 0`
   - Two-soliton solution

2. **Allen-Cahn (AC) Equation**:
   - `u_t = εu_{xx} + a(x,t)(u - u^3)`
   - Reaction-diffusion equation

### Key Features

- **Neural Network Parameterization**: Uses shallow/deep networks to represent the solution
- **Adaptive Particle Sampling**: SVGD dynamically updates sampling points
- **Adaptive Anchor Particles**: Automatically supplements sparse regions
- **Galerkin Projection**: Projects PDE onto parameter space for efficient time evolution

---

## 🔧 Environment Setup

### Prerequisites

- Python 3.8+
- pip

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd Neural\ Galerkin
```

2. **Create and activate virtual environment**:
```bash
python -m venv neural_galerkin
source neural_galerkin/bin/activate  # On macOS/Linux
# or
neural_galerkin\Scripts\activate  # On Windows
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import jax; import flax; print('Installation successful!')"
```

---

## 📁 Project Structure

```
Neural Galerkin/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── config.py                          # Centralized configuration
│
├── Core Modules/
│   ├── nn.py                         # Neural network architectures
│   ├── physics.py                    # PDE residual functions
│   ├── integrator.py                 # RK4 time integrator
│   ├── sampler.py                    # SVGD particle updater
│   ├── utils.py                      # Galerkin assembly functions
│   └── exact_solutions.py            # Exact/reference solutions
│
├── Main Runners/
│   ├── runner.py                     # KdV equation solver
│   └── runner_AC.py                   # Allen-Cahn equation solver
│
└── Neural_Galerkin_Sampling/
    ├── Initial_Fit/                  # Step 1: Fit initial condition
    │   ├── step1_fit_initial_condition.py
    │   ├── step1_fit_initial_condition_AC.py
    │   ├── theta0.npy                # Saved initial parameters (KdV)
    │   └── theta0_AC.npy             # Saved initial parameters (AC)
    │
    ├── Initial_sampling/             # Step 2: Initialize particles
    │   ├── step2_initial_sampling.py
    │   ├── step2_initial_sampling_AC.py
    │   ├── particle0.npy             # Saved initial particles (KdV)
    │   └── particle0_AC.npy          # Saved initial particles (AC)
    │
    ├── baselines/                     # Baseline methods
    │   ├── pinn_kdv.py               # PINN baseline for KdV
    │   └── pinn_output/              # PINN results
    │
    └── results/                       # Final results
        ├── run_output_kdv/           # KdV results
        └── run_output_AC/            # AC results
│
└── Stein-Variational-Gradient-Descent/  # SVGD reproduction experiments
    ├── python/
    │   ├── banana_distribution_final_fix.py  # 2D Banana distribution
    │   ├── run_bayesian_lr_experiment.py     # Bayesian logistic regression
    │   └── results/                          # SVGD experiment results
    └── data/                                 # Datasets
```

---

## 🚀 Quick Start

### For KdV Equation

```bash
# Step 1: Fit initial condition
cd Neural_Galerkin_Sampling/Initial_Fit
python step1_fit_initial_condition.py

# Step 2: Initialize particles
cd ../Initial_sampling
python step2_initial_sampling.py

# Step 3: Run time evolution
cd ../..
python runner.py
```

### For Allen-Cahn Equation

```bash
# Step 1: Fit initial condition
cd Neural_Galerkin_Sampling/Initial_Fit
python step1_fit_initial_condition_AC.py

# Step 2: Initialize particles
cd ../Initial_sampling
python step2_initial_sampling_AC.py

# Step 3: Run time evolution
cd ../..
python runner_AC.py
```

---

## 📖 Detailed Workflow

### Step 1: Fit Initial Condition

**Purpose**: Train a neural network to accurately represent the initial condition `u(x, 0)`.

**What it does**:
- Samples points from the domain
- Computes exact initial condition values
- Trains network using MSE loss
- Saves initial parameters `theta0.npy` to `data/` directory

**Output**: 
- `data/theta0.npy` (KdV) or `data/theta0_AC.npy` (AC)
- The `data/` directory will be created automatically if it doesn't exist

**Example**:
```bash
cd Neural_Galerkin_Sampling/Initial_Fit
python step1_fit_initial_condition.py
```

**Note**: If you see files in `Neural_Galerkin_Sampling/Initial_Fit/`, you may need to copy them to `data/` or update `config.py` paths.

### Step 2: Initialize Particles

**Purpose**: Generate initial particle positions for SVGD sampling.

**What it does**:
- Uses acceptance-rejection sampling based on initial condition
- Generates particles concentrated in high-activity regions
- Saves initial particles `particle0.npy` to `data/` directory

**Output**: 
- `data/particle0.npy` (KdV) or `data/particle0_AC.npy` (AC)
- The `data/` directory will be created automatically if it doesn't exist

**Example**:
```bash
cd Neural_Galerkin_Sampling/Initial_sampling
python step2_initial_sampling.py
```

**Note**: If you see files in `Neural_Galerkin_Sampling/Initial_sampling/`, you may need to copy them to `data/` or update `config.py` paths.

### Step 3: Time Evolution

**Purpose**: Solve the PDE over time using Neural Galerkin + SVGD.

**What it does**:
1. Loads initial parameters and particles
2. For each time step:
   - Combines dynamic particles with adaptive anchor particles
   - Computes Galerkin projection: `M θ̇ = F`
   - Updates particles using SVGD
   - Updates network parameters using RK4
   - Adaptively adds anchor particles in sparse regions
3. Saves results and generates plots

**Output**: 
- `Neural_Galerkin_Sampling/results/run_output_kdv/` (or `run_output_AC/`)
  - `evolution_plot.png`: Solution snapshots
  - `spacetime_solution.png`: Full spacetime visualization
  - `l2_error_analysis.png`: Error evolution
  - `l2_error_data.json`: Quantitative error data

**Example**:
```bash
python runner.py  # For KdV
# or
python runner_AC.py  # For AC
```

---

## ⚙️ Configuration

All hyperparameters are centralized in `config.py`. Key parameters:

### Problem Definition
```python
PROBLEM_DATA = {
    'domain': [-20.0, 40.0],  # Spatial domain
    'L': 60.0,                 # Domain length
}
```

### Network Architecture
```python
NETWORK_PARAMS = {
    'm': 20,  # Number of neurons (KdV)
    # For AC: 'm': 20, 'l': 2 (layers)
}
```

### SVGD Parameters
```python
SVGD_PARAMS = {
    'enabled': True,
    'steps': 250,              # SVGD iterations per time step
    'epsilon': 1e-2,           # SVGD step size
    'gamma': 0.25,              # Temperature parameter
    'h': 0.05,                 # RBF kernel bandwidth
}
```

### Evolution Parameters
```python
EVOLUTION_PARAMS = {
    't_final': 4.0,            # Final time
    'dt': 1e-2,                # Time step size
    'ridge_lambda': 1e-4,      # Regularization
    'num_anchor_particles': 80, # Base anchor particles
    'adaptive_anchors': True,   # Enable adaptive anchors
    'extra_anchor_density': 3.0, # Extra anchors per sparse region
}
```

**To modify parameters**: Edit `config.py` and re-run the experiments.

### File Paths Configuration

The default paths in `config.py` are:
```python
OUTPUT_PATHS = {
    'initial_theta': 'data/theta0.npy',
    'initial_particles': 'data/particle0.npy'
}
```

**Important**: 
- The `data/` directory will be created automatically when running Step 1 and Step 2
- If you have existing files in `Neural_Galerkin_Sampling/Initial_Fit/` or `Initial_sampling/`, you can either:
  1. Copy them to `data/` directory, or
  2. Update the paths in `config.py` to point to the actual file locations

---

## 📊 Results

### Output Files

After running the experiments, results are saved in:

- **KdV**: `Neural_Galerkin_Sampling/results/run_output_kdv/`
- **AC**: `Neural_Galerkin_Sampling/results/run_output_AC/`

### Generated Plots

1. **`evolution_plot.png`**: Solution at different time points
   - Black dashed: Exact solution
   - Purple solid: Neural Galerkin prediction
   - Green crosses: Particle positions

2. **`spacetime_solution.png`**: Full spacetime visualization
   - Heatmap showing solution evolution

3. **`l2_error_analysis.png`**: Error analysis
   - L2 error over time
   - Spatial error distribution
   - Error vs particle density

4. **`l2_error_data.json`**: Quantitative metrics
   - Final L2 error
   - Mean/max errors
   - Relative errors

### Expected Performance

**KdV Equation**:
- Final relative L2 error: ~1e-3 to 1e-4
- Soliton interactions accurately captured

**Allen-Cahn Equation**:
- Final relative L2 error: ~1e-2 to 1e-3
- Reaction-diffusion dynamics preserved

---

## 🧪 SVGD Experiments

This repository also includes SVGD (Stein Variational Gradient Descent) reproduction experiments:

### Banana Distribution (Qualitative Analysis)

```bash
cd Stein-Variational-Gradient-Descent/python
python banana_distribution_final_fix.py
```

**Output**: `results/banana_svgd_evolution_extended.png`, `banana_svgd_final_extended.png`

### Bayesian Logistic Regression (Quantitative Analysis)

```bash
cd Stein-Variational-Gradient-Descent/python
python run_bayesian_lr_experiment.py
```

**Output**: `results/bayesian_lr_comparison.png`, `bayesian_lr_results.txt`

**Note**: Requires `covertype.mat` dataset in `Stein-Variational-Gradient-Descent/data/`

---

## 🔬 Baseline Comparison

### PINN Baseline

A Physics-Informed Neural Network (PINN) baseline is provided for comparison:

```bash
cd Neural_Galerkin_Sampling/baselines
python pinn_kdv.py
```

**Output**: `pinn_output/` directory with:
- Training loss history
- Solution snapshots
- Error analysis
- Comparison with exact solution

### Comparison Metrics

Compare results using:
- **L2 Error**: `l2_error_data.json` files
- **Visualization**: Side-by-side plot comparison
- **Computational Cost**: Training time and memory usage

---

## 🔍 Key Algorithms

### 1. Neural Galerkin Method

Projects the PDE onto the parameter space:
\[
\langle J, J \rangle \dot{\theta} = -\langle J, f(u) \rangle
\]

where:
- \(J = \frac{\partial u_\theta}{\partial \theta}\): Jacobian matrix
- \(f(u)\): PDE spatial residual
- \(\theta\): Network parameters

### 2. SVGD Particle Update

Updates particles to minimize PDE residual:
\[
\phi(x_i) = \frac{1}{n}\sum_{j=1}^n [K(x_i, x_j) \nabla_{x_j} \log \mu(x_j) + \nabla_{x_j} K(x_i, x_j)]
\]
\[
x_i \leftarrow x_i + \epsilon \phi(x_i)
\]

### 3. Adaptive Anchor Particles

- **Base anchors**: Uniformly distributed (80 particles)
- **Adaptive anchors**: Added in sparse regions detected every 10 steps
- **Detection**: Density analysis using sliding window
- **Purpose**: Ensure coverage in "information vacuum" regions

---

## 🐛 Troubleshooting

### Common Issues

1. **Import errors**:
   ```bash
   # Ensure virtual environment is activated
   source neural_galerkin/bin/activate
   ```

2. **File not found errors**:
   - Ensure Step 1 and Step 2 are run before Step 3
   - Check that `data/theta0.npy` and `data/particle0.npy` exist
   - If files are in `Neural_Galerkin_Sampling/Initial_Fit/` or `Initial_sampling/`, copy them to `data/`:
     ```bash
     mkdir -p data
     cp Neural_Galerkin_Sampling/Initial_Fit/theta0*.npy data/
     cp Neural_Galerkin_Sampling/Initial_sampling/particle0*.npy data/
     ```

3. **NaN/Inf errors**:
   - Reduce time step `dt` in `config.py`
   - Increase regularization `ridge_lambda`
   - Check initial condition fitting quality

4. **Slow training**:
   - Reduce `SVGD_PARAMS['steps']`
   - Reduce network size (`NETWORK_PARAMS['m']`)
   - Use smaller time step for stability

### Debug Mode

Enable debug output by modifying `runner.py`:
- Debug information is printed every 50 steps
- Check particle statistics, parameter norms, and residual values

---

## 📚 References

- Neural Galerkin Method
- Stein Variational Gradient Descent (SVGD)
- Physics-Informed Neural Networks (PINNs)





**Last Updated**: 2025

