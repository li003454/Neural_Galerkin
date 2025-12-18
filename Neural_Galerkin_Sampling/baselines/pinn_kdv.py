# pinn_kdv.py
# Physics-Informed Neural Network (PINN) for KdV Equation
# Baseline comparison with Neural Galerkin + SVGD method

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from flax import linen as nn
from flax.training import train_state
import optax
from functools import partial
import os
from tqdm import trange

# Import exact solution from existing code
from exact_solutions import kdv_two_soliton
from config import PROBLEM_DATA, EVOLUTION_PARAMS

# ============================================================
# 1. PINN Network: u_phi(t, x)
# ============================================================

class PINN_KdV(nn.Module):
    """Space-time PINN for KdV equation: u(t, x)"""
    hidden_dim: int = 64
    num_layers: int = 4

    @nn.compact
    def __call__(self, tx: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            tx: (..., 2) where tx[...,0]=t, tx[...,1]=x
        Returns:
            (...,) scalar u(t,x)
        """
        x = tx
        for _ in range(self.num_layers):
            x = nn.tanh(nn.Dense(self.hidden_dim)(x))
        x = nn.Dense(1)(x)
        return x[..., 0]  # squeeze last dim


# ============================================================
# 2. PDE Residual Functions
# ============================================================

def make_u_fns(model: PINN_KdV, params):
    """Create functions for u, u_t, u_x, u_xxx using automatic differentiation."""
    
    def u_scalar(tx):
        """tx: (2,) vector [t, x], returns scalar u(t,x)"""
        return model.apply({'params': params}, tx[None, :])[0]
    
    # Gradient w.r.t (t,x)
    def grad_tx(tx):
        return jax.grad(u_scalar)(tx)  # shape (2,)
    
    def u_t(tx):
        g = grad_tx(tx)
        return g[0]
    
    def u_x(tx):
        g = grad_tx(tx)
        return g[1]
    
    # Second derivative w.r.t x
    def u_xx(tx):
        def f_x(x_scalar):
            return u_scalar(jnp.array([tx[0], x_scalar]))
        d1 = jax.grad(f_x)(tx[1])
        d2 = jax.grad(lambda z: jax.grad(f_x)(z))(tx[1])
        return d2
    
    # Third derivative w.r.t x
    def u_xxx(tx):
        def f_x(x_scalar):
            return u_scalar(jnp.array([tx[0], x_scalar]))
        # Compute third derivative by nesting grad
        d1_fn = jax.grad(f_x)
        d2_fn = jax.grad(d1_fn)
        d3_fn = jax.grad(d2_fn)
        return d3_fn(tx[1])
    
    return u_scalar, u_t, u_x, u_xxx


def kdv_residual(model: PINN_KdV, params, tx_batch: jnp.ndarray) -> jnp.ndarray:
    """
    Compute KdV PDE residual: r = u_t + 6*u*u_x + u_xxx
    
    Args:
        model: PINN_KdV model
        params: model parameters
        tx_batch: (N_f, 2), each row [t, x]
    Returns:
        residuals r(t,x) of shape (N_f,)
    """
    u, u_t, u_x, u_xxx = make_u_fns(model, params)
    
    def r_single(tx):
        val_u = u(tx)
        val_ut = u_t(tx)
        val_ux = u_x(tx)
        val_uxxx = u_xxx(tx)
        return val_ut + 6.0 * val_u * val_ux + val_uxxx
    
    r_vmap = jax.vmap(r_single)
    return r_vmap(tx_batch)


# ============================================================
# 3. Initial Condition
# ============================================================

def kdv_initial_condition(x: jnp.ndarray) -> jnp.ndarray:
    """
    Initial condition: u(0, x) = kdv_two_soliton(x, t=0)
    
    Args:
        x: (N_ic,) or (N_ic, 1)
    Returns:
        u0: (N_ic,)
    """
    x = x.reshape(-1)
    return kdv_two_soliton(x, t=0.0)


# ============================================================
# 4. Sampling Functions
# ============================================================

def sample_pinn_batch(rng_key, N_ic: int, N_f: int, 
                      x_min: float, x_max: float, T: float):
    """
    Sample collocation points for PINN training.
    
    Returns:
        tx_ic: (N_ic, 2)  t=0, x ~ Uniform[x_min, x_max]
        u0_ic: (N_ic,)    initial values u(0,x)
        tx_f:  (N_f, 2)   (t,x) ~ Uniform([0,T] x [x_min,x_max])
    """
    key_ic, key_f_t, key_f_x = jax.random.split(rng_key, 3)
    
    # IC points: t=0, x uniformly distributed
    x_ic = jax.random.uniform(key_ic, (N_ic, 1), minval=x_min, maxval=x_max)
    t_ic = jnp.zeros_like(x_ic)
    tx_ic = jnp.concatenate([t_ic, x_ic], axis=1)
    u0_ic = kdv_initial_condition(x_ic[:, 0])
    
    # PDE collocation points: (t, x) uniformly distributed
    t_f = jax.random.uniform(key_f_t, (N_f, 1), minval=0.0, maxval=T)
    x_f = jax.random.uniform(key_f_x, (N_f, 1), minval=x_min, maxval=x_max)
    tx_f = jnp.concatenate([t_f, x_f], axis=1)
    
    return tx_ic, u0_ic, tx_f


# ============================================================
# 5. Loss Function & Training Step
# ============================================================

class TrainState(train_state.TrainState):
    pass


def pinn_loss(model: PINN_KdV, params, tx_ic, u0_ic, tx_f,
              lambda_ic=1.0, lambda_pde=1.0):
    """
    PINN loss: L = lambda_ic * L_ic + lambda_pde * L_pde
    
    Args:
        model: PINN_KdV model
        params: model parameters
        tx_ic: (N_ic, 2) initial condition points
        u0_ic: (N_ic,) initial condition values
        tx_f: (N_f, 2) PDE collocation points
        lambda_ic: weight for IC loss
        lambda_pde: weight for PDE loss
    Returns:
        total_loss, (ic_loss, pde_loss)
    """
    # IC loss: ||u(0, x) - u0(x)||^2
    u_ic_pred = model.apply({'params': params}, tx_ic)
    ic_loss = jnp.mean((u_ic_pred - u0_ic)**2)
    
    # PDE residual loss: ||r(t, x)||^2
    r_f = kdv_residual(model, params, tx_f)
    pde_loss = jnp.mean(r_f**2)
    
    total = lambda_ic * ic_loss + lambda_pde * pde_loss
    return total, (ic_loss, pde_loss)


@partial(jax.jit, static_argnames=('model', 'N_ic', 'N_f', 'x_min', 'x_max', 'T', 'lambda_ic', 'lambda_pde'))
def train_step(state: TrainState, model: PINN_KdV, rng_key,
               N_ic, N_f, x_min, x_max, T,
               lambda_ic=1.0, lambda_pde=1.0):
    """Single training step."""
    tx_ic, u0_ic, tx_f = sample_pinn_batch(rng_key, N_ic, N_f, x_min, x_max, T)
    
    def loss_fn(params):
        loss_value, (ic_loss, pde_loss) = pinn_loss(
            model, params, tx_ic, u0_ic, tx_f,
            lambda_ic=lambda_ic, lambda_pde=lambda_pde
        )
        return loss_value, (ic_loss, pde_loss)
    
    grads, (ic_loss, pde_loss) = jax.grad(loss_fn, has_aux=True)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, ic_loss, pde_loss


# ============================================================
# 6. Evaluation & Error Computation
# ============================================================

def evaluate_pinn_error(model: PINN_KdV, params, x_eval, t_eval):
    """
    Evaluate PINN solution and compute L2 error against exact solution.
    
    Args:
        model: PINN_KdV model
        params: trained parameters
        x_eval: (N_x,) spatial grid points
        t_eval: (N_t,) time points
    Returns:
        u_pred: (N_x, N_t) predicted solution
        u_true: (N_x, N_t) exact solution
        l2_errors: (N_t,) L2 error at each time
        relative_errors: (N_t,) relative L2 error at each time
    """
    N_x, N_t = len(x_eval), len(t_eval)
    u_pred = np.zeros((N_x, N_t))
    u_true = np.zeros((N_x, N_t))
    
    # Evaluate at each time point
    for j, t in enumerate(t_eval):
        # Create (t, x) pairs
        tx_grid = jnp.stack([
            jnp.full(N_x, t),
            jnp.array(x_eval)
        ], axis=1)
        
        # Predict
        u_pred[:, j] = np.array(model.apply({'params': params}, tx_grid))
        
        # Exact solution
        u_true[:, j] = np.array(kdv_two_soliton(x_eval, t))
    
    # Compute L2 errors at each time
    l2_errors = np.zeros(N_t)
    relative_errors = np.zeros(N_t)
    
    for j in range(N_t):
        error = u_pred[:, j] - u_true[:, j]
        l2_errors[j] = np.sqrt(np.mean(error**2))
        
        # Relative error: ||u_pred - u_true||_2 / ||u_true||_2
        norm_true = np.sqrt(np.mean(u_true[:, j]**2))
        relative_errors[j] = l2_errors[j] / (norm_true + 1e-10)
    
    return u_pred, u_true, l2_errors, relative_errors


# ============================================================
# 7. Visualization
# ============================================================

def plot_pinn_results(model: PINN_KdV, params, x_min, x_max, T,
                      save_dir='pinn_output'):
    """Generate comprehensive visualization plots."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create evaluation grid
    x_eval = np.linspace(x_min, x_max, 512)
    t_eval = np.linspace(0.0, T, 100)
    
    print("📊 Evaluating PINN solution...")
    u_pred, u_true, l2_errors, relative_errors = evaluate_pinn_error(
        model, params, x_eval, t_eval
    )
    
    # Print error statistics
    print(f"\n📈 Error Statistics:")
    print(f"   Final L2 Error: {l2_errors[-1]:.6e}")
    print(f"   Final Relative Error: {relative_errors[-1]:.6e}")
    print(f"   Mean L2 Error: {np.mean(l2_errors):.6e}")
    print(f"   Max L2 Error: {np.max(l2_errors):.6e}")
    
    # Plot 1: Solution snapshots at different times
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plot_times = [0.0, T/2, T]
    
    for ax, t_plot in zip(axes, plot_times):
        idx = np.argmin(np.abs(t_eval - t_plot))
        t_actual = t_eval[idx]
        
        ax.plot(x_eval, u_true[:, idx], 'k--', linewidth=2, label='Exact')
        ax.plot(x_eval, u_pred[:, idx], 'darkviolet', linewidth=2, label='PINN')
        ax.set_title(f't = {t_actual:.2f}')
        ax.set_xlabel('x')
        ax.set_ylabel('u(x, t)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pinn_snapshots.png'), dpi=300)
    print(f"💾 Snapshots saved to '{save_dir}/pinn_snapshots.png'")
    plt.close()
    
    # Plot 2: Spacetime solution (heatmap)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    X, T_mesh = np.meshgrid(x_eval, t_eval)
    
    # Predicted solution
    im1 = ax1.contourf(X, T_mesh, u_pred.T, levels=50, cmap='plasma')
    ax1.set_xlabel('x')
    ax1.set_ylabel('t')
    ax1.set_title('PINN Solution')
    plt.colorbar(im1, ax=ax1, label='u(x, t)')
    
    # Exact solution
    im2 = ax2.contourf(X, T_mesh, u_true.T, levels=50, cmap='plasma')
    ax2.set_xlabel('x')
    ax2.set_ylabel('t')
    ax2.set_title('Exact Solution')
    plt.colorbar(im2, ax=ax2, label='u(x, t)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pinn_spacetime.png'), dpi=300)
    print(f"💾 Spacetime plot saved to '{save_dir}/pinn_spacetime.png'")
    plt.close()
    
    # Plot 3: Error evolution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # L2 error over time
    ax1.semilogy(t_eval, l2_errors, 'b-', linewidth=2, marker='o', markersize=3)
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('L2 Error (log scale)')
    ax1.set_title('L2 Error Evolution')
    ax1.grid(True, alpha=0.3)
    
    # Relative error over time
    ax2.semilogy(t_eval, relative_errors, 'r-', linewidth=2, marker='s', markersize=3)
    ax2.set_xlabel('Time t')
    ax2.set_ylabel('Relative L2 Error (log scale)')
    ax2.set_title('Relative Error Evolution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pinn_errors.png'), dpi=300)
    print(f"💾 Error plots saved to '{save_dir}/pinn_errors.png'")
    plt.close()
    
    # Plot 4: Error distribution at final time
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    error_final = u_pred[:, -1] - u_true[:, -1]
    ax.plot(x_eval, error_final, 'r-', linewidth=2)
    ax.set_xlabel('x')
    ax.set_ylabel('Error = u_pred - u_true')
    ax.set_title(f'Error Distribution at Final Time (t = {T:.2f})')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pinn_error_distribution.png'), dpi=300)
    print(f"💾 Error distribution saved to '{save_dir}/pinn_error_distribution.png'")
    plt.close()
    
    # Save error data
    error_data = {
        'times': t_eval.tolist(),
        'l2_errors': l2_errors.tolist(),
        'relative_errors': relative_errors.tolist(),
        'final_l2_error': float(l2_errors[-1]),
        'final_relative_error': float(relative_errors[-1]),
        'mean_l2_error': float(np.mean(l2_errors)),
        'max_l2_error': float(np.max(l2_errors))
    }
    
    import json
    with open(os.path.join(save_dir, 'pinn_error_data.json'), 'w') as f:
        json.dump(error_data, f, indent=2)
    
    print(f"💾 Error data saved to '{save_dir}/pinn_error_data.json'")
    
    return error_data


# ============================================================
# 8. Main Training Function
# ============================================================

def main():
    """Main training function."""
    print("=" * 60)
    print("🚀 PINN Training for KdV Equation")
    print("=" * 60)
    
    # Configuration (aligned with Neural Galerkin setup)
    x_min, x_max = PROBLEM_DATA['domain']
    T = EVOLUTION_PARAMS['t_final']
    
    # PINN hyperparameters
    N_ic = 256          # Initial condition points
    N_f = 4096          # PDE collocation points per step
    num_steps = 20000   # Training steps
    hidden_dim = 64     # Network width
    num_layers = 4      # Network depth
    lr = 1e-3           # Learning rate
    lambda_ic = 1.0     # IC loss weight
    lambda_pde = 1.0    # PDE loss weight
    
    print(f"\n📋 Configuration:")
    print(f"   Domain: [{x_min}, {x_max}]")
    print(f"   Time: [0, {T}]")
    print(f"   IC points: {N_ic}")
    print(f"   PDE collocation points: {N_f}")
    print(f"   Training steps: {num_steps}")
    print(f"   Network: {num_layers} layers, {hidden_dim} hidden units")
    print(f"   Learning rate: {lr}")
    
    # Initialize random key
    rng = jax.random.PRNGKey(42)
    
    # Create model
    model = PINN_KdV(hidden_dim=hidden_dim, num_layers=num_layers)
    
    # Initialize parameters
    rng, init_key = jax.random.split(rng)
    tx_init = jax.random.uniform(init_key, (1, 2),
                                minval=jnp.array([0.0, x_min]),
                                maxval=jnp.array([T, x_max]))
    params = model.init(init_key, tx_init)['params']
    
    # Create optimizer
    optimizer = optax.adam(lr)
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )
    
    # Training loop
    print(f"\n🔄 Starting training...")
    loss_history = {'ic': [], 'pde': [], 'total': []}
    
    for step in trange(1, num_steps + 1, desc="Training"):
        rng, subkey = jax.random.split(rng)
        state, ic_loss, pde_loss = train_step(
            state, model, subkey,
            N_ic=N_ic, N_f=N_f,
            x_min=x_min, x_max=x_max, T=T,
            lambda_ic=lambda_ic, lambda_pde=lambda_pde
        )
        
        total_loss = lambda_ic * ic_loss + lambda_pde * pde_loss
        loss_history['ic'].append(float(ic_loss))
        loss_history['pde'].append(float(pde_loss))
        loss_history['total'].append(float(total_loss))
        
        if step % 500 == 0:
            print(f"\n[Step {step}] "
                  f"IC loss: {ic_loss:.3e}, "
                  f"PDE loss: {pde_loss:.3e}, "
                  f"Total: {total_loss:.3e}")
    
    print(f"\n✅ Training complete!")
    
    # Plot loss history
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    steps = np.arange(1, num_steps + 1)
    ax.semilogy(steps, loss_history['ic'], 'b-', label='IC Loss', alpha=0.7)
    ax.semilogy(steps, loss_history['pde'], 'r-', label='PDE Loss', alpha=0.7)
    ax.semilogy(steps, loss_history['total'], 'g-', label='Total Loss', alpha=0.7)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('PINN Training Loss History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs('pinn_output', exist_ok=True)
    plt.savefig('pinn_output/pinn_loss_history.png', dpi=300)
    print(f"💾 Loss history saved to 'pinn_output/pinn_loss_history.png'")
    plt.close()
    
    # Evaluate and plot results
    print(f"\n📊 Evaluating solution...")
    error_data = plot_pinn_results(model, state.params, x_min, x_max, T)
    
    print(f"\n" + "=" * 60)
    print(f"✅ PINN Training and Evaluation Complete!")
    print(f"=" * 60)
    
    return state, model, error_data


if __name__ == "__main__":
    state, model, error_data = main()

