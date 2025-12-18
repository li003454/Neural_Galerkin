# step1_fit_initial_condition_AC.py

import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from time import time
import os

# Import our custom modules
from nn import DeepNetAC
from config import (
    AC_PROBLEM_DATA, AC_NETWORK_PARAMS, AC_TRAINING_DATA, AC_OUTPUT_PATHS
)

def loss_fn_wrapper(model, xs, us):
    """Creates a JIT-compiled loss function."""
    # Ensure us is 1D before JIT compilation
    us_1d = us.squeeze() if us.ndim > 1 else us
    
    @jax.jit
    def loss_fn(params):
        # Direct batch computation - more efficient
        pred = model.apply({'params': params}, xs)  # xs shape: (batch, 1), pred shape: (batch,)
        # Compute MSE
        return jnp.mean((us_1d - pred) ** 2) / 2.0
    return loss_fn

def fit_initial_condition_AC():
    """
    Trains a neural network to fit the initial condition u(x, 0) of the Allen-Cahn equation.
    Saves the initial parameters (theta0) to a file.
    """
    # --- Setup ---
    key = jax.random.key(AC_TRAINING_DATA['seed'])
    key, model_key, data_key = jax.random.split(key, 3)

    # Instantiate the network model
    net = DeepNetAC(
        m=AC_NETWORK_PARAMS['m'], 
        l=AC_NETWORK_PARAMS['l'], 
        L=AC_NETWORK_PARAMS['L']
    )

    # Generate training data
    x_train = jax.random.uniform(
        data_key,
        (AC_TRAINING_DATA['batch_size'], AC_PROBLEM_DATA['d']),
        minval=AC_PROBLEM_DATA['domain'][0],
        maxval=AC_PROBLEM_DATA['domain'][1]
    )
    u_train = AC_PROBLEM_DATA['initial_fn'](x_train)

    # Initialize model parameters
    params = net.init(model_key, x_train)['params']

    # --- Training Setup ---
    loss_fn = loss_fn_wrapper(net, x_train, u_train)
    value_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    
    optimizer = optax.adam(learning_rate=AC_TRAINING_DATA['gamma'])
    opt_state = optimizer.init(params)
    
    losses = []
    print('🚀 Fitting the initial condition for Allen-Cahn equation...')
    print(f'   Network: m={AC_NETWORK_PARAMS["m"]}, l={AC_NETWORK_PARAMS["l"]}')
    print(f'   Training for {AC_TRAINING_DATA["epochs"]} epochs...')
    timer = time()

    # --- Training Loop ---
    for epoch in range(AC_TRAINING_DATA['epochs']):
        loss, grads = value_and_grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        losses.append(loss)

        if epoch % 1000 == 0 or epoch == AC_TRAINING_DATA['epochs'] - 1:
            err = jnp.linalg.norm(u_train - net.apply({'params': params}, x_train)) / jnp.linalg.norm(u_train)
            print(f'Epoch {epoch:5d}/{AC_TRAINING_DATA["epochs"]} | Loss: {loss:.4e} | Relative L2 Error: {err:.4e}')

    print(f'✅ Fitting complete. Elapsed time: {time() - timer:.2f}s')

    # --- Save and Visualize Results ---
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(AC_OUTPUT_PATHS['initial_theta']), exist_ok=True)
    
    # Flatten and save the parameters
    theta_flat, _ = jax.flatten_util.ravel_pytree(params)
    jnp.save(AC_OUTPUT_PATHS['initial_theta'], theta_flat)
    print(f"💾 Initial parameters saved to '{AC_OUTPUT_PATHS['initial_theta']}'")

    # Plot loss history
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Log Loss')
    plt.title('Initial Fit - Loss History (Allen-Cahn)')
    plt.grid(True)
    plt.savefig('data/ac_training_loss.png', dpi=150, bbox_inches='tight')
    print("💾 Loss plot saved to 'data/ac_training_loss.png'")
    plt.close()  # Close instead of show to avoid blocking

    # Plot final fit vs true solution
    x_plot = jnp.linspace(
        AC_PROBLEM_DATA['domain'][0], 
        AC_PROBLEM_DATA['domain'][1], 
        AC_PROBLEM_DATA['N']
    )
    u_pred = net.apply({'params': params}, x_plot.reshape(-1, 1))
    u_true_plot = AC_PROBLEM_DATA['initial_fn'](x_plot)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, u_true_plot, 'k--', label='True Initial Condition')
    plt.plot(x_plot, u_pred, 'r-', label='Fitted Network')
    plt.title('Initial Fit - True vs. Fitted Solution (Allen-Cahn)')
    plt.xlabel('x')
    plt.ylabel('u(x, 0)')
    plt.legend()
    plt.grid(True)
    plt.savefig('data/ac_initial_fit.png', dpi=150, bbox_inches='tight')
    print("💾 Fit plot saved to 'data/ac_initial_fit.png'")
    plt.close()  # Close instead of show to avoid blocking

if __name__ == "__main__":
    fit_initial_condition_AC()

