# step1_fit_initial_condition.py

import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from time import time
import os

# Import our custom modules
from config import PROBLEM_DATA, NETWORK_PARAMS, TRAINING_DATA, OUTPUT_PATHS
from nn import ShallowNetKdV
from exact_solutions import kdv_two_soliton

def loss_fn_wrapper(model, xs, us):
    """Creates a JIT-compiled loss function."""
    @jax.jit
    def loss_fn(params):
        def mse(x, u):
            pred = model.apply({'params': params}, x.reshape(1, -1))
            return jnp.inner(u - pred, u - pred) / 2.0
        return jnp.mean(jax.vmap(mse)(xs, us), axis=0)
    return loss_fn

def fit_initial_condition():
    """
    Trains a neural network to fit the initial condition u(x, 0) of the PDE.
    Saves the initial parameters (theta0) to a file.
    """
    # --- Setup ---
    key = jax.random.key(TRAINING_DATA['seed'])
    key, model_key, data_key = jax.random.split(key, 3)

    # Instantiate the network model
    net = ShallowNetKdV(m=NETWORK_PARAMS['m'], L=NETWORK_PARAMS['L'])

    # Generate training data
    x_train = jax.random.uniform(
        data_key,
        (TRAINING_DATA['batch_size'], PROBLEM_DATA['d']),
        minval=PROBLEM_DATA['domain'][0],
        maxval=PROBLEM_DATA['domain'][1]
    )
    u_train = PROBLEM_DATA['initial_fn'](x_train)

    # Initialize model parameters
    params = net.init(model_key, x_train)['params']

    # --- Training Setup ---
    loss_fn = loss_fn_wrapper(net, x_train, u_train)
    value_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    
    optimizer = optax.adam(learning_rate=TRAINING_DATA['gamma'])
    opt_state = optimizer.init(params)
    
    losses = []
    print('🚀 Fitting the initial condition...')
    timer = time()

    # --- Training Loop ---
    for epoch in range(TRAINING_DATA['epochs']):
        loss, grads = value_and_grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        losses.append(loss)

        if epoch % 1000 == 0 or epoch == TRAINING_DATA['epochs'] - 1:
            err = jnp.linalg.norm(u_train - net.apply({'params': params}, x_train)) / jnp.linalg.norm(u_train)
            print(f'Epoch {epoch:5d}/{TRAINING_DATA["epochs"]} | Loss: {loss:.4e} | Relative L2 Error: {err:.4e}')

    print(f'✅ Fitting complete. Elapsed time: {time() - timer:.2f}s')

    # --- Save and Visualize Results ---
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_PATHS['initial_theta']), exist_ok=True)
    
    # Flatten and save the parameters
    theta_flat, _ = jax.flatten_util.ravel_pytree(params)
    jnp.save(OUTPUT_PATHS['initial_theta'], theta_flat)
    print(f"💾 Initial parameters saved to '{OUTPUT_PATHS['initial_theta']}'")

    # Plot loss history
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Log Loss')
    plt.title('Initial Fit - Loss History')
    plt.grid(True)
    plt.show()

    # Plot final fit vs true solution
    x_plot = jnp.linspace(PROBLEM_DATA['domain'][0], PROBLEM_DATA['domain'][1], PROBLEM_DATA['N'])
    u_pred = net.apply({'params': params}, x_plot)
    u_true_plot = PROBLEM_DATA['initial_fn'](x_plot)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, u_true_plot, 'k--', label='True Initial Condition')
    plt.plot(x_plot, u_pred, 'r-', label='Fitted Network')
    plt.title('Initial Fit - True vs. Fitted Solution')
    plt.xlabel('x')
    plt.ylabel('u(x, 0)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    fit_initial_condition()