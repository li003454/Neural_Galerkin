# neural_network.py

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable

# --- KdV Equation Network Architectures ---

class PeriodicPhiKdV(nn.Module):
    """Custom periodic feature layer for KdV."""
    m: int
    L: float
    param_init: Callable = nn.initializers.uniform()

    @nn.compact
    def __call__(self, x):
        w = self.param('kernel', self.param_init, (self.m, ))
        b = self.param('bias', self.param_init, (self.m, x.shape[-1]))

        def apply_phi(single_x):
            # Corrected broadcasting for x and b
            diff = jnp.sin(jnp.pi * (jnp.expand_dims(single_x, 0) - b) / self.L)
            norm_sq = jnp.sum(diff ** 2, axis=1)
            return jnp.exp(-w ** 2 * norm_sq)

        # Use vmap over the batch dimension of x
        return jax.vmap(apply_phi)(x)

class ShallowNetKdV(nn.Module):
    """Shallow network model for KdV using the periodic feature layer."""
    m: int
    L: float

    @nn.compact
    def __call__(self, x):
        # Ensure x has a batch dimension, even if it's 1
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        # Define the network sequence
        phi_layer = PeriodicPhiKdV(m=self.m, L=self.L)
        dense_layer = nn.Dense(features=1, use_bias=False)
        
        features = phi_layer(x)
        output = dense_layer(features)
        
        # Ensure proper squeezing: if single input, return scalar; otherwise keep batch dim
        output = jnp.squeeze(output, axis=-1)  # Remove the last dimension (features=1)
        return output

# # --- Allen-Cahn (AC) Equation Network Architectures ---
# # (我们暂时保留它们，以备将来扩展)

# class PeriodicPhiAC(nn.Module):
#     # ... (你的 PeriodicPhiAC 代码) ...

# class DeepNetAC(nn.Module):
#     # ... (你的 DeepNetAC 代码) ...