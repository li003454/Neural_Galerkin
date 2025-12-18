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

# --- Allen-Cahn (AC) Equation Network Architectures ---

class PeriodicPhiAC(nn.Module):
    """Custom periodic feature layer for Allen-Cahn equation."""
    m: int
    L: float
    param_init: Callable = nn.initializers.truncated_normal(stddev=1.0)
    # param_init: Callable = nn.initializers.uniform()
    # param_init: Callable = nn.initializers.constant(1)

    @nn.compact
    def __call__(self, x):
        d = x.shape[-1]  # input dimension
        
        # w = self.param('kernel', self.param_init, (self.m, d)) # w.shape = (m, d)
        # b = self.param('bias', self.param_init, (d, )) # b.shape = (d, )
        a = self.param('a', self.param_init, (self.m, d))
        b = self.param('b', self.param_init, (self.m, d))
        c = self.param('c', self.param_init, (self.m, d))

        def apply_phi(single_x):
            # return w @ jnp.sin(2 * jnp.pi * (x - b) / self.L)
            return jnp.sum(a * jnp.cos((2 * jnp.pi / self.L) * single_x + b) + c, axis=1)

        # Apply phi to each input
        phi = jax.vmap(apply_phi)(x)
        return phi

class DeepNetAC(nn.Module):
    """Deep network model for Allen-Cahn equation."""
    m: int
    l: int
    L: float

    @nn.compact
    def __call__(self, x):
        # Ensure x has a batch dimension, even if it's 1
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        layers = [PeriodicPhiAC(self.m, self.L), nn.activation.tanh]
        
        for _ in range(self.l - 1):
            layers.append(nn.Dense(self.m))
            layers.append(nn.activation.tanh)
        
        layers.append(nn.Dense(features=1, use_bias=False))
        
        net = nn.Sequential(layers)
        
        output = net(x)
        # Ensure proper squeezing: if single input, return scalar; otherwise keep batch dim
        output = jnp.squeeze(output, axis=-1)  # Remove the last dimension (features=1)
        return output