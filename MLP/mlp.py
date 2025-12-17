import jax
import jax.numpy as jnp
from typing import List, Dict


class MLP:

    def __init__(self, layer_dims: List[int], activation: str = 'tanh'):
        """
        Initialize L-layer MLP

        Args:
            layer_dims: [n0, n1, n2, ..., n_{L-1}, n_L] dimensions of each layer
            activation: 'tanh' or 'sigmoid'
        """
        self.layer_dims = layer_dims
        self.L = len(layer_dims) - 1  # Number of layers
        self.activation_type = activation

        # Initialize parameters v = (W1, b1, ..., WL, bL)
        self.v = self._initialize_parameters()

    def _initialize_parameters(self) -> Dict:
        """Initialize parameters"""
        key = jax.random.PRNGKey(42)
        params = {}

        for l in range(1, self.L + 1):
            # For layer l: Wl ∈ R^{nl × n_{l-1}}, bl in R^{nl}
            n_out, n_in = self.layer_dims[l], self.layer_dims[l - 1]

            # Xavier initialization
            scale = jnp.sqrt(2.0 / (n_in + n_out))

            key, subkey = jax.random.split(key)
            params[f'W{l}'] = jax.random.normal(subkey, (n_out, n_in), dtype=jnp.float64) * scale

            key, subkey = jax.random.split(key)
            params[f'b{l}'] = jax.random.normal(subkey, (n_out,), dtype=jnp.float64) * scale*0.1

        return params

    def activation(self, u) -> jnp.ndarray:
        """Activation function sigma(u)"""
        if self.activation_type == 'tanh':
            return jnp.tanh(u)
        elif self.activation_type == 'sigmoid':
            return 1.0 / (1.0 + jnp.exp(-u))
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_type}")

    def forward(self, params: Dict, x) -> jnp.ndarray:
        """
        Forward MLP(x; params)
        """
        # Ensure 2D shape for batch processing
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Hidden layers: l = 1 to L-1
        for l in range(1, self.L):
            W = params[f'W{l}']  # shape: (n_l, n_{l-1})
            b = params[f'b{l}']  # shape: (n_l,)
            x = self.activation(jnp.dot(x, W.T) + b)

        # Output layer: l = L
        W_out = params[f'W{self.L}']  # shape: (n_L, n_{L-1})
        b_out = params[f'b{self.L}']  # shape: (n_L,)
        output = jnp.dot(x, W_out.T) + b_out

        # Always return 1D array, squeeze singleton dimensions
        return output.reshape(-1)

    def get_parameters(self) -> Dict:
        """Get current parameters"""
        return self.v.copy()

    def set_parameters(self, new_params: Dict):
        """Set new parameters"""
        self.v = new_params.copy()