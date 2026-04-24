"""Metal-compatible 1D convolution (no lax.conv_general_dilated)."""

import jax
import jax.numpy as jnp
from jax import lax
from typing import Optional


def causal_conv1d_metal(
    x: jnp.ndarray,  # [B, D, T] - input
    weight: jnp.ndarray,  # [D, K] - kernel weights
    bias: Optional[jnp.ndarray] = None,  # [D]
    activation: str = "silu",
) -> jnp.ndarray:
    """Metal-compatible causal 1D convolution using einsum.
    
    Replaces lax.conv_general_dilated which Metal doesn't support for 1D.
    
    Args:
        x: Input tensor [batch, dim, seq_len]
        weight: Convolution weights [dim, kernel_size]
        bias: Optional bias [dim]
        activation: Activation function
        
    Returns:
        Output tensor [batch, dim, seq_len]
    """
    batch, dim, seq_len = x.shape
    kernel_size = weight.shape[-1]
    
    # Pad input on the left for causal convolution
    # We need (kernel_size - 1) padding on the left
    padded = jnp.pad(x, ((0, 0), (0, 0), (kernel_size - 1, 0)))
    
    # Compute output using einsum over sliding windows
    # For each output position t, we need:
    # out[b, d, t] = sum_k(padded[b, d, t+k] * weight[d, k])
    
    # Create output by iterating over kernel positions
    out = jnp.zeros((batch, dim, seq_len), dtype=x.dtype)
    for k in range(kernel_size):
        # Extract the slice for this kernel position
        # padded[b, d, t+k] for t in [0, seq_len)
        slice_k = padded[:, :, k:k+seq_len]
        # Add contribution from this kernel position
        out = out + slice_k * weight[:, k:k+1]
    
    # Add bias
    if bias is not None:
        out = out + bias[:, None]
    
    # Activation
    if activation == "silu":
        out = jax.nn.silu(out)
    elif activation == "relu":
        out = jax.nn.relu(out)
    
    return out


def causal_conv1d_scan(
    x: jnp.ndarray,  # [B, D, T]
    weight: jnp.ndarray,  # [D, K]
    bias: Optional[jnp.ndarray] = None,
    activation: str = "silu",
) -> jnp.ndarray:
    """Causal 1D convolution using lax.scan for efficiency.
    
    This is more efficient than the simple loop version.
    
    Args:
        x: Input tensor [batch, dim, seq_len]
        weight: Convolution weights [dim, kernel_size]
        bias: Optional bias [dim]
        activation: Activation function
        
    Returns:
        Output tensor [batch, dim, seq_len]
    """
    batch, dim, seq_len = x.shape
    kernel_size = weight.shape[-1]
    
    # Pad input on the left for causal convolution
    padded = jnp.pad(x, ((0, 0), (0, 0), (kernel_size - 1, 0)))
    
    # Process each position using scan
    def conv_step(carry, t):
        # Extract window of size kernel_size ending at position t
        # padded[:, :, t:t+kernel_size] has shape [B, D, K]
        window = padded[:, :, t:t+kernel_size]
        # Dot product with weights: [B, D, K] x [D, K] -> [B, D]
        out_t = jnp.einsum('bdk,dk->bd', window, weight)
        if bias is not None:
            out_t = out_t + bias
        if activation == "silu":
            out_t = jax.nn.silu(out_t)
        elif activation == "relu":
            out_t = jax.nn.relu(out_t)
        return carry, out_t
    
    _, outputs = lax.scan(conv_step, None, jnp.arange(seq_len))
    # outputs is [T, B, D], transpose to [B, D, T]
    return outputs.transpose(1, 2, 0)


# Choose which implementation to use
causal_conv1d = causal_conv1d_metal  # Use simple loop version for Metal compatibility

