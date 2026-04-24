"""Layer implementations for Qwen 3.5."""

import jax
import jax.numpy as jnp
import numpy as np
from jax import nn
from typing import Tuple, Optional
from jax import lax


def rms_norm(x: jnp.ndarray, weight: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """RMSNorm implementation for Qwen 3.5.
    
    Args:
        x: Input tensor
        weight: Weight tensor
        eps: Epsilon for numerical stability
    
    Returns:
        Normalized output
    """
    mean_sq = jnp.mean(x ** 2, axis=-1, keepdims=True)
    x_norm = x / jnp.sqrt(mean_sq + eps)
    return x_norm * (1.0 + weight)


def l2norm(x: jnp.ndarray, axis: int = -1, eps: float = 1e-6) -> jnp.ndarray:
    """L2 normalization.
    
    Args:
        x: Input tensor
        axis: Axis to normalize over
        eps: Epsilon for numerical stability
    
    Returns:
        Normalized tensor
    """
    norm = jnp.sqrt(jnp.sum(x ** 2, axis=axis, keepdims=True) + eps)
    return x / norm


def apply_rope(
    x: jnp.ndarray,
    positions: jnp.ndarray,
    head_dim: int,
    rope_theta: float,
    partial_rotary_factor: float = 1.0,
    layout: str = "BTHD",
    mrope_section: Optional[list] = None,
) -> jnp.ndarray:
    """Apply rotary position embeddings with optional mrope support.
    
    Supports two input layouts:
    - "BTHD": [batch, seq_len, num_heads, head_dim] (for linear_attention)
    - "BHTD": [batch, num_heads, seq_len, head_dim] (for full_attention)
    
    For mrope (multimodal RoPE), positions should have shape (3, batch, seq_len)
    for T (text), H (height), W (width) dimensions.
    
    Args:
        x: Input tensor
        positions: Position IDs [batch, seq_len] or (3, batch, seq_len) for mrope
        head_dim: Dimension of each head
        rope_theta: RoPE base frequency
        partial_rotary_factor: Fraction of head_dim to apply RoPE to
        layout: Input layout, either "BTHD" or "BHTD"
        mrope_section: Optional list [T_section, H_section, W_section] for mrope interleaving
    
    Returns:
        RoPE-rotated tensor with same shape as input
    """
    dim = int(head_dim * partial_rotary_factor)
    
    # Preserve input dtype for final output
    input_dtype = x.dtype
    
    if mrope_section is not None and len(mrope_section) == 3:
        # Interleaved mrope - matches HF Qwen3_5RMSNorm.apply_interleaved_mrope
        # positions shape: (3, batch, seq_len) or (batch, seq_len) for text-only
        # mrope_section: [T_section, H_section, W_section] e.g., [11, 11, 10]
        
        # If positions is 2D, expand to 3D by repeating
        if positions.ndim == 2:
            positions = jnp.stack([positions, positions, positions], axis=0)
        
        # Compute inv_freq matching HF: arange(0, dim, 2) / dim
        # This gives dim//2 frequencies
        inv_freq = 1.0 / (rope_theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
        
        # Compute freqs for each of the 3 dimensions
        # positions: (3, batch, seq_len)
        # freqs_*: (batch, seq_len, dim//2)
        freqs_0 = jnp.einsum("bs,d->bsd", positions[0], inv_freq)
        freqs_1 = jnp.einsum("bs,d->bsd", positions[1], inv_freq)
        freqs_2 = jnp.einsum("bs,d->bsd", positions[2], inv_freq)
        
        # Stack to get (3, batch, seq_len, dim//2)
        freqs = jnp.stack([freqs_0, freqs_1, freqs_2], axis=0)
        
        # Apply interleaved mrope
        # Take T (freqs[0]) as base: (batch, seq_len, dim//2)
        freqs_t = freqs[0]
        
        # For H (dim_idx=1): replace indices slice(1, mrope_section[1]*3, 3)
        # For W (dim_idx=2): replace indices slice(2, mrope_section[2]*3, 3)
        for dim_idx in [1, 2]:
            length = mrope_section[dim_idx] * 3
            # Create slice indices - use numpy for concrete filtering
            idx_np = np.arange(dim_idx, length, 3)
            idx_np = idx_np[idx_np < dim // 2]  # Filter with numpy (concrete)
            idx = jnp.array(idx_np)
            
            if len(idx) == 0:
                continue
            
            # Get values from freqs[dim_idx]
            freqs_update = jnp.take(freqs[dim_idx], idx, axis=-1)
            
            # Scatter update
            freqs_t = freqs_t.at[:, :, idx].set(freqs_update)
        
        freqs = freqs_t
        # HF concatenates freqs with itself before taking cos/sin
        # emb = torch.cat((freqs, freqs), dim=-1)
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        sin = jnp.sin(emb)
        cos = jnp.cos(emb)
    else:
        # Standard RoPE (text-only, use first row of positions if 3D)
        if positions.ndim == 3:
            positions = positions[0]  # Take T (text) positions
        
        # IMPORTANT: denominator is dim (full rotary dim), arange goes 0, 2, 4, ...
        inv_freq = 1.0 / (
            rope_theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
        )
        
        freqs = jnp.einsum("bs,d->bsd", positions, inv_freq)
        sin = jnp.sin(freqs)
        cos = jnp.cos(freqs)
    
    # Reshape based on layout
    if layout == "BHTD":
        # [B, H, T, D] layout - need [B, 1, T, D] for broadcasting
        sin = sin[:, None, :, :]
        cos = cos[:, None, :, :]
    else:  # BTHD
        # [B, T, H, D] layout - need [B, T, 1, D] for broadcasting
        sin = sin[:, :, None, :]
        cos = cos[:, :, None, :]
    
    # HF applies RoPE to all 'dim' dimensions using cos/sin of shape dim
    # The cos/sin are computed as cat([freqs, freqs], dim=-1) where freqs has dim//2
    # So cos = [c0, c1, ..., c_{n-1}, c0, c1, ..., c_{n-1}] where n = dim//2
    # And sin = [s0, s1, ..., s_{n-1}, s0, s1, ..., s_{n-1}]
    #
    # HF's rotate_half splits the dim-dimensional tensor in half and rotates:
    # rotate_half(x) = cat(-x[n:], x[:n])
    #
    # Then: output = x * cos + rotate_half(x) * sin
    # For i < n: output[i] = x[i]*cos[i] + (-x[i+n])*sin[i]
    # For i >= n: output[i] = x[i]*cos[i] + x[i-n]*sin[i]
    
    n = dim // 2
    x_rot = x[..., :dim]  # The portion to rotate (first 64 dims)
    x1 = x_rot[..., :n]   # First 32 dims
    x2 = x_rot[..., n:]   # Second 32 dims
    
    # cos and sin have shape [..., dim] = [..., 64]
    # But they are duplicated: cos[..., :n] == cos[..., n:]
    # We can use the full cos/sin directly
    
    # First half: output[i] = x1[i] * cos[i] - x2[i] * sin[i]
    # Second half: output[i+n] = x2[i] * cos[i+n] + x1[i] * sin[i+n]
    # Since cos[i] == cos[i+n] and sin[i] == sin[i+n]:
    # First half: x1 * cos[:n] - x2 * sin[:n]
    # Second half: x2 * cos[:n] + x1 * sin[:n]
    
    # Cast cos/sin to input dtype to preserve precision
    cos_half = cos[..., :n].astype(input_dtype)
    sin_half = sin[..., :n].astype(input_dtype)
    
    rotated_first_half = x1 * cos_half - x2 * sin_half
    rotated_second_half = x2 * cos_half + x1 * sin_half
    
    rotated = jnp.concatenate([rotated_first_half, rotated_second_half], axis=-1)
    
    if partial_rotary_factor < 1.0:
        rotated = jnp.concatenate([rotated, x[..., dim:]], axis=-1)
    
    return rotated.astype(input_dtype)


def repeat_kv(x: jnp.ndarray, n_rep: int) -> jnp.ndarray:
    """Repeat KV heads for grouped-query attention.
    
    Args:
        x: KV tensor [batch, seq_len, num_kv_heads, head_dim]
        n_rep: Number of times to repeat each head
    
    Returns:
        Repeated tensor [batch, seq_len, num_kv_heads * n_rep, head_dim]
    """
    if n_rep == 1:
        return x
    
    batch, seq_len, num_kv_heads, head_dim = x.shape
    x = x[:, :, :, None, :]
    x = jnp.tile(x, (1, 1, 1, n_rep, 1))
    x = x.reshape((batch, seq_len, num_kv_heads * n_rep, head_dim))
    return x


def causal_mask(seq_len: int, kv_len: int) -> jnp.ndarray:
    """Create causal attention mask.
    
    Args:
        seq_len: Query sequence length
        kv_len: Key/value sequence length
    
    Returns:
        Float mask [seq_len, kv_len] with 0.0 for unmasked, -inf for masked
    """
    mask = jnp.triu(jnp.ones((seq_len, kv_len)), k=1)
    # Convert to -inf for masked positions, 0.0 for unmasked (matches HF)
    return jnp.where(mask == 1, -jnp.finfo(jnp.float32).max, 0.0)


def silu(x: jnp.ndarray) -> jnp.ndarray:
    """SiLU activation function."""
    return nn.silu(x)


def gelu(x: jnp.ndarray) -> jnp.ndarray:
    """GeLU activation function."""
    return nn.gelu(x)


def get_activation(name: str):
    """Get activation function by name."""
    activations = {
        "silu": silu,
        "gelu": gelu,
        "gelu_pytorch_tanh": lambda x: nn.gelu(x, approximate=True),
        "relu": nn.relu,
    }
    return activations.get(name, silu)


def causal_conv1d_update(
    x: jnp.ndarray,  # [B, D, 1] - new token
    conv_state: jnp.ndarray,  # [B, D, kernel_size]
    weight: jnp.ndarray,  # [D, kernel_size]
    bias: Optional[jnp.ndarray],  # [D]
    activation: str = "silu",
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Update convolution state and compute output for single token.
    
    Args:
        x: New token projection [B, D, 1]
        conv_state: Sliding window cache [B, D, K]
        weight: Convolution weights [D, K]
        bias: Optional bias [D]
        activation: Activation function
        
    Returns:
        tuple: (output [B, D, 1], updated_conv_state [B, D, K])
    """
    batch, dim, _ = x.shape
    kernel_size = conv_state.shape[-1]
    
    # 1. Shift state left and append new token
    # state[:, :, :-1] = state[:, :, 1:]
    # state[:, :, -1] = x
    new_state = jnp.roll(conv_state, shift=-1, axis=-1)
    new_state = new_state.at[:, :, -1].set(x.squeeze(-1))
    
    # 2. Convolution: dot product of state and weight
    # out = sum(state[:, :, k] * weight[:, k]) for k in kernel
    out = jnp.einsum('bdk,dk->bd', new_state, weight)
    
    # 3. Add bias
    if bias is not None:
        out = out + bias
    
    # 4. Activation
    if activation == "silu":
        out = nn.silu(out)
    
    return out[:, :, None], new_state
