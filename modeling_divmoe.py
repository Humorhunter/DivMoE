"""
Implementation of DivMoE: Domain-Specialized Mixture of Experts with Diversity-Augmented Routing
This model is modified from the Qwen2 architecture
with a mixture of experts approach, where each expert is initialized from a domain-specialized model 
and routing is constrained to enforce diversity.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import transformers
from packaging import version

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import logging
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

# Version-aware imports for compatibility across transformers versions
_transformers_version = version.parse(transformers.__version__)

# Flash Attention availability checks
try:
    from transformers.utils import is_flash_attn_2_available, is_torch_sdpa_available
except ImportError:
    def is_flash_attn_2_available():
        try:
            import flash_attn
            return True
        except ImportError:
            return False
    
    def is_torch_sdpa_available():
        return hasattr(torch.nn.functional, 'scaled_dot_product_attention')

try:
    from transformers.utils import is_flash_attn_greater_or_equal_2_10
except ImportError:
    def is_flash_attn_greater_or_equal_2_10():
        try:
            from flash_attn import __version__ as flash_attn_version
            return version.parse(flash_attn_version) >= version.parse("2.1.0")
        except ImportError:
            return False

# Flash Attention imports
if is_flash_attn_2_available():
    try:
        from flash_attn import flash_attn_func, flash_attn_varlen_func
    except ImportError:
        flash_attn_func = None
        flash_attn_varlen_func = None
else:
    flash_attn_func = None
    flash_attn_varlen_func = None

# Flash attention utils - with fallback for different transformers versions
# Import _get_unpad_data
try:
    from transformers.modeling_flash_attention_utils import _get_unpad_data
except ImportError:
    try:
        from transformers.models.llama.modeling_llama import _get_unpad_data
    except ImportError:
        _get_unpad_data = None
        print("_get_unpad_data not available, flash attention with padding may not work")

# Import index_first_axis (named _index_first_axis in newer transformers)
try:
    from transformers.modeling_flash_attention_utils import _index_first_axis as index_first_axis
except ImportError:
    try:
        from transformers.modeling_flash_attention_utils import index_first_axis
    except ImportError:
        try:
            from transformers.models.llama.modeling_llama import _index_first_axis as index_first_axis
        except ImportError:
            try:
                from transformers.models.llama.modeling_llama import index_first_axis
            except ImportError:
                index_first_axis = None
                print("index_first_axis not available, flash attention may not work")

# Import pad_input (named _pad_input in newer transformers)
try:
    from transformers.modeling_flash_attention_utils import _pad_input as pad_input
except ImportError:
    try:
        from transformers.modeling_flash_attention_utils import pad_input
    except ImportError:
        try:
            from transformers.models.llama.modeling_llama import _pad_input as pad_input
        except ImportError:
            try:
                from transformers.models.llama.modeling_llama import pad_input
            except ImportError:
                pad_input = None
                print("pad_input not available, flash attention may not work")

# Import unpad_input (named _unpad_input in newer transformers)
try:
    from transformers.modeling_flash_attention_utils import _unpad_input as unpad_input
except ImportError:
    try:
        from transformers.modeling_flash_attention_utils import unpad_input
    except ImportError:
        try:
            from transformers.models.llama.modeling_llama import _unpad_input as unpad_input
        except ImportError:
            try:
                from transformers.models.llama.modeling_llama import unpad_input
            except ImportError:
                unpad_input = None
                print("unpad_input not available, flash attention may not work")

# Attention mask utils - with fallback for different transformers versions
try:
    from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
except ImportError:
    try:
        # Try alternate location for older versions
        from transformers.models.llama.modeling_llama import _prepare_4d_causal_attention_mask
    except ImportError:
        # Fallback implementation
        print("prepare_4d_causal_attention_mask not available, flash attention may not work")

logger = logging.get_logger(__name__)

class DivMoEConfig(PretrainedConfig):
    """
    Configuration class for DivMoE model.
    """
    
    model_type = "divmoe"
    
    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 2048,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 2,
        head_dim: Optional[int] = None,  # Head dimension (if None, computed as hidden_size // num_attention_heads)
        intermediate_size: int = 11008,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int = None,
        bos_token_id: int = None,
        eos_token_id: int = None,
        tie_word_embeddings: bool = True,
        use_sliding_window: bool = False,
        sliding_window: Optional[int] = None,
        max_window_layers: Optional[int] = None,
        attention_dropout: float = 0.0,
        attn_implementation: str = "auto",
        torch_dtype: str = "auto",
        num_domain_groups: int = 3,  # Number of domain groups (n)
        experts_per_domain: int = 4,  # Number of experts per domain (m)
        num_experts_per_tok: int = 2,  # Number of experts to route each token to
        use_shared_expert: bool = True,  # Whether to use a shared generalist expert
        aux_loss_alpha: float = 0.01,  # Weight for the auxiliary load balancing loss
        router_bias: bool = False,  # Whether to use bias in the router
        norm_router_scores: bool = True,  # Whether to normalize router scores across all selected experts
        routing_type: str = "diversity_augmented",  # The type of routing mechanism to use
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        # Support explicit head_dim (like Qwen3) or compute from hidden_size
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self.attention_dropout = attention_dropout
        self.attn_implementation = attn_implementation
        self.torch_dtype = torch_dtype
        
        # MoE specific parameters
        self.num_domain_groups = num_domain_groups
        self.experts_per_domain = experts_per_domain
        self.total_num_experts = num_domain_groups * experts_per_domain
        self.num_experts_per_tok = num_experts_per_tok
        self.use_shared_expert = use_shared_expert
        self.aux_loss_alpha = aux_loss_alpha
        self.router_bias = router_bias
        self.norm_router_scores = norm_router_scores
        self.routing_type = routing_type

class BasicMLP(nn.Module):
    """
    Basic MLP module used as the expert networks.
    """
    def __init__(self, config: DivMoEConfig, is_shared_expert: bool = False):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.is_shared_expert = is_shared_expert
        if is_shared_expert:
            self.intermediate_size = config.intermediate_size
        else:
            self.intermediate_size = config.intermediate_size // config.experts_per_domain
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class DivMoERMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Modified from Qwen2RMSNorm.
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class DivMoERotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding.
    Modified from Qwen2RotaryEmbedding.
    """
    def __init__(self, config: DivMoEConfig, device=None):
        super().__init__()
        # Support for different RoPE types
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
            
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        
        # Initialize rotary embeddings for head dimension
        head_dim = config.head_dim
        inv_freq = 1.0 / (config.max_position_embeddings ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq
        self.attention_scaling = 1.0

    def _dynamic_frequency_update(self, position_ids, device):
        """
        Dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            head_dim = self.config.head_dim
            inv_freq = 1.0 / (seq_len ** (torch.arange(0, head_dim, 2).float() / head_dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        
        # Force float32
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float().to(x.device) @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Apply attention scaling
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    
class DivMoEAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.
    Modified from Qwen2Attention.
    """
    def __init__(self, config: DivMoEConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = self._apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Repeat KV heads if needed
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = self.o_proj(attn_output.view(bsz, q_len, -1))
        
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights

    def _apply_rotary_pos_emb(self, q, k, cos, sin, unsqueeze_dim=1):
        """Applies Rotary Position Embedding to the query and key tensors."""
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed

    def _rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeats key/value heads to match the number of query heads."""
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim) 

class DivMoEFlashAttention2(DivMoEAttention):
    """
    DivMoE flash attention module. This module inherits from `DivMoEAttention` but uses the `flash-attn`
    library to significantly speed up attention calculations by handling padded sequences more efficiently.
    This implementation is inspired by the LlamaFlashAttention2 implementation in transformers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # This attribute is used to handle the difference in causal mask alignment between flash_attn<2.1 and flash_attn>=2.1.
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if output_attentions:
            warnings.warn("DivMoEFlashAttention2 does not support `output_attentions=True`", UserWarning)

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = self._apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)

        # Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if hasattr(self.config, "torch_dtype") and self.config.torch_dtype is not None:
                target_dtype = self.config.torch_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            softmax_scale=self.scaling,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, None

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
    ):
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            causal = self.is_causal and query_length != 1

        # Check if we need to handle padding
        # If attention_mask is None or all ones (no padding), use flash_attn_func directly
        use_varlen = False
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # 2D mask: check if there's actual padding (any zeros)
                use_varlen = not attention_mask.all()
            elif attention_mask.dim() == 4:
                # 4D mask: check last row for padding info
                use_varlen = not (attention_mask[:, 0, -1, :] == 0).all()

        if use_varlen:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._unpad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

        return attn_output

    def _unpad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        # For Flash Attention 2, we receive a 2D attention mask (batch_size, seq_length)
        # where 1 indicates valid (non-padded) tokens and 0 indicates padding.
        # We need to convert this to the format expected by flash_attn_varlen_func.
        
        # Handle both 2D and 4D attention masks
        if attention_mask.dim() == 4:
            # 4D mask: extract padding info from [batch, 1, seq, seq]
            # Take the last query row - it can attend to all valid keys
            padding_mask = attention_mask[:, 0, -1, :] == 0
        elif attention_mask.dim() == 2:
            # 2D mask: (batch_size, seq_length), 1=valid, 0=padding
            # Convert to boolean where True = valid token
            padding_mask = attention_mask.bool()
        else:
            raise ValueError(f"Unexpected attention_mask dimensions: {attention_mask.dim()}")
        
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape
        
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(padding_mask)

        # Use direct tensor indexing instead of index_first_axis to avoid version compatibility issues
        # index_select preserves all dimensions correctly
        key_layer = key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim).index_select(0, indices_k)
        value_layer = value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim).index_select(0, indices_k)

        if query_length == kv_seq_len:
            query_layer = query_layer.reshape(batch_size * kv_seq_len, self.num_attention_heads, head_dim).index_select(0, indices_k)
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=query_layer.device)
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            attention_mask_q = padding_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask_q)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )

class DivMoESdpaAttention(DivMoEAttention):
    """
    DivMoE SDPA attention module. This module inherits from `DivMoEAttention` but uses the PyTorch
    scaled dot-product attention function.
    """
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if output_attentions:
            raise ValueError("DivMoESdpaAttention does not support `output_attentions=True`")

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = self._apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)

        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=attention_mask is None and q_len > 1,
            scale=self.scaling,  # Explicit scaling for consistency with eager/flash
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, None

class DivMoERouter(nn.Module):
    """
    Diversity-Augmented Router for DivMoE.
    This router ensures that tokens are routed to at most one expert from each domain group.
    """
    
    def __init__(self, config: DivMoEConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_domain_groups = config.num_domain_groups
        self.experts_per_domain = config.experts_per_domain
        self.total_num_experts = config.total_num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.aux_loss_alpha = config.aux_loss_alpha
        
        # Router projections - one per domain group for more efficient diverse routing
        self.router_weights = nn.Parameter(
            torch.zeros(self.hidden_size, self.total_num_experts)
        )
        if config.router_bias:
            self.router_bias = nn.Parameter(torch.zeros(self.total_num_experts))
        else:
            self.register_parameter("router_bias", None)
        
        # Initialize router weights
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.router_weights, a=math.sqrt(5))
        if self.router_bias is not None:
            bound = 1 / math.sqrt(self.hidden_size)
            nn.init.uniform_(self.router_bias, -bound, bound)
    
    def forward(self, hidden_states):
        """
        Route tokens to experts with diversity constraint.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            
        Returns:
            expert_indices: [batch_size, seq_len, num_experts_per_tok] - indices of selected experts
            expert_weights: [batch_size, seq_len, num_experts_per_tok] - weights for selected experts
            aux_loss: auxiliary load balancing loss
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Reshape hidden states for linear projection
        # [batch_size * seq_len, hidden_size]
        flat_hidden = hidden_states.view(-1, hidden_size)
        
        # Calculate router scores
        # [batch_size * seq_len, total_num_experts]
        router_logits = F.linear(flat_hidden, self.router_weights.transpose(0, 1), self.router_bias)
        
        # Reshape back to include sequence dimension
        # [batch_size, seq_len, total_num_experts]
        router_logits = router_logits.view(batch_size, seq_len, self.total_num_experts)
        
        # Convert to [batch_size, seq_len, num_domain_groups, experts_per_domain]
        router_logits = router_logits.view(
            batch_size, seq_len, self.num_domain_groups, self.experts_per_domain
        )
        
        if self.config.routing_type == "diversity_augmented":
            # Get the expert with highest score from each domain
            # [batch_size, seq_len, num_domain_groups]
            top_expert_per_domain_weight, top_expert_per_domain_idx = torch.max(
                router_logits, dim=-1
            )
            
            # Convert relative indices to absolute indices
            # [batch_size, seq_len, num_domain_groups]
            domain_offsets = torch.arange(
                0, self.total_num_experts, self.experts_per_domain, 
                device=hidden_states.device
            ).view(1, 1, self.num_domain_groups)
            top_expert_per_domain_idx = top_expert_per_domain_idx + domain_offsets
            
            # Sort domains by their top expert's score (highest first)
            # [batch_size, seq_len, num_domain_groups]
            _, sorted_domains = torch.sort(
                top_expert_per_domain_weight, dim=-1, descending=True
            )
            
            # Select top-k domains
            # [batch_size, seq_len, num_experts_per_tok]
            top_k_domains = sorted_domains[:, :, :self.num_experts_per_tok]
            
            # Gather the expert indices from the top-k domains
            # [batch_size, seq_len, num_experts_per_tok]
            expert_indices = torch.gather(
                top_expert_per_domain_idx,
                dim=-1,
                index=top_k_domains
            )
            
            # Gather weights for selected experts
            router_probs = F.softmax(router_logits.view(batch_size, seq_len, -1), dim=-1)
            
            # [batch_size, seq_len, num_experts_per_tok]
            expert_weights = torch.gather(
                router_probs,
                dim=-1,
                index=expert_indices
            )
            
            # Normalize weights if needed
            if self.config.norm_router_scores:
                expert_weights = expert_weights / torch.sum(
                    expert_weights, dim=-1, keepdim=True
                )
        else:
            # Fallback to standard top-k routing across all experts
            router_probs = F.softmax(router_logits.view(batch_size, seq_len, -1), dim=-1)
            expert_weights, expert_indices = torch.topk(
                router_probs, k=self.num_experts_per_tok, dim=-1
            )
            
            # Normalize weights if needed
            if self.config.norm_router_scores:
                expert_weights = expert_weights / torch.sum(
                    expert_weights, dim=-1, keepdim=True
                )
        
        # Calculate auxiliary load balancing loss
        if self.training and self.aux_loss_alpha > 0:
            # Calculate the fraction of tokens routed to each expert
            # [total_num_experts]
            expert_mask = F.one_hot(
                expert_indices.view(-1), num_classes=self.total_num_experts
            ).float()
            
            # [total_num_experts]
            token_fraction = expert_mask.mean(dim=0)
            
            # [total_num_experts]
            router_prob_fraction = router_probs.mean(dim=[0, 1])
            
            # Loss to encourage balanced routing
            # We want each expert to receive a similar fraction of tokens
            # relative to its routing probability
            balance_term = token_fraction * self.total_num_experts
            aux_loss = torch.sum(router_prob_fraction * balance_term) * self.aux_loss_alpha
        else:
            aux_loss = None
            
        return expert_indices, expert_weights, aux_loss

class DivMoEMLP(nn.Module):
    """
    Mixture of Experts MLP layer with domain-specialized experts.
    """
    
    def __init__(self, config: DivMoEConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_domain_groups = config.num_domain_groups
        self.experts_per_domain = config.experts_per_domain
        self.total_num_experts = config.total_num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.use_shared_expert = config.use_shared_expert
        
        # Create domain-specialized experts
        self.experts = nn.ModuleList([
            BasicMLP(config) for _ in range(self.total_num_experts)
        ])
        
        # Create shared generalist expert (if enabled)
        if self.use_shared_expert:
            self.shared_expert = BasicMLP(config, is_shared_expert=True)
        
        # Create router
        self.router = DivMoERouter(config)
        
    def forward(self, hidden_states):
        """
        Forward pass through the MoE layer.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Get routing information
        expert_indices, expert_weights, aux_loss = self.router(hidden_states)
        
        if self.training:
            expert_indices = expert_indices.view(-1, expert_indices.shape[-1])
            expert_weights = expert_weights.view(-1, expert_weights.shape[-1])
            expand_hidden_states = hidden_states.view(-1, hidden_size)
            flat_topk_idx = expert_indices.view(-1)

            expand_hidden_states = expand_hidden_states.repeat_interleave(
                self.num_experts_per_tok, dim=0
            )
            output = torch.empty_like(expand_hidden_states)
            for i, expert in enumerate(self.experts):
                output[flat_topk_idx==i] = expert(expand_hidden_states[flat_topk_idx==i])
            output = (output.view(*expert_weights.shape, -1) * expert_weights.unsqueeze(-1)).sum(dim=1)
            output = output.to(hidden_states.dtype).view(batch_size, seq_len, hidden_size)
        else:
            # Initialize output tensor
            output = torch.zeros_like(hidden_states)

            # Pass through experts
            for i in range(self.num_experts_per_tok):
                # Get the expert indices and weights for this position
                indices = expert_indices[:, :, i].view(-1)  # [batch_size * seq_len]
                weights = expert_weights[:, :, i].view(-1, 1)  # [batch_size * seq_len, 1]
                
                # Flatten hidden states
                flat_hidden = hidden_states.view(-1, hidden_size)  # [batch_size * seq_len, hidden_size]
                
                # Process each expert separately to avoid excessive memory usage
                for expert_idx in range(self.total_num_experts):
                    # Create a mask for tokens assigned to this expert
                    expert_mask = (indices == expert_idx)
                    
                    # Skip if no tokens are assigned to this expert
                    if not expert_mask.any():
                        continue
                    
                    # Get inputs for this expert
                    expert_inputs = flat_hidden[expert_mask]
                    
                    # Forward through expert
                    expert_output = self.experts[expert_idx](expert_inputs)
                    
                    # Get corresponding weights
                    expert_output_weights = weights[expert_mask]
                    
                    # Add to output (weighted)
                    output.view(-1, hidden_size)[expert_mask] += expert_output * expert_output_weights
        
        # Add shared expert contribution (if enabled)
        if self.use_shared_expert:
            shared_output = self.shared_expert(hidden_states)
            if self.config.use_shared_expert:
                # When using shared expert, we scale its contribution by (1 - sum(expert_weights))
                shared_weight = 1.0
                output = output + shared_weight * shared_output
        
        # Attach auxiliary loss if needed
        if aux_loss is not None:
            output = AddAuxiliaryLoss.apply(output, aux_loss)
            
        return output

class AddAuxiliaryLoss(torch.autograd.Function):
    """
    Helper function to add auxiliary loss to the output during training.
    """
    
    @staticmethod
    def forward(ctx, output, loss):
        ctx.loss = loss
        return output
        
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, ctx.loss

DIVMOE_ATTENTION_CLASSES = {
    "eager": DivMoEAttention,
    "flash_attention_2": DivMoEFlashAttention2,
    "sdpa": DivMoESdpaAttention,
}

class DivMoEDecoderLayer(nn.Module):
    """
    DivMoE decoder layer that replaces the MLP with a mixture of experts.
    """
    
    def __init__(self, config: DivMoEConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        attn_implementation = getattr(config, "_attn_implementation", "eager")
        self.self_attn = DIVMOE_ATTENTION_CLASSES[attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = DivMoEMLP(config)
        self.input_layernorm = DivMoERMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DivMoERMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Mixture of Experts MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs

class DivMoEPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained models.
    Modified from Qwen2PreTrainedModel.
    """
    config_class = DivMoEConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DivMoEDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class DivMoEModel(DivMoEPreTrainedModel):
    """
    DivMoE model with mixture of experts layers.
    Modified from Qwen2Model.
    """
    def __init__(self, config: DivMoEConfig):
        super().__init__(config)
        
        if config.attn_implementation == "auto":
            if is_flash_attn_2_available():
                logger.info("Using Flash Attention 2")
                config._attn_implementation = "flash_attention_2"
            elif is_torch_sdpa_available():
                logger.info("Using PyTorch SDPA")
                config._attn_implementation = "sdpa"
            else:
                logger.info("Using Eager Attention")
                config._attn_implementation = "eager"
        else:
            config._attn_implementation = config.attn_implementation

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([DivMoEDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = DivMoERMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = DivMoERotaryEmbedding(config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
            inputs_embeds = self.embed_tokens(input_ids)
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You must specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        if use_cache:
            # Initialize past_key_values if needed
            if past_key_values is None:
                past_key_values = DynamicCache()
            # Handle different DynamicCache API versions
            if hasattr(past_key_values, 'get_usable_length'):
                # Newer transformers versions may require layer_idx
                try:
                    past_key_values_length = past_key_values.get_usable_length(seq_length)
                except TypeError:
                    # Fallback for versions that require layer_idx
                    past_key_values_length = past_key_values.get_usable_length(seq_length, 0)
            elif hasattr(past_key_values, 'get_seq_length'):
                past_key_values_length = past_key_values.get_seq_length()
            else:
                # Fallback: check if cache has any content
                past_key_values_length = past_key_values.seen_tokens if hasattr(past_key_values, 'seen_tokens') else 0

        if position_ids is None:
            position_ids = torch.arange(past_key_values_length, seq_length+past_key_values_length, dtype=torch.long, device=inputs_embeds.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # For Flash Attention 2, we don't convert to 4D mask - it handles padding internally
        # For other attention implementations, we use the 4D causal attention mask
        attn_implementation = getattr(self.config, "_attn_implementation", "eager")
        if attn_implementation == "flash_attention_2":
            # Flash Attention 2 expects 2D mask (batch_size, seq_length) or None
            # Don't convert to 4D
            pass
        else:
            # 4d mask is passed through the layers for eager/sdpa
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        
        # Get position embeddings
        cos_sin = self.rotary_emb(inputs_embeds, position_ids)

        # Forward through layers
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=cos_sin,
                **kwargs,
            )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class DivMoEForCausalLM(DivMoEPreTrainedModel, GenerationMixin):
    """
    DivMoE model with a language modeling head.
    Modified from Qwen2ForCausalLM.
    """
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = DivMoEModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs["last_hidden_state"]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Prepare inputs for generation. This method is required for compatibility with
        newer versions of transformers' GenerationMixin.
        """
        # If we have cache, only use the last token
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens if hasattr(past_key_values, 'seen_tokens') else cache_length
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]

            # Keep only the unprocessed tokens
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask
            if (
                cache_position is not None
                and cache_position[0] > 0
                and attention_mask is not None
                and cache_length is not None
            ):
                # Crop attention mask to cache_length
                attention_mask = attention_mask[:, :cache_length + input_ids.shape[1]]

        # Update position_ids if needed
        if position_ids is None and attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        # If inputs_embeds is passed, we only use it in the first generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @classmethod
    def from_domain_models(
        cls,
        base_model_name_or_path: str,
        domain_model_name_or_paths: List[str],
        num_experts_per_domain: int = 4,
        **kwargs
    ):
        """
        Initialize a DivMoE model from domain-specialized models.
        Each expert in a domain group gets a vertical slice of the MLP weights from the respective domain model.
        """
        from transformers import AutoModelForCausalLM
        
        # Load base model
        logger.info(f"Loading base model from {base_model_name_or_path}")
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path)
        
        # Load domain models
        domain_models = []
        for path in domain_model_name_or_paths:
            logger.info(f"Loading domain model from {path}")
            domain_models.append(AutoModelForCausalLM.from_pretrained(path))
        
        # Validate architecture compatibility between base and domain models
        base_config = base_model.config
        for i, domain_model in enumerate(domain_models):
            domain_config = domain_model.config
            if base_config.hidden_size != domain_config.hidden_size:
                raise ValueError(
                    f"Domain model {i} has incompatible hidden_size: "
                    f"{domain_config.hidden_size} vs base model {base_config.hidden_size}"
                )
            if base_config.intermediate_size != domain_config.intermediate_size:
                raise ValueError(
                    f"Domain model {i} has incompatible intermediate_size: "
                    f"{domain_config.intermediate_size} vs base model {base_config.intermediate_size}"
                )
            if base_config.num_hidden_layers != domain_config.num_hidden_layers:
                raise ValueError(
                    f"Domain model {i} has incompatible num_hidden_layers: "
                    f"{domain_config.num_hidden_layers} vs base model {base_config.num_hidden_layers}"
                )
            if base_config.num_attention_heads != domain_config.num_attention_heads:
                raise ValueError(
                    f"Domain model {i} has incompatible num_attention_heads: "
                    f"{domain_config.num_attention_heads} vs base model {base_config.num_attention_heads}"
                )
            # Check head_dim if present (e.g., Qwen3 uses explicit head_dim)
            base_head_dim = getattr(base_config, 'head_dim', None)
            domain_head_dim = getattr(domain_config, 'head_dim', None)
            if base_head_dim is not None and domain_head_dim is not None and base_head_dim != domain_head_dim:
                raise ValueError(
                    f"Domain model {i} has incompatible head_dim: "
                    f"{domain_head_dim} vs base model {base_head_dim}"
                )
        
        logger.info("All domain models have compatible architectures with base model")
        
        # Create DivMoE config explicitly from base model config
        # This ensures we properly copy all relevant fields
        logger.info(f"Creating DivMoE config with {len(domain_models)} domain groups and {num_experts_per_domain} experts per domain")
        
        config = DivMoEConfig(
            vocab_size=base_config.vocab_size,
            hidden_size=base_config.hidden_size,
            num_hidden_layers=base_config.num_hidden_layers,
            num_attention_heads=base_config.num_attention_heads,
            num_key_value_heads=getattr(base_config, 'num_key_value_heads', base_config.num_attention_heads),
            # Support explicit head_dim (like Qwen3) or compute from hidden_size
            head_dim=getattr(base_config, 'head_dim', None),
            intermediate_size=base_config.intermediate_size,
            hidden_act=getattr(base_config, 'hidden_act', 'silu'),
            max_position_embeddings=base_config.max_position_embeddings,
            initializer_range=getattr(base_config, 'initializer_range', 0.02),
            rms_norm_eps=getattr(base_config, 'rms_norm_eps', 1e-6),
            use_cache=getattr(base_config, 'use_cache', True),
            pad_token_id=base_config.pad_token_id,
            bos_token_id=base_config.bos_token_id,
            eos_token_id=base_config.eos_token_id,
            tie_word_embeddings=getattr(base_config, 'tie_word_embeddings', True),
            use_sliding_window=getattr(base_config, 'use_sliding_window', False),
            sliding_window=getattr(base_config, 'sliding_window', None),
            max_window_layers=getattr(base_config, 'max_window_layers', None),
            attention_dropout=getattr(base_config, 'attention_dropout', 0.0),
            # MoE specific parameters
            num_domain_groups=len(domain_models),
            experts_per_domain=num_experts_per_domain,
            **kwargs
        )

        logger.info(f"Config: {config}")
        
        # Create new model with MoE architecture
        logger.info("Initializing DivMoE model")
        model = cls(config)
        logger.info(f"Number of parameters in the model: {sum(p.numel() for p in model.parameters())}")
        
        # Initialize parameters
        logger.info("Initializing parameters from averaged domain models...")
        with torch.no_grad():
            # Initialize non-MLP parameters as average of all models
            models_to_average = [base_model] + domain_models
            for name, param in model.named_parameters():
                if "mlp" not in name and "router" not in name:
                    for i, source_model in enumerate(models_to_average):
                        source_param_name = name
                        
                        # Handle parameter name differences if needed
                        if name not in dict(source_model.named_parameters()):
                            if name.startswith("model."):
                                # Try without the "model." prefix
                                source_param_name = name[6:]  # Remove "model."
                        
                        if source_param_name in dict(source_model.named_parameters()):
                            source_param = source_model.get_parameter(source_param_name)
                            if i == 0:
                                param.copy_(source_param / len(models_to_average))
                            else:
                                param.add_(source_param / len(models_to_average))
                        else:
                            logger.warning(f"Parameter {name} not found in source model {i}, skipping")
            
            # Initialize experts from domain models
            logger.info("Initializing experts from domain models...")
            expansion_factor = config.num_domain_groups
            granularity = num_experts_per_domain
            expert_scale = (
                expansion_factor * granularity ** 2 / config.num_experts_per_tok
            ) ** (1.0 / 3.0)
            logger.info(
                f"Applying expert weight scaling gamma={expert_scale:.4f} "
                f"(E={expansion_factor}, G={granularity}, k={config.num_experts_per_tok})"
            )
            for layer_idx in range(config.num_hidden_layers):
                # For each domain group
                for domain_idx, domain_model in enumerate(domain_models):
                    logger.info(f"Processing layer {layer_idx}, domain {domain_idx}")
                    
                    # Get the MLP from the source domain model
                    source_mlp = domain_model.model.layers[layer_idx].mlp
                    
                    # Get the experts in this domain group
                    expert_start_idx = domain_idx * num_experts_per_domain
                    expert_end_idx = (domain_idx + 1) * num_experts_per_domain
                    experts = [model.model.layers[layer_idx].mlp.experts[i] for i in range(expert_start_idx, expert_end_idx)]
                    
                    # Get source MLP weights
                    source_gate_weight = source_mlp.gate_proj.weight
                    source_up_weight = source_mlp.up_proj.weight
                    source_down_weight = source_mlp.down_proj.weight
                    
                    # Log source weight shapes
                    logger.info(f"Source weight shapes: gate={source_gate_weight.shape}, up={source_up_weight.shape}, down={source_down_weight.shape}")
                    
                    # Get intermediate size dimensions
                    source_intermediate_size = source_gate_weight.shape[0]
                    target_intermediate_size = experts[0].intermediate_size
                    
                    logger.info(f"Source intermediate size: {source_intermediate_size}, Target intermediate size: {target_intermediate_size}")
                    
                    # Split weights along the intermediate dimension
                    split_size = source_intermediate_size // num_experts_per_domain
                    
                    for expert_idx, expert in enumerate(experts):
                        # Calculate slice indices for this expert
                        start_idx = expert_idx * split_size
                        end_idx = (expert_idx + 1) * split_size
                        
                        # Extract chunk of gate and up projection weights (out_features, in_features)
                        gate_chunk = source_gate_weight[start_idx:end_idx, :] * expert_scale
                        up_chunk = source_up_weight[start_idx:end_idx, :] * expert_scale
                        
                        # Extract chunk of down projection weights (in_features, out_features)
                        # For down_proj, we need to slice along the in_features dimension (dim 1)
                        down_chunk = source_down_weight[:, start_idx:end_idx] * expert_scale
                        
                                                
                        # Verify shapes
                        assert gate_chunk.shape[0] == target_intermediate_size, f"Expected {target_intermediate_size}, got {gate_chunk.shape[0]}"
                        assert up_chunk.shape[0] == target_intermediate_size, f"Expected {target_intermediate_size}, got {up_chunk.shape[0]}"
                        assert down_chunk.shape[1] == target_intermediate_size, f"Expected {target_intermediate_size}, got {down_chunk.shape[1]}"
                        
                        # Copy weights to expert - handle shape mismatches by checking target shapes
                        if expert.gate_proj.weight.shape == gate_chunk.shape:
                            expert.gate_proj.weight.copy_(gate_chunk)
                        else:
                            logger.warning(f"Shape mismatch for gate_proj: source={gate_chunk.shape}, target={expert.gate_proj.weight.shape}")
                            if expert.gate_proj.weight.shape == gate_chunk.t().shape:
                                expert.gate_proj.weight.copy_(gate_chunk.t())
                            else:
                                raise ValueError(f"Cannot resolve shape mismatch for gate_proj: source={gate_chunk.shape}, target={expert.gate_proj.weight.shape}")
                        
                        if expert.up_proj.weight.shape == up_chunk.shape:
                            expert.up_proj.weight.copy_(up_chunk)
                        else:
                            logger.warning(f"Shape mismatch for up_proj: source={up_chunk.shape}, target={expert.up_proj.weight.shape}")
                            if expert.up_proj.weight.shape == up_chunk.t().shape:
                                expert.up_proj.weight.copy_(up_chunk.t())
                            else:
                                raise ValueError(f"Cannot resolve shape mismatch for up_proj: source={up_chunk.shape}, target={expert.up_proj.weight.shape}")
                        
                        if expert.down_proj.weight.shape == down_chunk.shape:
                            expert.down_proj.weight.copy_(down_chunk)
                        else:
                            logger.warning(f"Shape mismatch for down_proj: source={down_chunk.shape}, target={expert.down_proj.weight.shape}")
                            if expert.down_proj.weight.shape == down_chunk.t().shape:
                                expert.down_proj.weight.copy_(down_chunk.t())
                            else:
                                raise ValueError(f"Cannot resolve shape mismatch for down_proj: source={down_chunk.shape}, target={expert.down_proj.weight.shape}")
                
                # Initialize shared expert from base model if enabled
                if config.use_shared_expert:
                    source_mlp = base_model.model.layers[layer_idx].mlp
                    target_shared = model.model.layers[layer_idx].mlp.shared_expert
                    
                    # Log shared expert shapes
                    logger.info(f"Shared expert source shapes: gate={source_mlp.gate_proj.weight.shape}, "
                               f"up={source_mlp.up_proj.weight.shape}, down={source_mlp.down_proj.weight.shape}")
                    logger.info(f"Shared expert target shapes: gate={target_shared.gate_proj.weight.shape}, "
                               f"up={target_shared.up_proj.weight.shape}, down={target_shared.down_proj.weight.shape}")
                    
                    # For the shared expert, apply the same robust shape handling
                    for param_name in ["gate_proj", "up_proj", "down_proj"]:
                        source_param = getattr(source_mlp, param_name).weight
                        target_param = getattr(target_shared, param_name).weight
                        
                        if target_param.shape == source_param.shape:
                            target_param.copy_(source_param)
                        else:
                            logger.warning(f"Shape mismatch for shared expert {param_name}: source={source_param.shape}, target={target_param.shape}")
                            if target_param.shape == source_param.t().shape:
                                target_param.copy_(source_param.t())
                            else:
                                raise ValueError(f"Cannot resolve shape mismatch for shared expert {param_name}: source={source_param.shape}, target={target_param.shape}")
        
        return model


# Register model with transformers Auto classes for automatic loading
def _register_divmoe_model():
    """Register DivMoE model with transformers Auto classes."""
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
    
    try:
        AutoConfig.register("divmoe", DivMoEConfig)
        AutoModel.register(DivMoEConfig, DivMoEModel)
        AutoModelForCausalLM.register(DivMoEConfig, DivMoEForCausalLM)
        logger.info("Successfully registered DivMoE model with transformers Auto classes")
    except Exception as e:
        # Model might already be registered, which is fine
        logger.debug(f"DivMoE model registration: {e}")


# Auto-register when module is imported
_register_divmoe_model()

