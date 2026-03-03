from typing import Tuple, Optional, Callable, Dict, Any

import torch
import torch.nn as nn
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb,Qwen2_5_VLRotaryEmbedding,eager_attention_forward
from transformers.activations import ACT2FN

from ..quantization.qlinear import QLinear
from ..quantization.quantizer import Quantizer
from ..transforms.transforms import BaseTransform, IdentityTransform


# class QuantizedQwen3MLP(nn.Module):

#     def __init__(
#         self, 
#         config: Qwen3Config,
#         weight_quantizer_kwargs: Dict[str, Any] | None = None,
#         act_quantizer_kwargs: Dict[str, Any] | None = None,
#         gate_up_in_transform: BaseTransform = IdentityTransform(),
#         down_in_transform: BaseTransform = IdentityTransform()
#     ):
#         super().__init__()
#         # gate, up accept the same input
#         gate_up_act_quantizer = Quantizer(**act_quantizer_kwargs) if act_quantizer_kwargs else None
#         # Init layers   
#         self.up_proj = QLinear(
#             config.hidden_size,
#             config.intermediate_size,
#             bias=False,
#             weight_quantizer=Quantizer(**weight_quantizer_kwargs) if weight_quantizer_kwargs else None,
#             act_quantizer=gate_up_act_quantizer
#         )
#         self.gate_proj = QLinear(
#             config.hidden_size,
#             config.intermediate_size,
#             bias=False,
#             weight_quantizer=Quantizer(**weight_quantizer_kwargs) if weight_quantizer_kwargs else None,
#             act_quantizer=gate_up_act_quantizer
#         )
#         self.down_proj = QLinear(
#             config.intermediate_size,
#             config.hidden_size,
#             bias=False,
#             weight_quantizer=Quantizer(**weight_quantizer_kwargs) if weight_quantizer_kwargs else None,
#             act_quantizer=Quantizer(**act_quantizer_kwargs) if act_quantizer_kwargs else None
#         )
#         self.act_fn = ACT2FN[config.hidden_act] 

#         self.gate_up_in_transform = gate_up_in_transform
#         self.down_in_transform = down_in_transform

#         self._train_mode = True

#     def forward(self, x: torch.Tensor):
#         # Rotate input
#         x = self.gate_up_in_transform(x)
#         # Get up and gate projection outputs
#         up = self.up_proj(x, self.gate_up_in_transform)
#         gate = self.gate_proj(x, self.gate_up_in_transform)
#         # Apply activation function
#         x = self.act_fn(gate) * up
#         # Get down projection output
#         x = self.down_in_transform(x)
#         down = self.down_proj(x, self.down_in_transform)
#         return down

#     def fix_parametrization(self):
#         # Fix layer parametrizations
#         self.up_proj.fix_parametrization(self.gate_up_in_transform)
#         self.gate_proj.fix_parametrization(self.gate_up_in_transform)
#         self.down_proj.fix_parametrization(self.down_in_transform)

#         self._train_mode = False

class QuantizedQwen2MLP(nn.Module):
    def __init__(
        self, 
        config,
        weight_quantizer_kwargs: Dict[str, Any] | None = None,
        act_quantizer_kwargs: Dict[str, Any] | None = None,
        gate_up_in_transform: BaseTransform = IdentityTransform(),
        down_in_transform: BaseTransform = IdentityTransform()
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        gate_up_act_quantizer = Quantizer(**act_quantizer_kwargs) if act_quantizer_kwargs else None
        # self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # Init layers   
        self.up_proj = QLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            weight_quantizer=Quantizer(**weight_quantizer_kwargs) if weight_quantizer_kwargs else None,
            act_quantizer=gate_up_act_quantizer
        )
        self.gate_proj = QLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            weight_quantizer=Quantizer(**weight_quantizer_kwargs) if weight_quantizer_kwargs else None,
            act_quantizer=gate_up_act_quantizer
        )
        self.down_proj = QLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            weight_quantizer=Quantizer(**weight_quantizer_kwargs) if weight_quantizer_kwargs else None,
            act_quantizer=Quantizer(**act_quantizer_kwargs) if act_quantizer_kwargs else None
        )       
        self.act_fn = ACT2FN[config.hidden_act]

        self.gate_up_in_transform = gate_up_in_transform
        self.down_in_transform = down_in_transform

        self._train_mode = True

    def forward(self, x):
        # Rotate input
        x = self.gate_up_in_transform(x)
        # Get up and gate projection outputs
        up = self.up_proj(x, self.gate_up_in_transform)
        gate = self.gate_proj(x, self.gate_up_in_transform)
        # Apply activation function
        x = self.act_fn(gate) * up
        # Get down projection output
        x = self.down_in_transform(x)
        down = self.down_proj(x, self.down_in_transform)
        return down

    def fix_parametrization(self):
        # Fix layer parametrizations
        self.up_proj.fix_parametrization(self.gate_up_in_transform)
        self.gate_proj.fix_parametrization(self.gate_up_in_transform)
        self.down_proj.fix_parametrization(self.down_in_transform)

        self._train_mode = False

# class QuantizedQwen3Attention(nn.Module):

#     def __init__(
#         self, 
#         config: Qwen3Config, 
#         layer_idx: int,
#         weight_quantizer_kwargs: Dict[str, Any] | None = None,
#         act_quantizer_kwargs: Dict[str, Any] | None = None,
#         qkv_in_transform: BaseTransform = IdentityTransform(),
#         o_in_transform: BaseTransform = IdentityTransform()
#     ):
#         super().__init__()
#         self.config = config
#         self.layer_idx = layer_idx
#         self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
#         self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
#         self.scaling = self.head_dim ** -0.5
#         self.attention_dropout = config.attention_dropout
#         self.is_causal = True

#         # q, k, v accept the same input
#         qkv_act_quantizer = Quantizer(**act_quantizer_kwargs) if act_quantizer_kwargs else None
        
#         self.q_proj = QLinear(
#             config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias,
#             weight_quantizer=Quantizer(**weight_quantizer_kwargs) if weight_quantizer_kwargs else None,
#             act_quantizer=qkv_act_quantizer
#         )
#         self.k_proj = QLinear(
#             config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias,
#             weight_quantizer=Quantizer(**weight_quantizer_kwargs) if weight_quantizer_kwargs else None,
#             act_quantizer=qkv_act_quantizer
#         )
#         self.v_proj = QLinear(
#             config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias,
#             weight_quantizer=Quantizer(**weight_quantizer_kwargs) if weight_quantizer_kwargs else None,
#             act_quantizer=qkv_act_quantizer
#         )
#         self.o_proj = QLinear(
#             config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias,
#             weight_quantizer=Quantizer(**weight_quantizer_kwargs) if weight_quantizer_kwargs else None,
#             act_quantizer=Quantizer(**act_quantizer_kwargs) if act_quantizer_kwargs else None
#         )

#         self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
#         self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
#         self.sliding_window = config.sliding_window
        
#         # Init transformations
#         self.qkv_in_transform = qkv_in_transform
#         self.o_in_transform = o_in_transform

#         self._train_mode = True

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         position_embeddings: Tuple[torch.Tensor, torch.Tensor],
#         attention_mask: Optional[torch.Tensor],
#         past_key_value: Optional[Cache] = None,
#         cache_position: Optional[torch.LongTensor] = None,
#         **kwargs: Unpack[FlashAttentionKwargs],
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#         input_shape = hidden_states.shape[:-1]
#         hidden_shape = (*input_shape, -1, self.head_dim)

#         # Rotate input
#         hidden_states = self.qkv_in_transform(hidden_states)

#         query_states =  self.q_norm(self.q_proj(hidden_states, self.qkv_in_transform).view(hidden_shape)).transpose(1, 2)
#         key_states = self.k_norm(self.k_proj(hidden_states, self.qkv_in_transform).view(hidden_shape)).transpose(1, 2)
#         value_states = self.v_proj(hidden_states, self.qkv_in_transform).view(hidden_shape).transpose(1, 2)

#         cos, sin = position_embeddings
#         query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

#         if past_key_value is not None:
#             # sin and cos are specific to RoPE models; cache_position needed for the static cache
#             cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
#             key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

#         attention_interface: Callable = eager_attention_forward

#         if self.config._attn_implementation != "eager":
#             if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
#                 ValueError(
#                     "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
#                     'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
#                 )
#             else:
#                 attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

#         attn_output, attn_weights = attention_interface(
#             self,
#             query_states,
#             key_states,
#             value_states,
#             attention_mask,
#             dropout=0.0 if not self.training else self.attention_dropout,
#             scaling=self.scaling,
#             sliding_window=self.sliding_window,  # diff with Llama
#             **kwargs,
#         )

#         attn_output = attn_output.reshape(*input_shape, -1).contiguous()
#         # Rotate attn output
#         attn_output = self.o_in_transform(attn_output)
#         attn_output = self.o_proj(attn_output, self.o_in_transform)
#         return attn_output, attn_weights

#     def fix_parametrization(self):
#         # Fix layer parametrizations
#         self.q_proj.fix_parametrization(self.qkv_in_transform)
#         self.k_proj.fix_parametrization(self.qkv_in_transform)
#         self.v_proj.fix_parametrization(self.qkv_in_transform)
#         self.o_proj.fix_parametrization(self.o_in_transform)

#         self._train_mode = False





class QuantizedQwen2_5_VLAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, 
        config: Qwen2_5_VLTextConfig, 
        layer_idx: Optional[int] = None,
        weight_quantizer_kwargs: Dict[str, Any] | None = None,
        act_quantizer_kwargs: Dict[str, Any] | None = None,
        qkv_in_transform: BaseTransform = IdentityTransform(),
        o_in_transform: BaseTransform = IdentityTransform()
        ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            # logger.warning_once(
            #     f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
            #     "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
            #     "when creating this class."
            # )
            pass

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.rope_scaling = config.rope_scaling
        self.scaling = self.head_dim**-0.5

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        qkv_act_quantizer = Quantizer(**act_quantizer_kwargs) if act_quantizer_kwargs else None

        # self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        # self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        # self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        # self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.q_proj = QLinear(
            self.hidden_size, self.num_heads * self.head_dim, bias=True,
            weight_quantizer=Quantizer(**weight_quantizer_kwargs) if weight_quantizer_kwargs else None,
            act_quantizer=qkv_act_quantizer
        )
        self.k_proj = QLinear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True,
            weight_quantizer=Quantizer(**weight_quantizer_kwargs) if weight_quantizer_kwargs else None,
            act_quantizer=qkv_act_quantizer
        )
        self.v_proj = QLinear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True,
            weight_quantizer=Quantizer(**weight_quantizer_kwargs) if weight_quantizer_kwargs else None,
            act_quantizer=qkv_act_quantizer
        )
        self.o_proj = QLinear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False,
            weight_quantizer=Quantizer(**weight_quantizer_kwargs) if weight_quantizer_kwargs else None,
            act_quantizer=Quantizer(**act_quantizer_kwargs) if act_quantizer_kwargs else None
        )
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=config)

        # Init transformations
        self.qkv_in_transform = qkv_in_transform
        self.o_in_transform = o_in_transform

        self._train_mode = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # Rotate input
        hidden_states = self.qkv_in_transform(hidden_states)

        # query_states = self.q_proj(hidden_states)
        # key_states = self.k_proj(hidden_states)
        # value_states = self.v_proj(hidden_states)
        query_states = self.q_proj(hidden_states,self.qkv_in_transform)
        key_states = self.k_proj(hidden_states,self.qkv_in_transform)
        value_states = self.v_proj(hidden_states,self.qkv_in_transform)


        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            position_ids=position_ids,  # pass positions for FA2
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        # attn_output = self.o_proj(attn_output)

        # Rotate attn output
        attn_output = self.o_in_transform(attn_output)
        attn_output = self.o_proj(attn_output, self.o_in_transform)
        return attn_output, attn_weights

    def fix_parametrization(self):
        # Fix layer parametrizations
        self.q_proj.fix_parametrization(self.qkv_in_transform)
        self.k_proj.fix_parametrization(self.qkv_in_transform)
        self.v_proj.fix_parametrization(self.qkv_in_transform)
        self.o_proj.fix_parametrization(self.o_in_transform)

        self._train_mode = False