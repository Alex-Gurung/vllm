# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen3 model compatible with HuggingFace weights."""
from collections.abc import Iterable
from typing import Optional, Union

import torch
from torch import nn
from transformers import Qwen3Config

from vllm.attention import Attention, AttentionType
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .adapters import as_seq_cls_model
from .interfaces import SupportsLoRA, SupportsPP
from .qwen2 import Qwen2MLP as Qwen3MLP
from .qwen2 import Qwen2Model
from .utils import AutoWeightsLoader, PPMissingLayer, maybe_prefix

logger = init_logger(__name__)


class Qwen3ReasoningProjector(nn.Module):
    """
    Reasoning projector that applies MLP processing and optionally projects to vocab space.
    This is used when reasoning tokens (<|quad_start|>) are encountered.
    """
    
    def __init__(self, config: Qwen3Config, vocab_projection: bool = True):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.vocab_size = config.vocab_size
        self.vocab_projection = vocab_projection
        
        # Main MLP for reasoning processing
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=None,  # No quantization for reasoning projector
            prefix="reasoning_projector.mlp",
        )
        
        # Optional vocab projection components
        if vocab_projection:
            self.norm_before_projection = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
            self.projection_lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.projection_activation = nn.LeakyReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the reasoning projector.
        
        Args:
            x: Input hidden states of shape (batch_size, hidden_size)
            
        Returns:
            Processed hidden states or vocab logits depending on vocab_projection setting
        """
        # Apply MLP processing
        residual = x
        x = self.mlp(x)
        
        if not self.vocab_projection:
            return x
        
        # Add residual connection
        x = residual + x
        
        # Apply normalization before projection
        x = self.norm_before_projection(x)
        
        # Project to vocab space
        x = self.projection_lm_head(x)
        
        # Apply activation
        x = self.projection_activation(x)
        
        return x


class Qwen3Attention(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 max_position: int = 4096 * 32,
                 head_dim: Optional[int] = None,
                 rms_norm_eps: float = 1e-06,
                 qkv_bias: bool = False,
                 rope_theta: float = 10000,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 rope_scaling: Optional[tuple] = None,
                 prefix: str = "",
                 attn_type: str = AttentionType.DECODER) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=self.rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn",
                              attn_type=attn_type)
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        input_document_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # Add qk-norm
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim,
                           self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim,
                           self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)

        # By default, Qwen3 uses causal attention as it is a decoder-only model.
        # You can override the HF config with `is_causal=False` to enable
        # bidirectional attention, which is used in some embedding models
        # (e.g. Alibaba-NLP/gte-Qwen3-7B-instruct)
        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = Qwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            cache_config=cache_config,
            quant_config=quant_config,
            rope_scaling=rope_scaling,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        input_document_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            input_document_ids=input_document_ids,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


ALL_DECODER_LAYER_TYPES = {
    "attention": Qwen3DecoderLayer,
}


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    })
class Qwen3Model(Qwen2Model):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config,
                         prefix=prefix,
                         decoder_layer_type=Qwen3DecoderLayer)


class Qwen3ForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config
        self.model = Qwen3Model(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "model"))

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(config.vocab_size,
                                              config.hidden_size,
                                              quant_config=quant_config,
                                              prefix=maybe_prefix(
                                                  prefix, "lm_head"))
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)
        
        # Add reasoning projector for handling reasoning tokens
        self.reasoning_projector = Qwen3ReasoningProjector(config, vocab_projection=True)
        
        # Reasoning configuration
        self.reasoning_trigger_token_id = 151650  # <|quad_start|> token ID
        self.reasoning_steps = 5  # Number of reasoning steps to perform
        self.reasoning_end_token_id = 151651  # <|quad_end|> token ID

    def _get_last_mlp_and_lm_head(self):
        """
        Utility to robustly find the last MLP and LM head from the base model.
        This is used for initializing the reasoning projector weights.
        """
        # Find the last MLP layer
        last_mlp = None
        try:
            if hasattr(self.model, 'layers') and isinstance(self.model.layers, nn.ModuleList):
                last_mlp = self.model.layers[-1].mlp
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                last_mlp = self.model.model.layers[-1].mlp
        except (IndexError, AttributeError):
            logger.warning("Could not find last MLP layer for reasoning projector initialization")
        
        # Find the LM head
        lm_head = None
        try:
            if hasattr(self, 'lm_head'):
                lm_head = self.lm_head
            elif hasattr(self.model, 'lm_head'):
                lm_head = self.model.lm_head
        except AttributeError:
            logger.warning("Could not find LM head for reasoning projector initialization")
        
        return last_mlp, lm_head

    def _init_reasoning_projector_from_base(self):
        """
        Initialize the reasoning projector weights from the last MLP and LM head.
        This ensures the reasoning projector starts with sensible weights.
        """
        last_mlp, lm_head = self._get_last_mlp_and_lm_head()
        
        if last_mlp is not None:
            try:
                self.reasoning_projector.mlp.load_state_dict(last_mlp.state_dict())
                logger.info("Successfully initialized reasoning projector MLP from last layer")
            except Exception as e:
                logger.warning(f"Failed to initialize reasoning projector MLP: {e}")
        
        if lm_head is not None and self.reasoning_projector.vocab_projection:
            try:
                # For ParallelLMHead, we need to access the weight differently
                if hasattr(lm_head, 'weight') and isinstance(lm_head.weight, torch.Tensor):
                    self.reasoning_projector.projection_lm_head.weight.data.copy_(lm_head.weight.data)
                    logger.info("Successfully initialized reasoning projector vocab projection from LM head")
            except Exception as e:
                logger.warning(f"Failed to initialize reasoning projector vocab projection: {e}")

    def _apply_reasoning_steps(self, hidden_states: torch.Tensor, 
                              past_key_values: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply reasoning steps by running the reasoning projector multiple times.
        
        Args:
            hidden_states: Current hidden states
            past_key_values: Past key-value cache
            
        Returns:
            Updated hidden states and past_key_values after reasoning
        """
        # Get the last hidden state for reasoning
        last_hidden = hidden_states[:, -1:, :]  # Shape: (batch_size, 1, hidden_size)
        
        # Apply reasoning steps
        for step in range(self.reasoning_steps):
            # Get the last token's hidden state
            sample_context_hidden = last_hidden[:, -1, :]  # Shape: (batch_size, hidden_size)
            
            # Apply reasoning projector
            reasoning_projection = self.reasoning_projector(sample_context_hidden)
            
            if self.reasoning_projector.vocab_projection:
                # Project back to hidden space using embedding weights
                if hasattr(self.model, 'embed_tokens') and self.model.embed_tokens is not None:
                    # reasoning_projection is in vocab space, multiply by embeddings to get hidden states
                    # (batch_size, vocab_size) @ (vocab_size, hidden_size) -> (batch_size, hidden_size)
                    reasoning_projection = reasoning_projection @ self.model.embed_tokens.weight
            
            # Add the reasoning projection as a new token
            reasoning_hidden = reasoning_projection.unsqueeze(1)  # Shape: (batch_size, 1, hidden_size)
            last_hidden = torch.cat([last_hidden, reasoning_hidden], dim=1)
        
        # Add the reasoning end token
        if hasattr(self.model, 'embed_tokens') and self.model.embed_tokens is not None:
            end_token_ids = torch.tensor([[self.reasoning_end_token_id]], 
                                        device=hidden_states.device, dtype=torch.long)
            end_token_embeddings = self.model.embed_tokens(end_token_ids)
            last_hidden = torch.cat([last_hidden, end_token_embeddings], dim=1)
        
        # Update the hidden states with reasoning results
        updated_hidden_states = torch.cat([hidden_states, last_hidden[:, 1:, :]], dim=1)
        
        return updated_hidden_states, past_key_values

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        input_document_ids: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        # Check for reasoning trigger tokens
        if input_ids is not None and input_ids.shape[0] == 1:  # Only support batch size 1 for now
            batch_idx = 0
            if self.reasoning_trigger_token_id in input_ids[batch_idx]:
                # Find all occurrences of reasoning trigger tokens
                trigger_indices = [i for i, x in enumerate(input_ids[batch_idx]) if x == self.reasoning_trigger_token_id]
                
                # Process each section that ends with a reasoning trigger
                current_hidden_states = None
                processed_tokens = 0
                
                for trigger_idx in trigger_indices:
                    # Get tokens from last processed position up to and including the trigger
                    section_start = processed_tokens
                    section_end = trigger_idx + 1  # Include the trigger token
                    
                    # Extract section input_ids
                    section_input_ids = input_ids[batch_idx:batch_idx+1, section_start:section_end]
                    section_positions = positions[section_start:section_end]
                    
                    # Run normal forward on this section
                    section_hidden = self.model(
                        section_input_ids, 
                        section_positions, 
                        intermediate_tensors,
                        inputs_embeds,
                        input_document_ids
                    )
                    
                    # Apply reasoning steps
                    if current_hidden_states is None:
                        current_hidden_states = section_hidden
                    else:
                        current_hidden_states = torch.cat([current_hidden_states, section_hidden], dim=1)
                    
                    # Apply reasoning steps to the current hidden states
                    current_hidden_states, _ = self._apply_reasoning_steps(current_hidden_states)
                    processed_tokens = section_end
                
                # Process any remaining tokens after the last reasoning trigger
                if processed_tokens < input_ids.shape[1]:
                    remaining_input_ids = input_ids[batch_idx:batch_idx+1, processed_tokens:]
                    remaining_positions = positions[processed_tokens:]
                    
                    remaining_hidden = self.model(
                        remaining_input_ids,
                        remaining_positions,
                        intermediate_tensors,
                        None,  # No inputs_embeds for remaining tokens
                        input_document_ids
                    )
                    
                    if current_hidden_states is not None:
                        current_hidden_states = torch.cat([current_hidden_states, remaining_hidden], dim=1)
                    else:
                        current_hidden_states = remaining_hidden
                
                if current_hidden_states is not None:
                    return current_hidden_states
                else:
                    # Fallback to normal forward if something went wrong
                    return self.model(input_ids, positions, intermediate_tensors,
                                     inputs_embeds, input_document_ids=input_document_ids)
        
        # No reasoning triggers found, run normal forward
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds, input_document_ids=input_document_ids)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        loaded_weights = loader.load_weights(weights)
        
        # Initialize reasoning projector after loading base weights
        self._init_reasoning_projector_from_base()
        
        return loaded_weights


Qwen3ForSequenceClassification = as_seq_cls_model(Qwen3ForCausalLM)