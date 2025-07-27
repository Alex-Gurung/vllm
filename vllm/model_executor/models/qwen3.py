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
from typing import Optional, Union, List

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

from .interfaces import SupportsLoRA, SupportsPP
from .qwen2 import Qwen2MLP as Qwen3MLP
from .qwen2 import Qwen2Model
from .utils import AutoWeightsLoader, PPMissingLayer, maybe_prefix

from transformers import AutoTokenizer
from vllm.forward_context import ForwardContext, get_forward_context, set_forward_context
import copy

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")


logger = init_logger(__name__)

class Qwen3MLPWithVocabProjection(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        vocab_projection: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            # intermediate_size=self.hidden_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

        self.vocab_projection = vocab_projection
        if vocab_projection:
            self.norm_before_projection = RMSNorm(config.hidden_size,
                eps=config.rms_norm_eps)
            self.projection_lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.projection_activation = nn.LeakyReLU()
            # self.projection_activation = nn.Tanh()
        # print the sizes of the layers
        # print(f"norm_before_projection: {self.norm_before_projection.weight.shape}")
        # print(f"projection_lm_head: {self.projection_lm_head.weight.shape}")
        # print(f"mlp: {self.mlp.weight.shape}")
        # print(f"vocab_size: {config.vocab_size}")
        # print(f"hidden_size: {config.hidden_size}")
        # print(f"intermediate_size: {config.intermediate_size}")
        # x = 1/0
        
    def forward(self, x):
        residual = x
        x = self.mlp(x)
        if not self.vocab_projection:
            return x
        x = residual + x
        x = self.norm_before_projection(x)
        x = self.projection_lm_head(x)
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
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


ALL_DECODER_LAYER_TYPES = {
    "attention": Qwen3DecoderLayer,
}

def check_if_matches_special_sequence(
    input_ids: torch.Tensor,
    special_start_sequence: List[int],
    special_end_sequence: List[int]
) -> Optional[List[int]]:
    input_list = input_ids.tolist()

    len_end = len(special_end_sequence)
    len_start = len(special_start_sequence)

    # Minimum total length to contain start + end + gap (gap can be 0)
    min_total_length = len_start + len_end
    if len(input_list) < min_total_length:
        return None

    print(f"input_list: {input_list}")
    print(f"special_end_sequence: {special_end_sequence}")
    # Check if the sequence ends with the end sequence
    if input_list[-len_end:].tolist() != special_end_sequence:
        return None

    # Search for the start sequence within the allowed gap (0 to 3 tokens)
    for gap in range(0, 4):  # inclusive 0 to 3
        start_idx = -(len_end + gap + len_start)
        end_idx = -(len_end + gap) if (len_end + gap) != 0 else None
        if input_list[start_idx:end_idx].tolist() == special_start_sequence:
            # Return the tokens in the gap
            gap_start = end_idx
            gap_end = -len_end if len_end != 0 else None
            gap_tokens = input_list[gap_start:gap_end]
            return gap_tokens

    return None

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
        self.handles_full_sequence = True
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        # self.vocab_projection = True
        self.vocab_projection = False
        self.reasoning_projector = Qwen3MLPWithVocabProjection(config, 
            cache_config, 
            quant_config, 
            prefix="reasoning_projector",
            vocab_projection=self.vocab_projection)

    def normal_forward(self, input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for layer in self.layers[self.start_layer:self.end_layer]:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def reasoning_forward(self, 
        num_reasoning_steps: int,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        # print(f"num_reasoning_steps: {num_reasoning_steps}")
        # print(f"input_ids: {input_ids}")
        # print(f"positions: {positions}")
        # print(f"intermediate_tensors: {intermediate_tensors}")
        # print(f"inputs_embeds: {inputs_embeds}")

        # attn_metadata = copy.deepcopy(get_forward_context())
        # attn_metadata = get_forward_context().attn_metadata
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings(input_ids)
        
        # x = 1/0
        # projections = []
        for idx in range(num_reasoning_steps):
            # with torch.no_grad():
            outputs = self.normal_forward(input_ids=None, 
                    positions=positions + idx, 
                    intermediate_tensors=None, 
                    inputs_embeds=inputs_embeds)
            # pass it to reasoning projector
            if idx != num_reasoning_steps - 1:
                inputs_embeds = self.reasoning_projector(outputs)
                # print(self.reasoning_projector)
                # reasoning_projection is in vocab space, we multiply by self.llm's token embeddings to get
                # (batch_size, vocab_size) X (vocab_size, hidden_size) -> (batch_size, hidden_size)
                # print(f"inputs_embeds shape: {inputs_embeds.shape}")
                # print(f"embed_tokens.weight shape: {self.embed_tokens.weight.shape}")
                if self.reasoning_projector.vocab_projection:
                    inputs_embeds = inputs_embeds @ self.embed_tokens.weight.to(inputs_embeds.device)
                # projections.append(inputs_embeds)
        
        # set_forward_context(attn_metadata)
        # by default positions looks like [235] or something where 235 is the current position
        # we want the new positions to look like [235, 236, 237, ...]
        # we can use arange to get a list like [1, 2, 3]
        # original_position = positions[0]
        # positions_to_add = torch.arange(1, num_reasoning_steps + 1)
        # positions_to_add += original_position
        # positions_to_add = positions_to_add.to(positions.device)
        # positions = torch.cat([positions, positions_to_add], dim=0)
        # print(f"positions: {positions}")

        # inputs_embeds = torch.stack([p.squeeze().to(inputs_embeds.device) for p in projections], dim=0).unsqueeze(0)
        # print(f"inputs_embeds shape: {inputs_embeds.shape}")
        # # now run forward with all of the projections
        # # return self.normal_forward(input_ids=None, 
        # #         positions=positions, 
        # #         intermediate_tensors=None, 
        # #         inputs_embeds=inputs_embeds)
        return outputs

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        all_token_ids: Optional[torch.Tensor] = None,
    # ) -> Union[torch.Tensor, IntermediateTensors]:
    ) -> Union[tuple[Union[torch.Tensor, IntermediateTensors], int], Union[torch.Tensor, IntermediateTensors]]:
        # print(f"INSIDE FORWARD")
        # print(f"positions: {positions}")
        # special_start_token = 151650
        # special_end_token = 151651
        # special_start_token = 151665
        # special_end_token = 151666
        # special_start_token = 151657
        # special_end_token = 151658
        special_start_sequence = [27, 30940, 5854, 2450, 29]
        special_end_sequence = [522, 30940, 5854, 2450, 29]
        do_reasoning = False
        num_reasoning_steps = 0
        if inputs_embeds is None and all_token_ids is not None:
            # print(f"all_token_ids: {all_token_ids[-10:]}")
            # print(f"input_ids shape: {input_ids.shape}")
            is_special_end_token = (input_ids == special_end_sequence[-1]).sum()
            # print(f"input_ids: {input_ids}")
            # print(f"is_special_end_token: {is_special_end_token}")
            
            if input_ids.shape[0] == 1 and is_special_end_token == 1:
                match = check_if_matches_special_sequence(input_ids, special_start_sequence, special_end_sequence)
                if match is not None:
                    # decode the tokens in between
                    tokens = all_token_ids[match[0]+1:match[1]]
                    decoded_tokens = tokenizer.decode(tokens)
                    # print(f"Decoded Tokens: {decoded_tokens}")
                    # decoded tokens should be a number
                    try:
                        num_reasoning_steps = int(decoded_tokens)
                        do_reasoning = True
                    except ValueError:
                        pass
                        # print(f"Decoded tokens are not a number: {decoded_tokens}")
                else:
                    pass

                
                # # print(f"Found {special_end_token} in input_ids! should search backwards inside all_token_ids")
                # # search backwards to find the first 151651
                # # in theory only the last token is 151666
                # # search backwards to find the first 151650
                # end = all_token_ids.shape[0] - 1
                # start = None
                # for i in range(end, -1, -1):
                #     if all_token_ids[i] == special_start_token:
                #         # print(f"Found {special_start_token} at position {i}")
                #         start = i
                #         break
                #     if end - i > 3:
                #         break
                # if start is None:
                #     # print(f"No {special_start_token} found, skipping")
                #     pass
                # else:
                #     # now decode what's in between 
                #     # print(f"Decoding from {start} to {end}")
                #     # decode the tokens in between
                #     tokens = all_token_ids[start+1:end]
                #     # print(f"Tokens: {tokens}")
                #     decoded_tokens = tokenizer.decode(tokens)
                #     # print(f"Decoded Tokens: {decoded_tokens}")
                #     # decoded tokens should be a number
                #     try:
                #         num_reasoning_steps = int(decoded_tokens)
                #         # print(f"Number of steps: {num_reasoning_steps}")
                #         do_reasoning = True
                #     except ValueError:
                #         pass
                #         # print(f"Decoded tokens are not a number: {decoded_tokens}")

        if not do_reasoning:
            outputs = self.normal_forward(input_ids, 
                    positions, 
                    intermediate_tensors, 
                    inputs_embeds)
            if all_token_ids is not None:
                return outputs, 0
            else:
                return outputs
        # doing reasoning for the next num_steps
        # print(f"doing reasoning for {num_reasoning_steps} steps")
        outputs = self.reasoning_forward(num_reasoning_steps, 
            input_ids, 
            positions, 
            intermediate_tensors, 
            inputs_embeds)
        if all_token_ids is not None:
            return outputs, num_reasoning_steps
        else:
            return outputs


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
    handles_full_sequence = True

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

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        all_token_ids: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        # print(f"all_token_ids: {all_token_ids}")
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds, all_token_ids=all_token_ids)
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
        return loader.load_weights(weights)
