# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/qwen2/modeling_qwen2.py
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
"""Inference-only Qwen2 model compatible with HuggingFace weights."""
from collections.abc import Iterable
from typing import Any, Optional, Union

import torch
from torch import nn
from transformers import Qwen2Config

from vllm.attention import Attention, AttentionType
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsPP
from .utils import (AutoWeightsLoader, PPMissingLayer, extract_layer_index,
                    is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)

from transformers import AutoTokenizer
from typing import List, Optional

class Qwen2MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Qwen2Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        rope_theta: float = 10000,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        rope_scaling: Optional[tuple] = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        dual_chunk_attention_config: Optional[dict[str, Any]] = None,
    ) -> None:
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
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        dual_chunk_attention_config = None
        self.dual_chunk_attention_config = dual_chunk_attention_config

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
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
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            attn_type=attn_type,
            prefix=f"{prefix}.attn",
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": dual_chunk_attention_config,
            } if dual_chunk_attention_config else {})

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen2DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        dual_chunk_attention_config = getattr(config,
                                              "dual_chunk_attention_config",
                                              None)

        # By default, Qwen2 uses causal attention as it is a decoder-only model.
        # You can override the HF config with `is_causal=False` to enable
        # bidirectional attention, which is used in some embedding models
        # (e.g. Alibaba-NLP/gte-Qwen2-7B-instruct)
        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = Qwen2Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            cache_config=cache_config,
            quant_config=quant_config,
            rope_scaling=rope_scaling,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        self.mlp = Qwen2MLP(
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


def check_if_matches_special_sequence(
    input_ids: torch.Tensor,
    special_start_sequences: List[List[int]],
    special_end_sequences: List[List[int]],
    tokenizer: AutoTokenizer,
) -> Optional[List[int]]:
    input_list = input_ids.tolist()
    # print(f"input_list: {input_list}")
    # print(f"special_start_sequence: {special_start_sequence}")
    # print(f"special_end_sequence: {special_end_sequence}")

    len_end = len(special_end_sequences[0])
    len_start = len(special_start_sequences[0])

    # Minimum total length to contain start + end + gap (gap can be 0)
    min_total_length = len_start + len_end
    if len(input_list) < min_total_length:
        # print(f"input_list is too short")
        return None

    # print(f"input_list: {input_list[-20:]}")
    # print(f"decoded: {TOKENIZER.decode(input_list[-20:])}")
    # print(f"special_end_sequence: {special_end_sequence}")

    # Check if the sequence ends with the end sequence
    if input_list[-len_end:] not in special_end_sequences:
        # print(f"input_list does not end with special_end_sequence: {input_list[-len_end:]} != {special_end_sequence}")
        return None

    # print("MATCHES END")
    # print(f"input_list: {input_list[-20:]}")
    # print(f"decoded: {tokenizer.decode(input_list[-20:])}")
    # print(f"special_end_sequences: {special_end_sequences}")

    # Search for the start sequence within the allowed gap (0 to 3 tokens)
    for gap in range(0, 4):  # inclusive 0 to 3
        start_idx = -(len_end + gap + len_start)
        end_idx = -(len_end + gap) if (len_end + gap) != 0 else None
        substring = input_list[start_idx:end_idx]
        # print(f"decoded substring: {tokenizer.decode(substring)}")
        # print(f"substring: {substring}")
        # print(f"special_start_sequences: {special_start_sequences}")
        if substring in special_start_sequences:
            # Return the tokens in the gap
            gap_start = end_idx
            gap_end = -len_end if len_end != 0 else None
            gap_tokens = input_list[gap_start:gap_end]
            # print("FOUND")
            return gap_tokens
    # print("NO MATCH")
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
class Qwen2Model(nn.Module):

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 decoder_layer_type: type[nn.Module] = Qwen2DecoderLayer):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        # TODO (@robertgshaw2): see if this can be moved out
        if (cache_config.sliding_window is not None
                and hasattr(config, "max_window_layers")):
            assert config.max_window_layers == config.num_hidden_layers, (
                "Sliding window for some but all layers is not supported. "
                "This model uses sliding window but `max_window_layers` = {} "
                "is less than `num_hidden_layers` = {}. Please open an issue "
                "to discuss this feature.".format(
                    config.max_window_layers,
                    config.num_hidden_layers,
                ))

        self.config = config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Use the provided decoder layer type or default to Qwen2DecoderLayer
        decoder_layer_type = decoder_layer_type or Qwen2DecoderLayer
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: decoder_layer_type(config=config,
                                              cache_config=cache_config,
                                              quant_config=quant_config,
                                              prefix=prefix),
            prefix=f"{prefix}.layers",
        )

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.reasoning_projector = Qwen2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix="reasoning_projector",
        )
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct-1M")

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def normal_forward(
        self,
        input_ids: torch.Tensor,
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

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if (self.quant_config is not None and
                (scale_name := self.quant_config.get_cache_scale(name))):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = (loaded_weight if loaded_weight.dim() == 0 else
                                 loaded_weight[0])
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

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
            # print(f"idx: {idx}")
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
                # if self.reasoning_projector.vocab_projection:
                #     inputs_embeds = inputs_embeds @ self.embed_tokens.weight.to(inputs_embeds.device)
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
        # print(f"input_ids: {input_ids}")
        # print(f"all_token_ids: {all_token_ids}")
        # print(f"positions: {positions}")
        # special_start_token = 151650
        # special_end_token = 151651
        # special_start_token = 151665
        # special_end_token = 151666
        # special_start_token = 151657
        # special_end_token = 151658
        # special_start_sequences = [
        #     [27, 30940, 5854, 2450, 29], # <implicit_thought>
        #     [366, 30940, 5854, 2450, 29], # " <implicit_thought>"
        # ]
        # special_end_sequence = [522, 30940, 5854, 2450, 29]
        special_start_sequences = [
            [27, 30940, 5854, 2450, 29],  # <implicit_thought>
            [366, 30940, 5854, 2450, 29],  # " <implicit_thought>"
            [15757, 30940, 5854, 2450, 29],
            [1784, 30940, 5854, 2450, 29],
            [22476, 30940, 5854, 2450, 29]
        ]
        special_end_sequences = [
            [522, 30940, 5854, 2450, 29],
            [522, 30940, 5854, 2450, 397],
            [522, 30940, 5854, 2450, 1339],
            [522, 30940, 5854, 2450, 10370],
            [522, 30940, 5854, 2450, 14276],
            [522, 30940, 5854, 2450, 1472],
            [522, 30940, 5854, 2450, 9877],
        ]

        do_reasoning = False
        num_reasoning_steps = 0
        if inputs_embeds is None and all_token_ids is not None:
            # print(f"all_token_ids: {all_token_ids[-10:]}")
            # print(f"input_ids shape: {input_ids.shape}")
            # is_special_end_token = (input_ids in [s[-1] for s in special_end_sequences]).sum()
            is_special_end_token = input_ids[-1] in [s[-1] for s in special_end_sequences]
            # print(f"checking input_ids: {input_ids}")
            # print(f"end of special_end_sequences: {[s[-1] for s in special_end_sequences]}")
            # print(f"is_special_end_token: {is_special_end_token}")
            
            if input_ids.shape[0] == 1 and is_special_end_token:
                match = check_if_matches_special_sequence(all_token_ids, 
                    special_start_sequences, special_end_sequences, self.tokenizer)
                if match is not None:
                    # decode the tokens in between
                    # tokens = all_token_ids[match[0]+1:match[1]]
                    # decoded_tokens = self.tokenizer.decode(tokens)
                    decoded_tokens = self.tokenizer.decode(match)
                    # print(f"Decoded Tokens: {decoded_tokens}")
                    # decoded tokens should be a number
                    try:
                        num_reasoning_steps = int(decoded_tokens)
                        num_reasoning_steps = 2
                        do_reasoning = True
                        # print(f"Number of reasoning steps: {num_reasoning_steps}")
                    except ValueError:
                        pass
                        # print(f"Decoded tokens are not a number: {decoded_tokens}")
                else:
                    pass

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

        # dumb but do an extra forward pass with a " " token to leave a gap
        # input_ids = torch.tensor([220], dtype=input_ids.dtype, device=input_ids.device)
        # outputs = self.normal_forward(input_ids, 
        #     positions, 
        #     intermediate_tensors, 
        #     inputs_embeds)
        # num_reasoning_steps += 1

        if all_token_ids is not None:
            return outputs, num_reasoning_steps
        else:
            return outputs


class Qwen2ForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
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
        self.model = Qwen2Model(vllm_config=vllm_config,
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
