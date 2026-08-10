# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.deepseek_mtp import SharedHead
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MixtureOfExperts
from vllm.model_executor.models.utils import maybe_prefix
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from vllm_ascend.models.glm5_next import (
    AscendGlm5NextDecoderLayer,
    AscendGlm5NextMoE,
    _pad_nope_kv_a_weight,
    get_spec_layer_idx_from_weight_name,
)


class AscendGlm5NextMultiTokenPredictorLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=False,
        )

        topk_tokens = config.index_topk
        if topk_tokens is None:
            raise ValueError("GLM-5 MTP requires index_topk.")
        index_kpool = getattr(config, "index_kpool", 1) or 1
        topk_width = topk_tokens + index_kpool - 1
        sparse_block_size = 128
        topk_width = ((topk_width + sparse_block_size - 1) // sparse_block_size) * sparse_block_size
        topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            topk_width,
            dtype=torch.int32,
            device=current_platform.device_type,
        )

        self.shared_head = SharedHead(
            config=config,
            prefix=prefix,
            quant_config=quant_config,
        )
        layer_idx = int(prefix.rsplit(".", 1)[-1])
        self.mtp_block = AscendGlm5NextDecoderLayer(
            vllm_config=vllm_config,
            config=config,
            layer_idx=layer_idx,
            prefix=prefix,
            topk_indices_buffer=topk_indices_buffer,
            is_mtp_layer=True,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del input_ids, spec_step_index
        if inputs_embeds is None:
            raise ValueError("GLM-5 MTP requires input embeddings.")
        # The first token in each sequence has no "next token" embedding.
        # Match the upstream fused_eh_norm semantics without introducing a
        # device-to-host synchronization in the decode hot path.
        inputs_embeds = inputs_embeds.masked_fill(
            positions.eq(0).unsqueeze(-1),
            0,
        )
        embeddings = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)
        hidden_states = self.eh_proj(torch.cat([embeddings, previous_hidden_states], dim=-1))
        hidden_states, _ = self.mtp_block(
            positions=positions,
            hidden_states=hidden_states,
            residual=None,
        )
        hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        hidden_states = self.shared_head(hidden_states)
        return hidden_states, hidden_states


class AscendGlm5NextMultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        self.layers = torch.nn.ModuleDict(
            {
                str(layer_idx): AscendGlm5NextMultiTokenPredictorLayer(
                    vllm_config,
                    f"{prefix}.layers.{layer_idx}",
                )
                for layer_idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        current_step_idx = spec_step_idx % self.num_mtp_layers
        layer_idx = self.mtp_start_layer_idx + current_step_idx
        return self.layers[str(layer_idx)](
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            current_step_idx,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        current_step_idx = spec_step_idx % self.num_mtp_layers
        layer_idx = self.mtp_start_layer_idx + current_step_idx
        mtp_layer = self.layers[str(layer_idx)]
        return self.logits_processor(mtp_layer.shared_head.head, hidden_states)


class AscendGlm5NextMTP(nn.Module, DeepseekV2MixtureOfExperts):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = AscendGlm5NextMultiTokenPredictor(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self._set_moe_parameters()

    def _set_moe_parameters(self) -> None:
        self.num_moe_layers = self.config.num_nextn_predict_layers
        self.num_expert_groups = getattr(self.config, "n_group", 0)
        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        for layer in self.model.layers.values():
            mlp = layer.mtp_block.mlp
            if isinstance(mlp, AscendGlm5NextMoE):
                example_moe = mlp
                self.moe_mlp_layers.append(mlp)
                self.moe_layers.append(mlp.experts)
        self.extract_moe_parameters(example_moe)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del intermediate_tensors
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided.")
        return self.model(
            input_ids,
            positions,
            hidden_states,
            inputs_embeds,
            spec_step_idx,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.model.compute_logits(hidden_states, spec_step_idx)

    def _rewrite_spec_layer_name(self, spec_layer: int, name: str) -> str:
        if name.startswith("layers."):
            name = f"model.{name}"
        layer_prefix = f"model.layers.{spec_layer}."
        if ".embed_tokens." in name:
            return name.replace(layer_prefix, "model.")
        if any(component in name for component in ("enorm", "hnorm", "eh_proj", "shared_head")):
            return name
        return name.replace(layer_prefix, f"{layer_prefix}mtp_block.")

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            ("fused_qkv_a_proj", "q_a_proj", 0),
            ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1),
            ("wk_weights_proj", "wk", 0),
            ("wk_weights_proj", "weights_proj", 1),
        ]
        if self.config.n_routed_experts is not None:
            expert_params_mapping = fused_moe_make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=self.config.n_routed_experts,
            )
        else:
            expert_params_mapping = []
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for weight in weights:
            name, loaded_weight = weight[:2]
            loader_kwargs = weight[2] if len(weight) > 2 else {}
            if "rotary_emb.inv_freq" in name or "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue
            if name.startswith("model.language_model."):
                name = name.replace("model.language_model.", "model.", 1)
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is None:
                continue
            name = self._rewrite_spec_layer_name(spec_layer, name)
            loaded_weight = _pad_nope_kv_a_weight(
                self.config,
                name,
                loaded_weight,
            )

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts." in name and name not in params_dict:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                param = params_dict[mapped_name]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped_name)
                break
            else:
                for (
                    param_name,
                    weight_name,
                    expert_id,
                    expert_shard_id,
                ) in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    mapped_name = name.replace(weight_name, param_name)
                    param = params_dict[mapped_name]
                    param.weight_loader(
                        param,
                        loaded_weight,
                        mapped_name,
                        expert_id=expert_id,
                        shard_id=expert_shard_id,
                    )
                    loaded_params.add(mapped_name)
                    break
                else:
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    mapped_name = maybe_remap_kv_scale_name(name, params_dict)
                    if mapped_name is None:
                        continue
                    param = params_dict[mapped_name]
                    weight_loader = getattr(
                        param,
                        "weight_loader",
                        default_weight_loader,
                    )
                    weight_loader(param, loaded_weight, **loader_kwargs)
                    loaded_params.add(mapped_name)

        loaded_layers = {
            layer_idx
            for name in loaded_params
            if (
                layer_idx := get_spec_layer_idx_from_weight_name(
                    self.config,
                    name,
                )
            )
            is not None
        }
        expected_layers = set(
            range(
                self.model.mtp_start_layer_idx,
                self.model.mtp_start_layer_idx + self.model.num_mtp_layers,
            )
        )
        missing_layers = expected_layers - loaded_layers
        if missing_layers:
            raise ValueError(f"MTP speculative decoding weights are missing for layers {sorted(missing_layers)}.")
        return loaded_params
