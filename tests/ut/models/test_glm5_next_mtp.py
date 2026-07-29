# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import get_args
from unittest.mock import MagicMock, patch

import torch
import vllm.config.speculative as speculative_config

from vllm_ascend.models.glm5_next import (
    AscendGlm5NextForCausalLM,
    _pad_nope_kv_a_weight,
)
from vllm_ascend.models.glm5_next_mtp import (
    AscendGlm5NextMTP,
    AscendGlm5NextMultiTokenPredictor,
    AscendGlm5NextMultiTokenPredictorLayer,
    _get_spec_layer_idx,
)
from vllm_ascend.patch.platform.patch_speculative_config import (
    hf_config_override,
)
from vllm_ascend.transformers_utils.configs.glm5_next import Glm5NextTextConfig


def test_get_spec_layer_idx_accepts_checkpoint_prefixes():
    config = SimpleNamespace(
        num_hidden_layers=45,
        num_nextn_predict_layers=2,
    )

    assert _get_spec_layer_idx(config, "model.layers.45.enorm.weight") == 45
    assert _get_spec_layer_idx(config, "layers.46.self_attn.q_a_proj.weight") == 46
    assert _get_spec_layer_idx(config, "model.layers.44.mlp.weight") is None


def test_mtp_rewrites_layer_and_shared_weight_names():
    mtp = object.__new__(AscendGlm5NextMTP)

    assert (
        mtp._rewrite_spec_layer_name(
            45,
            "model.layers.45.self_attn.q_a_proj.weight",
        )
        == "model.layers.45.mtp_block.self_attn.q_a_proj.weight"
    )
    assert (
        mtp._rewrite_spec_layer_name(
            45,
            "model.layers.45.shared_head.norm.weight",
        )
        == "model.layers.45.shared_head.norm.weight"
    )
    assert (
        mtp._rewrite_spec_layer_name(
            45,
            "layers.45.embed_tokens.weight",
        )
        == "model.embed_tokens.weight"
    )


def test_mtp_compute_logits_delegates_to_predictor():
    mtp = object.__new__(AscendGlm5NextMTP)
    torch.nn.Module.__init__(mtp)
    expected = torch.randn(2, 3)
    predictor = MagicMock()
    predictor.compute_logits.return_value = expected
    mtp.model = predictor
    hidden_states = torch.randn(2, 4)

    result = mtp.compute_logits(hidden_states, spec_step_idx=1)

    assert result is expected
    predictor.compute_logits.assert_called_once_with(hidden_states, 1)


def test_single_mtp_layer_is_reused_for_all_five_draft_steps():
    class RecordingLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.step_indices = []

        def forward(
            self,
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            current_step_idx,
        ):
            del input_ids, positions, inputs_embeds
            self.step_indices.append(current_step_idx)
            return previous_hidden_states, previous_hidden_states

    predictor = object.__new__(AscendGlm5NextMultiTokenPredictor)
    torch.nn.Module.__init__(predictor)
    predictor.mtp_start_layer_idx = 45
    predictor.num_mtp_layers = 1
    layer = RecordingLayer()
    predictor.layers = torch.nn.ModuleDict({"45": layer})
    hidden_states = torch.ones((1, 4))

    for spec_step_idx in range(5):
        output, _ = predictor(
            input_ids=torch.ones(1, dtype=torch.int64),
            positions=torch.tensor([spec_step_idx]),
            previous_hidden_states=hidden_states,
            inputs_embeds=torch.zeros_like(hidden_states),
            spec_step_idx=spec_step_idx,
        )
        assert output is hidden_states

    assert layer.step_indices == [0, 0, 0, 0, 0]


def test_mtp_zeroes_position_zero_embedding_before_normalization():
    class PassthroughBlock(torch.nn.Module):
        def forward(self, *, positions, hidden_states, residual):
            del positions, residual
            return hidden_states, None

    layer = object.__new__(AscendGlm5NextMultiTokenPredictorLayer)
    torch.nn.Module.__init__(layer)
    layer.enorm = torch.nn.Identity()
    layer.hnorm = torch.nn.Identity()
    layer.eh_proj = torch.nn.Identity()
    layer.mtp_block = PassthroughBlock()
    layer.shared_head = torch.nn.Identity()

    inputs_embeds = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    previous_hidden_states = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    with patch(
        "vllm_ascend.models.glm5_next_mtp.tensor_model_parallel_all_reduce",
        side_effect=lambda hidden_states: hidden_states,
    ):
        hidden_states, recycled_hidden_states = layer(
            input_ids=torch.tensor([11, 12]),
            positions=torch.tensor([0, 3]),
            previous_hidden_states=previous_hidden_states,
            inputs_embeds=inputs_embeds,
        )

    expected = torch.tensor(
        [
            [0.0, 0.0, 5.0, 6.0],
            [3.0, 4.0, 7.0, 8.0],
        ]
    )
    assert torch.equal(hidden_states, expected)
    assert recycled_hidden_states is hidden_states


def test_nope_kv_projection_is_padded_with_zero_rope_rows():
    config = SimpleNamespace(
        mla_nope=True,
        kv_lora_rank=4,
        qk_rope_head_dim=2,
    )
    weight = torch.ones(4, 3)

    padded = _pad_nope_kv_a_weight(
        config,
        "model.layers.45.self_attn.kv_a_proj_with_mqa.weight",
        weight,
    )

    assert padded.shape == (6, 3)
    assert torch.equal(padded[:4], weight)
    assert torch.count_nonzero(padded[4:]) == 0


def test_glm5_mamba_config_shape_reserves_five_speculative_conv_slots():
    config = Glm5NextTextConfig()
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=16),
        model_config=SimpleNamespace(hf_config=config),
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
    )

    conv_shape, recurrent_shape = AscendGlm5NextForCausalLM.get_mamba_state_shape_from_config(vllm_config)

    assert sorted(conv_shape) == [8, 1536]
    assert recurrent_shape == (4, 128, 128)
    conv_bytes = torch.empty(conv_shape, dtype=torch.bfloat16).numel() * 2
    recurrent_bytes = torch.empty(recurrent_shape, dtype=torch.float32).numel() * 4
    assert conv_bytes + recurrent_bytes == 286720


def test_glm5_speculative_config_selects_mtp_architecture():
    config = Glm5NextTextConfig(
        architectures=["Glm5NextForCausalLM"],
        num_nextn_predict_layers=2,
    )

    result = hf_config_override(config)

    assert result.model_type == "glm5_next_mtp"
    assert result.n_predict == 2
    assert result.architectures == ["Glm5NextMTPModel"]


def test_glm5_mtp_model_type_is_detected_by_vllm_v023():
    assert "glm5_next_mtp" in get_args(speculative_config.MTPModelTypes)
