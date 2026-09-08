# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.models.glm5_next import AscendGlm5NextModel
from vllm_ascend.models.glm5_next_mtp import AscendGlm5NextMTP


EXPERT_MAPPING = [
    ("experts.w2_weight", "experts.0.down_proj.weight", 0, "w2"),
]


def _load_expert_weight(checkpoint_name, params):
    model = SimpleNamespace(
        config=SimpleNamespace(
            n_routed_experts=1,
            num_hidden_layers=45,
            num_nextn_predict_layers=1,
        ),
        named_parameters=lambda: params.items(),
    )
    loaded_weight = torch.randn(4, 4)
    with (
        patch("vllm_ascend.models.glm5_next._is_moe", return_value=True),
        patch(
            "vllm_ascend.models.glm5_next.fused_moe_make_expert_params_mapping",
            return_value=EXPERT_MAPPING,
        ),
        patch(
            "vllm_ascend.models.glm5_next._pad_nope_kv_a_weight",
            side_effect=lambda config, name, weight: weight,
        ),
        patch(
            "vllm_ascend.models.glm5_next.is_pp_missing_parameter",
            return_value=False,
        ),
    ):
        loaded_params = AscendGlm5NextModel.load_weights(
            model,
            [(checkpoint_name, loaded_weight)],
        )
    return loaded_params, loaded_weight


def test_main_model_skips_configured_mtp_expert_weight():
    loaded_params, _ = _load_expert_weight(
        "layers.45.mlp.experts.0.down_proj.weight",
        {},
    )

    assert loaded_params == set()


def test_main_model_loads_regular_expert_weight():
    weight_loader = MagicMock()
    param = SimpleNamespace(weight_loader=weight_loader)
    param_name = "layers.44.mlp.experts.w2_weight"

    loaded_params, loaded_weight = _load_expert_weight(
        "layers.44.mlp.experts.0.down_proj.weight",
        {param_name: param},
    )

    assert loaded_params == {param_name}
    weight_loader.assert_called_once_with(
        param,
        loaded_weight,
        param_name,
        expert_id=0,
        shard_id="w2",
    )


def test_main_model_rejects_missing_regular_expert_target():
    with pytest.raises(KeyError, match=r"layers\.44\.mlp\.experts\.w2_weight"):
        _load_expert_weight(
            "layers.44.mlp.experts.0.down_proj.weight",
            {},
        )


def test_mtp_model_rejects_missing_stacked_target():
    model = SimpleNamespace(
        config=SimpleNamespace(
            n_routed_experts=None,
            num_hidden_layers=45,
            num_nextn_predict_layers=1,
        ),
        named_parameters=lambda: iter(()),
        _rewrite_spec_layer_name=lambda layer, name: (
            AscendGlm5NextMTP._rewrite_spec_layer_name(model, layer, name)
        ),
    )

    with (
        patch(
            "vllm_ascend.models.glm5_next_mtp._pad_nope_kv_a_weight",
            side_effect=lambda config, name, weight: weight,
        ),
        pytest.raises(
            KeyError,
            match=(
                r"model\.layers\.45\.mtp_block\.self_attn\."
                r"fused_qkv_a_proj\.weight"
            ),
        ),
    ):
        AscendGlm5NextMTP.load_weights(
            model,
            [("model.layers.45.self_attn.q_a_proj.weight", torch.randn(4, 4))],
        )
