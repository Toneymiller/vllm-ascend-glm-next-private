# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.models.glm5_next import AscendGlm5NextModel


def _load_weights(model, weights):
    with (
        patch("vllm_ascend.models.glm5_next._is_moe", return_value=False),
        patch(
            "vllm_ascend.models.glm5_next._pad_nope_kv_a_weight",
            side_effect=lambda config, name, weight: weight,
        ),
        patch(
            "vllm_ascend.models.glm5_next.is_pp_missing_parameter",
            return_value=False,
        ),
    ):
        return AscendGlm5NextModel.load_weights(model, weights)


@pytest.mark.parametrize("projection", ["q", "k", "v"])
def test_separate_conv1d_weight_is_not_split_again(projection):
    name = f"layers.0.self_attn.{projection}_conv1d.weight"
    weight_loader = MagicMock()
    param = SimpleNamespace(weight_loader=weight_loader)
    model = SimpleNamespace(
        config=SimpleNamespace(),
        named_parameters=lambda: [(name, param)],
    )
    loaded_weight = torch.randn(8192, 1, 4)

    loaded_params = _load_weights(model, [(name, loaded_weight)])

    assert loaded_params == {name}
    weight_loader.assert_called_once_with(param, loaded_weight)


def test_fused_conv1d_weight_is_split_for_legacy_checkpoints():
    prefix = "layers.0.self_attn"
    params = {}
    loaders = {}
    for projection in ("q", "k", "v"):
        name = f"{prefix}.{projection}_conv1d.weight"
        loaders[projection] = MagicMock()
        params[name] = SimpleNamespace(weight_loader=loaders[projection])
    model = SimpleNamespace(
        config=SimpleNamespace(),
        named_parameters=lambda: params.items(),
    )
    loaded_weight = torch.arange(24 * 4).reshape(24, 1, 4)

    loaded_params = _load_weights(
        model,
        [(f"{prefix}.conv1d.weight", loaded_weight)],
    )

    expected_weights = loaded_weight.squeeze(1).chunk(3, dim=0)
    assert loaded_params == set(params)
    for projection, expected_weight in zip(
        ("q", "k", "v"),
        expected_weights,
        strict=True,
    ):
        loader = loaders[projection]
        loader.assert_called_once()
        called_param, called_weight = loader.call_args.args
        assert called_param is params[f"{prefix}.{projection}_conv1d.weight"]
        torch.testing.assert_close(called_weight, expected_weight.unsqueeze(1))
