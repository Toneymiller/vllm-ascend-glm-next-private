# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_ascend.models import glm5_next_multimodal as glm5_vit
from vllm_ascend.models.glm5_next_multimodal import (
    AscendGlm5NextVisionBlock,
    AscendGlm5NextVisionMLP,
    AscendGlm5NextVisionPatchMerger,
    AscendGlm5NextVisionTransformer,
    Glm5NextSiluAndMul,
    Glm5NextVisionRMSNorm,
)
from vllm_ascend.transformers_utils.configs.glm5_next import (
    Glm5NextVisionConfig,
)


class _FakeLinear(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int | None = None,
        output_sizes: list[int] | None = None,
        bias: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        del kwargs
        if output_sizes is not None:
            output_size = sum(output_sizes)
        assert output_size is not None
        self.linear = nn.Linear(input_size, output_size, bias=bias)

    def forward(self, x: torch.Tensor):
        return self.linear(x), None


class _FakePatchEmbed(nn.Module):
    def __init__(self, hidden_size: int, **kwargs) -> None:
        super().__init__()
        del kwargs
        self.proj = nn.Linear(1, hidden_size)


class _FakeVisionBlock(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.init_kwargs = kwargs


class _FakePatchMerger(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.init_kwargs = kwargs


class _FakeConv2d(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.init_kwargs = kwargs


class _FakeAttention(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        del kwargs

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        return 2 * x


class _FakeMLP(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        del kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 3 * x


def _make_fake_vision_weight_module():
    tower = AscendGlm5NextVisionTransformer.__new__(
        AscendGlm5NextVisionTransformer
    )
    nn.Module.__init__(tower)

    tower.patch_embed = nn.Module()
    tower.patch_embed.register_parameter(
        "weight",
        nn.Parameter(torch.zeros(2, 2)),
    )

    tower.mlp = nn.Module()
    tower.mlp.gate_up_proj = nn.Module()
    gate_up_weight = nn.Parameter(torch.zeros(4, 2))

    def load_gate_up_shard(param, loaded_weight, shard_id):
        param.data.chunk(2, dim=0)[shard_id].copy_(loaded_weight)

    gate_up_weight.weight_loader = load_gate_up_shard
    tower.mlp.gate_up_proj.register_parameter("weight", gate_up_weight)
    return tower


def test_glm5_next_swiglu_matches_transformers_reference():
    gate_up = torch.tensor(
        [[-12.0, -2.0, 4.0, 15.0, -20.0, -3.0, 6.0, 30.0]],
        dtype=torch.float32,
    )
    limit = 10.0

    gate, up = gate_up.chunk(2, dim=-1)
    expected = F.silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)

    actual = Glm5NextSiluAndMul(limit)(gate_up)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_glm5_next_swiglu_preserves_shape_and_dtype():
    gate_up = torch.randn(2, 3, 16, dtype=torch.bfloat16)

    actual = Glm5NextSiluAndMul(10.0)(gate_up)

    assert actual.shape == (2, 3, 8)
    assert actual.dtype == gate_up.dtype


def test_glm5_next_vision_rms_norm_matches_transformers_reference():
    hidden_states = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    norm = Glm5NextVisionRMSNorm(8, eps=1e-5)
    norm.weight.data.copy_(torch.linspace(0.5, 1.5, 8))

    hidden_states_fp32 = hidden_states.float()
    variance = hidden_states_fp32.pow(2).mean(-1, keepdim=True)
    expected = norm.weight * (
        hidden_states_fp32 * torch.rsqrt(variance + 1e-5)
    ).to(hidden_states.dtype)

    actual = norm(hidden_states)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert dict(norm.named_parameters()).keys() == {"weight"}
    assert not hasattr(norm, "bias")


def test_glm5_next_vision_config_preserves_checkpoint_semantics():
    config = Glm5NextVisionConfig(rms_norm_eps=1e-5, swiglu_limit=10.0)

    assert config.rms_norm_eps == 1e-5
    assert config.swiglu_limit == 10.0


def test_glm5_next_vision_mlp_matches_transformers_operation_order():
    with (
        patch.object(glm5_vit, "is_vit_use_data_parallel", return_value=False),
        patch.object(glm5_vit, "MergedColumnParallelLinear", _FakeLinear),
        patch.object(glm5_vit, "RowParallelLinear", _FakeLinear),
    ):
        mlp = AscendGlm5NextVisionMLP(
            in_features=4,
            hidden_features=6,
            swiglu_limit=2.0,
            bias=True,
        )

    x = torch.tensor([[4.0, -3.0, 2.0, -1.0]])
    gate_up, _ = mlp.gate_up_proj(x)
    gate, up = gate_up.chunk(2, dim=-1)
    activated = F.silu(gate.clamp(max=2.0)) * up.clamp(min=-2.0, max=2.0)
    expected, _ = mlp.down_proj(activated)

    torch.testing.assert_close(mlp(x), expected)


def test_glm5_next_patch_merger_matches_transformers_operation_order():
    with (
        patch.object(glm5_vit, "is_vit_use_data_parallel", return_value=False),
        patch.object(glm5_vit, "ColumnParallelLinear", _FakeLinear),
        patch.object(glm5_vit, "MergedColumnParallelLinear", _FakeLinear),
        patch.object(glm5_vit, "RowParallelLinear", _FakeLinear),
    ):
        merger = AscendGlm5NextVisionPatchMerger(
            d_model=4,
            context_dim=6,
            swiglu_limit=2.0,
            bias=False,
        )

    x = torch.tensor([[4.0, -3.0, 2.0, -1.0]])
    projected, _ = merger.proj(x)
    projected = F.gelu(merger.post_projection_norm(projected))
    gate_up, _ = merger.gate_up_proj(projected)
    gate, up = gate_up.chunk(2, dim=-1)
    activated = F.silu(gate.clamp(max=2.0)) * up.clamp(min=-2.0, max=2.0)
    expected, _ = merger.down_proj(activated)

    torch.testing.assert_close(merger(x), expected)


def test_glm5_next_vision_block_matches_transformers_residual_order():
    with (
        patch.object(
            glm5_vit,
            "Glm5NextVisionRMSNorm",
            lambda *args, **kwargs: nn.Identity(),
        ),
        patch.object(
            glm5_vit,
            "AscendGlm5NextVisionAttention",
            _FakeAttention,
        ),
        patch.object(glm5_vit, "AscendGlm5NextVisionMLP", _FakeMLP),
    ):
        block = AscendGlm5NextVisionBlock(
            dim=4,
            num_heads=2,
            mlp_hidden_dim=8,
            norm_eps=1e-5,
            swiglu_limit=10.0,
        )

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    actual = block(
        x,
        cu_seqlens=torch.tensor([0, 1], dtype=torch.int32),
        rotary_pos_emb_cos=torch.ones(1, 2),
        rotary_pos_emb_sin=torch.zeros(1, 2),
    )

    # First residual: x + 2x = 3x. Second residual: 3x + 3(3x) = 12x.
    torch.testing.assert_close(actual, 12 * x)


def test_glm5_next_transformer_builds_only_dedicated_vision_components():
    vision_config = SimpleNamespace(
        hidden_size=8,
        num_heads=2,
        patch_size=2,
        temporal_patch_size=2,
        in_channels=3,
        spatial_merge_size=2,
        out_hidden_size=16,
        projection_intermediate_size=24,
        intermediate_size=12,
        depth=2,
        swiglu_limit=10.0,
        attention_bias=True,
    )

    with (
        patch.object(glm5_vit, "AscendGlm5NextVisionPatchEmbed", _FakePatchEmbed),
        patch.object(glm5_vit, "AscendGlm5NextVisionBlock", _FakeVisionBlock),
        patch.object(
            glm5_vit,
            "AscendGlm5NextVisionPatchMerger",
            _FakePatchMerger,
        ),
        patch.object(glm5_vit, "Conv2dLayer", _FakeConv2d),
        patch.object(
            glm5_vit,
            "Glm5NextVisionRMSNorm",
            lambda *args, **kwargs: nn.Identity(),
        ),
        patch.object(glm5_vit, "get_rope", return_value=object()),
        patch.object(glm5_vit, "get_vit_attn_backend", return_value="test"),
    ):
        tower = AscendGlm5NextVisionTransformer(
            text_config=SimpleNamespace(),
            vision_config=vision_config,
            norm_eps=1e-5,
        )

    assert len(tower.blocks) == vision_config.depth
    assert all(isinstance(block, _FakeVisionBlock) for block in tower.blocks)
    assert tower.blocks[0].init_kwargs["mlp_hidden_dim"] == 12
    assert tower.blocks[0].init_kwargs["swiglu_limit"] == 10.0
    assert tower.blocks[0].init_kwargs["bias"] is True
    assert tower.merger.init_kwargs["context_dim"] == 24
    assert tower.merger.init_kwargs["swiglu_limit"] == 10.0
    assert not hasattr(tower, "embeddings")
    assert not hasattr(tower, "post_conv_layernorm")


def test_glm5_next_vision_weight_loader_maps_gate_and_up_shards():
    tower = _make_fake_vision_weight_module()
    weights = [
        ("patch_embed.weight", torch.ones(2, 2)),
        ("mlp.gate_proj.weight", torch.full((2, 2), 2.0)),
        ("mlp.up_proj.weight", torch.full((2, 2), 3.0)),
    ]

    loaded = tower.load_weights(weights)

    assert loaded == {"patch_embed.weight", "mlp.gate_up_proj.weight"}
    torch.testing.assert_close(tower.patch_embed.weight, torch.ones(2, 2))
    gate, up = tower.mlp.gate_up_proj.weight.chunk(2, dim=0)
    torch.testing.assert_close(gate, torch.full((2, 2), 2.0))
    torch.testing.assert_close(up, torch.full((2, 2), 3.0))


def test_glm5_next_vision_weight_loader_keeps_fused_qkv_name():
    assert AscendGlm5NextVisionTransformer._map_weight_name(
        "blocks.0.attn.qkv.weight"
    ) == ("blocks.0.attn.qkv.weight", None)


def test_glm5_next_vision_weight_loader_accepts_partial_shards():
    tower = _make_fake_vision_weight_module()
    first_shard = [
        ("patch_embed.weight", torch.ones(2, 2)),
        ("mlp.gate_proj.weight", torch.full((2, 2), 2.0)),
    ]
    second_shard = [
        ("mlp.up_proj.weight", torch.full((2, 2), 3.0)),
    ]

    first_loaded = tower.load_weights(first_shard)
    second_loaded = tower.load_weights(second_shard)

    assert first_loaded == {"patch_embed.weight", "mlp.gate_up_proj.weight"}
    assert second_loaded == {"mlp.gate_up_proj.weight"}
    torch.testing.assert_close(tower.patch_embed.weight, torch.ones(2, 2))
    gate, up = tower.mlp.gate_up_proj.weight.chunk(2, dim=0)
    torch.testing.assert_close(gate, torch.full((2, 2), 2.0))
    torch.testing.assert_close(up, torch.full((2, 2), 3.0))


def test_glm5_next_vision_weight_loader_rejects_unexpected_weight():
    tower = _make_fake_vision_weight_module()
    weights = [
        ("patch_embed.weight", torch.ones(2, 2)),
        ("mlp.gate_proj.weight", torch.full((2, 2), 2.0)),
        ("mlp.up_proj.weight", torch.full((2, 2), 3.0)),
        ("unexpected.weight", torch.ones(1)),
    ]

    with pytest.raises(ValueError, match="Unexpected GLM5Next vision weight"):
        tower.load_weights(weights)


def test_glm5_next_vision_weight_loader_rejects_duplicate_source():
    tower = _make_fake_vision_weight_module()
    weights = [
        ("patch_embed.weight", torch.ones(2, 2)),
        ("patch_embed.weight", torch.ones(2, 2)),
    ]

    with pytest.raises(ValueError, match="Duplicate GLM5Next vision weight"):
        tower.load_weights(weights)


def test_glm5_next_vision_weight_loader_reports_shape_mismatch():
    tower = _make_fake_vision_weight_module()
    weights = [
        ("patch_embed.weight", torch.ones(3, 2)),
        ("mlp.gate_proj.weight", torch.full((2, 2), 2.0)),
        ("mlp.up_proj.weight", torch.full((2, 2), 3.0)),
    ]

    with pytest.raises(
        ValueError,
        match=r"checkpoint shape \(3, 2\), target shape \(2, 2\)",
    ):
        tower.load_weights(weights)
