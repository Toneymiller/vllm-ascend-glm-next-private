"""Regression tests for excluding vision towers from FlashComm SP dispatch.

Vision encoders precompute inputs_embeds outside the forward context, so
their row/column parallel layers must not be routed to context-dependent SP
ops (upstream #10941). GLM-5-Next names its tower ``visual``; Step3p7 uses
``vision_model``.
"""

from unittest.mock import MagicMock, patch

from vllm_ascend.ops.linear_op import (
    SequenceColumnParallelOp,
    SequenceRowParallelOp,
    _get_column_parallel_op,
    _get_row_parallel_op,
    _is_vision_encoder_prefix,
)


def test_vision_encoder_prefixes_are_detected():
    assert _is_vision_encoder_prefix("visual.encoder.layers.0.attention.out_proj")
    assert _is_vision_encoder_prefix("language_model.vision_model.encoder.layers.0.mlp.down_proj")
    assert not _is_vision_encoder_prefix("language_model.layers.0.self_attn.o_proj")


@patch("vllm_ascend.ops.linear_op.enable_sp", return_value=True)
@patch("vllm_ascend.ops.linear_op.enable_dsa_cp", return_value=False)
@patch("vllm_ascend.ops.linear_op.enable_dsa_cp_with_layer_shard", return_value=False)
@patch("vllm_ascend.ops.linear_op.matmul_allreduce_enable", return_value=False)
@patch("vllm_ascend.ops.linear_op.flashcomm2_enable", return_value=False)
@patch("vllm_ascend.ops.linear_op.mlp_tp_enable", return_value=False)
@patch("vllm_ascend.ops.linear_op.oproj_tp_enable", return_value=False)
def test_vision_encoder_row_parallel_skips_sequence_row_op(*_):
    layer = MagicMock()

    vision_op = _get_row_parallel_op("visual.encoder.layers.0.attention.out_proj", layer)
    assert vision_op is None

    lm_op = _get_row_parallel_op("language_model.layers.0.self_attn.o_proj", layer)
    assert isinstance(lm_op, SequenceRowParallelOp)


@patch("vllm_ascend.ops.linear_op.enable_sp", return_value=True)
@patch("vllm_ascend.ops.linear_op.enable_dsa_cp", return_value=False)
@patch("vllm_ascend.ops.linear_op.mlp_tp_enable", return_value=False)
@patch("vllm_ascend.ops.linear_op.flashcomm2_oshard_manager.flashcomm2_oshard_enable", return_value=False)
def test_vision_encoder_column_parallel_skips_sequence_column_op(*_):
    layer = MagicMock()

    vision_op = _get_column_parallel_op("visual.encoder.layers.0.attention.qkv_proj", layer)
    assert vision_op is None

    lm_op = _get_column_parallel_op("language_model.layers.0.self_attn.qkv_proj", layer)
    assert isinstance(lm_op, SequenceColumnParallelOp)