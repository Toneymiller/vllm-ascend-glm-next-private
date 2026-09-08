"""Regression tests for GLM-5-Next FlashComm1 layer-0 attention output layout.

Under FlashComm1 the row-parallel ``o_proj`` reduce-scatters along the token
dimension, so every attention layer emits the local token shard. Layer 0
still receives the full embedding sequence, so the decoder must allocate a
N/tp output buffer there (Qwen3.5 pattern) and chunk the residual before the
add. The token all-gather for the following MLP/MoE compute is handled by the
SP-aware linear layers, not by the attention layer.
"""

import torch
from unittest.mock import patch

from vllm_ascend.models.glm5_next import AscendGlm5NextDecoderLayer

_TP_SIZE = 8
_FULL_TOKENS = 256
_LOCAL_TOKENS = _FULL_TOKENS // _TP_SIZE
_HIDDEN = 4096


def _make_layer(*, is_vl_first_layer: bool, layer_idx: int = 0) -> AscendGlm5NextDecoderLayer:
    layer = AscendGlm5NextDecoderLayer.__new__(AscendGlm5NextDecoderLayer)
    layer.is_vl_first_layer = is_vl_first_layer
    layer.layer_idx = layer_idx
    return layer


def _allocate(layer, tokens, *, flash_comm_v1_enabled):
    with (
        patch("vllm_ascend.models.glm5_next._EXTRA_CTX") as mock_ctx,
        patch(
            "vllm_ascend.models.glm5_next.get_tensor_model_parallel_world_size",
            return_value=_TP_SIZE,
        ),
    ):
        mock_ctx.flash_comm_v1_enabled = flash_comm_v1_enabled
        return layer._make_attention_output(torch.zeros((tokens, _HIDDEN)))


def test_vl_first_layer_flashcomm_attention_output_is_sharded():
    layer = _make_layer(is_vl_first_layer=True)

    output = _allocate(layer, _FULL_TOKENS, flash_comm_v1_enabled=True)

    assert output.shape == (_LOCAL_TOKENS, _HIDDEN)


def test_vl_first_layer_without_flashcomm_keeps_full_output():
    layer = _make_layer(is_vl_first_layer=True)

    output = _allocate(layer, _FULL_TOKENS, flash_comm_v1_enabled=False)

    assert output.shape == (_FULL_TOKENS, _HIDDEN)


def test_non_first_layer_keeps_input_token_count():
    layer = _make_layer(is_vl_first_layer=False, layer_idx=1)

    output = _allocate(layer, _LOCAL_TOKENS, flash_comm_v1_enabled=True)

    assert output.shape == (_LOCAL_TOKENS, _HIDDEN)