from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import vllm_ascend.ops.indexer_kpool_mla as indexer_kpool_mla_ops
from vllm_ascend.ops.indexer_kpool_mla import AscendIndexerKPoolMLAAttention


@pytest.mark.parametrize("flash_comm_v1_enabled", [False, True])
def test_indexer_kpool_mla_forwards_flashcomm_gather_flag(
    flash_comm_v1_enabled,
):
    wrapper = object.__new__(AscendIndexerKPoolMLAAttention)
    torch.nn.Module.__init__(wrapper)
    wrapper.hidden_size = 8
    wrapper.prefix = "model.layers.3.self_attn"
    hidden_states = torch.randn(4, wrapper.hidden_size)

    with (
        patch.object(
            indexer_kpool_mla_ops,
            "_EXTRA_CTX",
            SimpleNamespace(
                flash_comm_v1_enabled=flash_comm_v1_enabled,
            ),
        ),
        patch(
            "vllm_ascend.ops.indexer_kpool_mla."
            "torch.ops.vllm.indexer_kpool_mla_forward"
        ) as mock_forward,
    ):
        output = wrapper.forward(torch.arange(4), hidden_states)

    mock_forward.assert_called_once_with(
        hidden_states,
        flash_comm_v1_enabled,
        output,
        wrapper.prefix,
    )
    assert output.shape == hidden_states.shape
