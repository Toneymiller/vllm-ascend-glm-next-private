# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm_ascend.models.glm5_next import _expand_mhc_residual_streams


def test_expand_mhc_residual_streams_replicates_complete_hidden_vectors():
    hidden_states = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )

    expanded = _expand_mhc_residual_streams(hidden_states, num_streams=2)
    streams = expanded.view(2, 2, 3)

    assert expanded.shape == (2, 6)
    assert expanded.is_contiguous()
    assert torch.equal(streams[:, 0], hidden_states)
    assert torch.equal(streams[:, 1], hidden_states)
