import torch

from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm_ascend.ops.kimi_kda import (
    _restore_padded_conv_state,
    _stage_padded_conv_state,
)


def test_contiguous_conv_state_is_not_staged():
    state = torch.arange(24).reshape(4, 2, 3)
    indices = torch.tensor([[3], [1]], dtype=torch.int32)

    staged, kernel_indices, restore_metadata = _stage_padded_conv_state(
        state, indices
    )

    assert staged is state
    assert kernel_indices is indices
    assert restore_metadata is None


def test_padded_conv_state_is_gathered_and_restored_by_cache_index():
    storage = torch.full((40,), -1, dtype=torch.int64)
    state = torch.as_strided(storage, (4, 2, 3), (10, 3, 1))
    for block_id in range(state.shape[0]):
        state[block_id].fill_(block_id)
    indices = torch.tensor([[2], [0], [PAD_SLOT_ID]], dtype=torch.int32)

    staged, kernel_indices, restore_metadata = _stage_padded_conv_state(
        state, indices
    )

    assert staged.is_contiguous()
    torch.testing.assert_close(staged[0], torch.full((2, 3), 2))
    torch.testing.assert_close(staged[1], torch.full((2, 3), 0))
    assert kernel_indices.tolist() == [[0], [1], [PAD_SLOT_ID]]

    staged[0].fill_(20)
    staged[1].fill_(10)
    staged[2].fill_(99)
    _restore_padded_conv_state(state, staged, restore_metadata)

    torch.testing.assert_close(state[0], torch.full((2, 3), 10))
    torch.testing.assert_close(state[1], torch.full((2, 3), 1))
    torch.testing.assert_close(state[2], torch.full((2, 3), 20))
    torch.testing.assert_close(state[3], torch.full((2, 3), 3))
