import torch

from vllm_ascend.ops.kimi_kda import _gather_state_rows, _restore_state_rows


def test_contiguous_state_gathers_and_restores_selected_rows():
    state = torch.arange(4 * 2 * 2 * 3).reshape(4, 2, 2, 3)
    original = state.clone()
    indices = torch.tensor([3, 1], dtype=torch.int32)

    staged, restore_metadata = _gather_state_rows(state, indices, state)
    staged.add_(1000)
    _restore_state_rows(staged, restore_metadata)

    torch.testing.assert_close(state[0], original[0])
    torch.testing.assert_close(state[1], original[1] + 1000)
    torch.testing.assert_close(state[2], original[2])
    torch.testing.assert_close(state[3], original[3] + 1000)


def test_padded_state_only_gathers_and_restores_selected_physical_pages():
    num_pages = 4
    page_stride = 40
    allocation_offset = 7
    recurrent_offset = 10
    storage = torch.arange(
        allocation_offset + num_pages * page_stride + 5,
        dtype=torch.int64,
    )
    page_base = torch.as_strided(
        storage,
        (num_pages, 2, 3),
        (page_stride, 3, 1),
        storage_offset=allocation_offset,
    )
    state = torch.as_strided(
        storage,
        (num_pages, 2, 2, 3),
        (page_stride, 6, 3, 1),
        storage_offset=allocation_offset + recurrent_offset,
    )
    original_storage = storage.clone()
    indices = torch.tensor([2, 0], dtype=torch.int32)

    staged, restore_metadata = _gather_state_rows(state, indices, page_base)
    torch.testing.assert_close(staged[0], state[2])
    torch.testing.assert_close(staged[1], state[0])
    staged[0].fill_(200)
    staged[1].fill_(100)
    _restore_state_rows(staged, restore_metadata)

    torch.testing.assert_close(state[0], torch.full_like(state[0], 100))
    torch.testing.assert_close(state[2], torch.full_like(state[2], 200))
    torch.testing.assert_close(state[1], original_storage.as_strided(state.shape, state.stride(),
                                                                      state.storage_offset())[1])
    torch.testing.assert_close(state[3], original_storage.as_strided(state.shape, state.stride(),
                                                                      state.storage_offset())[3])

    for page in (0, 2):
        page_start = allocation_offset + page * page_stride
        torch.testing.assert_close(
            storage[page_start:page_start + recurrent_offset],
            original_storage[page_start:page_start + recurrent_offset],
        )
        payload_end = page_start + recurrent_offset + state[0].numel()
        torch.testing.assert_close(
            storage[payload_end:page_start + page_stride],
            original_storage[payload_end:page_start + page_stride],
        )


def test_padded_state_repeated_index_uses_last_update_and_preserves_padding():
    num_pages = 3
    page_stride = 32
    state_offset = 5
    storage = torch.arange(num_pages * page_stride, dtype=torch.int64)
    page_base = torch.as_strided(
        storage,
        (num_pages, 2, 2),
        (page_stride, 2, 1),
    )
    state = torch.as_strided(
        storage,
        (num_pages, 1, 2, 3),
        (page_stride, 6, 3, 1),
        storage_offset=state_offset,
    )
    original_storage = storage.clone()
    indices = torch.tensor([0, 2, 0], dtype=torch.int32)

    staged, restore_metadata = _gather_state_rows(state, indices, page_base)
    staged[0].fill_(10)
    staged[1].fill_(20)
    staged[2].fill_(30)
    _restore_state_rows(staged, restore_metadata)

    torch.testing.assert_close(state[0], torch.full_like(state[0], 30))
    torch.testing.assert_close(state[2], torch.full_like(state[2], 20))
    for page in (0, 2):
        page_start = page * page_stride
        torch.testing.assert_close(
            storage[page_start:page_start + state_offset],
            original_storage[page_start:page_start + state_offset],
        )
        payload_end = page_start + state_offset + state[0].numel()
        torch.testing.assert_close(
            storage[payload_end:page_start + page_stride],
            original_storage[payload_end:page_start + page_stride],
        )


def test_padded_state_uses_byte_stride_for_mixed_cache_dtypes():
    num_pages = 3
    page_size_bytes = 64
    allocation_offset_bytes = 16
    recurrent_offset_bytes = 16
    storage = torch.arange(
        allocation_offset_bytes + num_pages * page_size_bytes,
        dtype=torch.uint8,
    )
    conv_storage = storage.view(torch.bfloat16)
    recurrent_storage = storage.view(torch.float32)
    page_base = torch.as_strided(
        conv_storage,
        (num_pages, 2, 4),
        (page_size_bytes // 2, 4, 1),
        storage_offset=allocation_offset_bytes // 2,
    )
    state = torch.as_strided(
        recurrent_storage,
        (num_pages, 1, 2, 2),
        (page_size_bytes // 4, 4, 2, 1),
        storage_offset=(allocation_offset_bytes + recurrent_offset_bytes) // 4,
    )
    original_storage = storage.clone()
    indices = torch.tensor([2, 0], dtype=torch.int32)

    staged, restore_metadata = _gather_state_rows(state, indices, page_base)
    torch.testing.assert_close(staged[0], state[2])
    torch.testing.assert_close(staged[1], state[0])
    staged[0].fill_(20)
    staged[1].fill_(10)
    _restore_state_rows(staged, restore_metadata)

    torch.testing.assert_close(state[0], torch.full_like(state[0], 10))
    torch.testing.assert_close(state[2], torch.full_like(state[2], 20))
    for page in (0, 2):
        page_start = allocation_offset_bytes + page * page_size_bytes
        torch.testing.assert_close(
            storage[page_start:page_start + recurrent_offset_bytes],
            original_storage[page_start:page_start + recurrent_offset_bytes],
        )
        payload_end = page_start + recurrent_offset_bytes + state[0].numel() * state.element_size()
        torch.testing.assert_close(
            storage[payload_end:page_start + page_size_bytes],
            original_storage[payload_end:page_start + page_size_bytes],
        )
