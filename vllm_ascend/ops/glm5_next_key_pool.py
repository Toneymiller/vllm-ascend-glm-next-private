# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full GLM5 Next KeyPool CPU reference (adapted from the aclnnKeyPool golden).

The op implements the complete KeyPool pipeline used by GLM-5 Next sparse
attention:

  1. Project ``hidden_states`` with ``wk`` (KV weight) and ``gate_weight``
     (gate weight) to obtain per-token ``key``/``gate`` states.
  2. Optionally normalize ``key`` with a LayerNorm-style affine transform
     (``norm_weight``/``norm_bias``, mean-subtracted, matching the golden).
  3. For every pool completed by the current step, gather the pool rows from
     ``state_cache`` (overlaying the current step's rows), compute
     ``softmax(gate + ape)`` and return the weighted sum of the pool keys.
  4. Update the tail (incomplete pool) rows of ``state_cache`` in place.

The computation follows ``key_pool_model_debug_golden.py`` (the CPU golden of
the CANN ``aclnnKeyPool`` operator) with the following model-side adaptations:

- ``state_cache`` may be BF16 (the GLM-5 Next compressor state cache) or FP32;
- ``cu_seqlens[-1]`` may be smaller than the row count of ``hidden_states``
  (CUDA-graph padded inputs are ignored);
- block-table entries may be ``-1`` for unused logical blocks and physical
  block ``0`` is a valid block (vLLM cache-manager convention).

RoPE (``cos``/``sin``) is reserved by the golden and not implemented here.
"""

from __future__ import annotations

import torch
from vllm.utils.torch_utils import direct_register_custom_op

_SUPPORTED_RATIOS = (2, 4, 8, 16, 32, 64, 128)


def _tensor_values(tensor: torch.Tensor) -> list[int]:
    return [int(value) for value in tensor.detach().cpu().reshape(-1).tolist()]


def _split_hidden_states(
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    batch_size: int,
) -> tuple[list[torch.Tensor], list[int]]:
    if hidden_states.dim() == 3:
        if cu_seqlens is not None:
            raise ValueError("cu_seqlens must be absent for BSH layout")
        if hidden_states.size(0) != batch_size:
            raise ValueError("hidden_states batch does not match cache_block_table")
        return (
            [hidden_states[batch] for batch in range(batch_size)],
            [int(hidden_states.size(1))] * batch_size,
        )

    if hidden_states.dim() != 2:
        raise ValueError("hidden_states rank must be 2 (TH) or 3 (BSH)")
    if cu_seqlens is None:
        raise ValueError("cu_seqlens is required for TH layout")
    if cu_seqlens.dtype != torch.int32 or cu_seqlens.dim() != 1:
        raise ValueError("cu_seqlens must be a rank-1 INT32 tensor")

    offsets = _tensor_values(cu_seqlens)
    if len(offsets) != batch_size + 1:
        raise ValueError("cu_seqlens shape must be [B+1]")
    if not offsets or offsets[0] != 0:
        raise ValueError("cu_seqlens[0] must be 0")
    if any(end < begin for begin, end in zip(offsets, offsets[1:])):
        raise ValueError("cu_seqlens must be nondecreasing")
    # CUDA-graph padded rows may extend beyond the actual token count.
    if offsets[-1] > hidden_states.size(0):
        raise ValueError("cu_seqlens[-1] must not exceed hidden_states.shape[0]")

    lengths = [end - begin for begin, end in zip(offsets, offsets[1:])]
    return (
        [hidden_states[offsets[batch] : offsets[batch + 1]] for batch in range(batch_size)],
        lengths,
    )


def _validate_inputs(
    hidden_states: torch.Tensor,
    wk: torch.Tensor,
    gate_weight: torch.Tensor,
    ape: torch.Tensor,
    state_cache: torch.Tensor,
    cache_block_table: torch.Tensor,
    start_pos: torch.Tensor,
    norm_weight: torch.Tensor | None,
    norm_bias: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    cmp_ratio: int,
    norm_eps: float,
) -> tuple[list[torch.Tensor], list[int]]:
    if cmp_ratio not in _SUPPORTED_RATIOS:
        raise ValueError(f"unsupported cmp_ratio={cmp_ratio}")
    if norm_eps <= 0:
        raise ValueError("norm_eps must be positive")
    if (norm_weight is None) != (norm_bias is None):
        raise ValueError("norm_weight and norm_bias must be passed as a pair")

    if hidden_states.dtype != torch.bfloat16:
        raise TypeError("hidden_states must be BF16")
    if wk.dtype != torch.bfloat16 or gate_weight.dtype != torch.bfloat16:
        raise TypeError("wk and gate_weight must be BF16")
    if ape.dtype != torch.float32:
        raise TypeError("ape must be FP32")
    if state_cache.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("state_cache must be BF16 or FP32")
    if cache_block_table.dtype != torch.int32 or start_pos.dtype != torch.int32:
        raise TypeError("cache_block_table and start_pos must be INT32")
    if wk.dim() != 2 or gate_weight.shape != wk.shape:
        raise ValueError("wk and gate_weight must have the same rank-2 shape")

    head_dim, hidden_size = wk.shape
    if hidden_states.dim() not in (2, 3):
        raise ValueError("hidden_states rank must be 2 (TH) or 3 (BSH)")
    if hidden_states.size(-1) != hidden_size:
        raise ValueError("hidden_states last dimension must match wk.shape[1]")
    if ape.shape != (cmp_ratio, head_dim):
        raise ValueError("ape shape must be [cmp_ratio, D]")
    if state_cache.dim() != 3 or state_cache.size(-1) != 2 * head_dim:
        raise ValueError("state_cache shape must be [block_num, block_size, 2*D]")
    if state_cache.size(0) == 0 or state_cache.size(1) == 0:
        raise ValueError("state_cache must not be empty")
    if cache_block_table.dim() != 2:
        raise ValueError("cache_block_table shape must be [B, max_block_num_per_batch]")
    if start_pos.dim() != 1 or start_pos.numel() != cache_block_table.size(0):
        raise ValueError("start_pos shape must be [B]")

    tensors = [
        wk,
        gate_weight,
        ape,
        state_cache,
        cache_block_table,
        start_pos,
    ]
    tensors.extend(tensor for tensor in (norm_weight, norm_bias, cu_seqlens) if tensor is not None)
    if any(tensor.device != hidden_states.device for tensor in tensors):
        raise ValueError("all tensor inputs must be on the same device")

    if norm_weight is not None:
        if norm_weight.dtype != torch.float32 or norm_bias.dtype != torch.float32:
            raise TypeError("norm_weight and norm_bias must be FP32")
        if norm_weight.shape != (head_dim,) or norm_bias.shape != (head_dim,):
            raise ValueError("norm_weight and norm_bias must have shape [D]")

    batch_hidden, lengths = _split_hidden_states(hidden_states, cu_seqlens, cache_block_table.size(0))
    starts = _tensor_values(start_pos)
    if any(value < 0 for value in starts):
        raise ValueError("start_pos must be non-negative")

    # Sync the whole block table once and slice the synced list in the loop
    # instead of copying one device->host chunk per batch.
    block_table_width = cache_block_table.size(1)
    block_table_values = _tensor_values(cache_block_table)
    logical_capacity = block_table_width * state_cache.size(1)
    for batch, length in enumerate(lengths):
        end_pos = starts[batch] + length
        if end_pos > logical_capacity:
            raise ValueError("updated sequence length exceeds logical cache capacity")
        required_blocks = (end_pos + state_cache.size(1) - 1) // state_cache.size(1)
        used_blocks = block_table_values[
            batch * block_table_width : batch * block_table_width + required_blocks
        ]
        if any(value < 0 or value >= state_cache.size(0) for value in used_blocks):
            raise ValueError("cache_block_table contains an out-of-range physical block")
    return batch_hidden, lengths


def _project_and_normalize(
    hidden: torch.Tensor,
    wk: torch.Tensor,
    gate_weight: torch.Tensor,
    norm_weight: torch.Tensor | None,
    norm_bias: torch.Tensor | None,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_fp32 = hidden.float()
    key = torch.matmul(hidden_fp32, wk.float().transpose(-1, -2)).to(hidden.dtype)
    gate = torch.matmul(hidden_fp32, gate_weight.float().transpose(-1, -2)).to(hidden.dtype)

    key = key.float()
    if norm_weight is not None:
        mean = key.mean(dim=-1, keepdim=True)
        variance = (key - mean).square().mean(dim=-1, keepdim=True)
        key = (key - mean) / torch.sqrt(variance + norm_eps)
        key = key * norm_weight + norm_bias
    return key, gate


def _gather_cache_rows(
    state_cache: torch.Tensor,
    cache_block_table: torch.Tensor,
    batch: int,
    logical_positions: torch.Tensor,
) -> torch.Tensor:
    block_size = state_cache.size(1)
    logical_blocks = torch.div(logical_positions, block_size, rounding_mode="floor")
    offsets = torch.remainder(logical_positions, block_size).long()
    physical_blocks = cache_block_table[batch, logical_blocks.long()].long()
    return state_cache[physical_blocks, offsets]


def _pool_current_results(
    state_cache: torch.Tensor,
    cache_block_table: torch.Tensor,
    start: int,
    batch: int,
    key: torch.Tensor,
    gate: torch.Tensor,
    ape: torch.Tensor,
    cmp_ratio: int,
) -> torch.Tensor:
    end = start + key.size(0)
    first_pool = start // cmp_ratio
    pool_count = end // cmp_ratio - first_pool
    if pool_count <= 0:
        return torch.empty((0, key.size(-1)), dtype=torch.bfloat16, device=state_cache.device)

    logical_positions = torch.arange(
        first_pool * cmp_ratio,
        (first_pool + pool_count) * cmp_ratio,
        dtype=torch.long,
        device=state_cache.device,
    ).view(pool_count, cmp_ratio)
    cached = _gather_cache_rows(state_cache, cache_block_table, batch, logical_positions)
    head_dim = key.size(-1)
    keys = cached[..., :head_dim].clone()
    gates = cached[..., head_dim:].clone()

    current_mask = logical_positions >= start
    current_indices = (logical_positions - start).clamp(min=0, max=key.size(0) - 1)
    keys = torch.where(current_mask[..., None], key[current_indices], keys)
    gates = torch.where(current_mask[..., None], gate[current_indices].float(), gates)

    probabilities = torch.softmax(gates + ape.view(1, cmp_ratio, head_dim), dim=1)
    probabilities = probabilities.to(torch.bfloat16).float()
    return (probabilities * keys).sum(dim=1).to(torch.bfloat16)


def _update_tail_cache(
    state_cache: torch.Tensor,
    cache_block_table: torch.Tensor,
    start: int,
    batch: int,
    key: torch.Tensor,
    gate: torch.Tensor,
    cmp_ratio: int,
) -> None:
    end = start + key.size(0)
    tail_length = end % cmp_ratio
    if tail_length == 0:
        return

    tail_start = max(start, end - tail_length)
    logical_positions = torch.arange(tail_start, end, dtype=torch.long, device=state_cache.device)
    block_size = state_cache.size(1)
    logical_blocks = torch.div(logical_positions, block_size, rounding_mode="floor")
    offsets = torch.remainder(logical_positions, block_size).long()
    physical_blocks = cache_block_table[batch, logical_blocks.long()].long()

    current_indices = logical_positions - start
    head_dim = key.size(-1)
    state_cache[physical_blocks, offsets, :head_dim] = key[current_indices].to(state_cache.dtype)
    state_cache[physical_blocks, offsets, head_dim:] = gate[current_indices].to(state_cache.dtype)


def glm5_next_key_pool(
    hidden_states: torch.Tensor,
    wk: torch.Tensor,
    gate_weight: torch.Tensor,
    ape: torch.Tensor,
    state_cache: torch.Tensor,
    cache_block_table: torch.Tensor,
    start_pos: torch.Tensor,
    *,
    norm_weight: torch.Tensor | None = None,
    norm_bias: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    cmp_ratio: int = 4,
    norm_eps: float = 1e-6,
) -> torch.Tensor:
    """Compress the current step's completed pools and update the tail cache.

    CPU reference implementation (golden semantics). Non-CPU inputs are
    executed on CPU to avoid NPU kernel dtype/index-form restrictions; the
    mutated ``state_cache`` (tail update) is copied back to its device and
    ``pooled_key`` is returned on the input device.
    """
    if hidden_states.device.type != "cpu":
        state_cache_cpu = state_cache.cpu()
        pooled_key = _glm5_next_key_pool_impl(
            hidden_states.cpu(),
            wk.cpu(),
            gate_weight.cpu(),
            ape.cpu(),
            state_cache_cpu,
            cache_block_table.cpu(),
            start_pos.cpu(),
            norm_weight=norm_weight.cpu() if norm_weight is not None else None,
            norm_bias=norm_bias.cpu() if norm_bias is not None else None,
            cu_seqlens=cu_seqlens.cpu() if cu_seqlens is not None else None,
            cmp_ratio=cmp_ratio,
            norm_eps=norm_eps,
        )
        state_cache.copy_(state_cache_cpu)
        return pooled_key.to(hidden_states.device)
    return _glm5_next_key_pool_impl(
        hidden_states,
        wk,
        gate_weight,
        ape,
        state_cache,
        cache_block_table,
        start_pos,
        norm_weight=norm_weight,
        norm_bias=norm_bias,
        cu_seqlens=cu_seqlens,
        cmp_ratio=cmp_ratio,
        norm_eps=norm_eps,
    )


def _glm5_next_key_pool_impl(
    hidden_states: torch.Tensor,
    wk: torch.Tensor,
    gate_weight: torch.Tensor,
    ape: torch.Tensor,
    state_cache: torch.Tensor,
    cache_block_table: torch.Tensor,
    start_pos: torch.Tensor,
    *,
    norm_weight: torch.Tensor | None = None,
    norm_bias: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    cmp_ratio: int = 4,
    norm_eps: float = 1e-6,
) -> torch.Tensor:
    """Compress the current step's completed pools and update the tail cache.

    Args:
        hidden_states: ``[T, H]`` BF16 input activations (TH layout).
        wk: ``[D, H]`` BF16 KV projection weight (full, unsharded).
        gate_weight: ``[D, H]`` BF16 gate projection weight.
        ape: ``[cmp_ratio, D]`` FP32 positional biases.
        state_cache: ``[block_num, block_size, 2*D]`` BF16/FP32 compressor state
            cache; the tail (incomplete pool) rows are updated in place.
        cache_block_table: ``[B, max_block_num_per_batch]`` INT32 physical block
            mapping; ``-1`` marks unused logical blocks.
        start_pos: ``[B]`` INT32 start position of the current step per batch.
        norm_weight/norm_bias: optional ``[D]`` FP32 LayerNorm-style affine
            parameters applied to the projected key.
        cu_seqlens: ``[B+1]`` INT32 prefix sums of the step's token counts
            (required for the TH layout).
        cmp_ratio: pool size (2/4/8/16/32/64/128).
        norm_eps: epsilon for the optional key normalization.

    Returns:
        ``pooled_key`` ``[B, pool_capacity, D]`` BF16: the compressed key of
        every pool completed by the current step, per batch.  Rows beyond the
        completed pools are zero.
    """
    batch_hidden, lengths = _validate_inputs(
        hidden_states,
        wk,
        gate_weight,
        ape,
        state_cache,
        cache_block_table,
        start_pos,
        norm_weight,
        norm_bias,
        cu_seqlens,
        cmp_ratio,
        norm_eps,
    )

    projected = [
        _project_and_normalize(hidden, wk, gate_weight, norm_weight, norm_bias, norm_eps) for hidden in batch_hidden
    ]
    logical_capacity = cache_block_table.size(1) * state_cache.size(1)
    pool_capacity = (logical_capacity + cmp_ratio - 1) // cmp_ratio
    pooled_key = torch.zeros(
        (len(batch_hidden), pool_capacity, wk.size(0)),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    starts = _tensor_values(start_pos)
    for batch, ((key, gate), length) in enumerate(zip(projected, lengths)):
        pooled = _pool_current_results(
            state_cache,
            cache_block_table,
            starts[batch],
            batch,
            key,
            gate,
            ape,
            cmp_ratio,
        )
        pooled_key[batch, : pooled.size(0)] = pooled
        _update_tail_cache(
            state_cache,
            cache_block_table,
            starts[batch],
            batch,
            key,
            gate,
            cmp_ratio,
        )
    return pooled_key


def glm5_next_key_pool_fake(
    hidden_states: torch.Tensor,
    wk: torch.Tensor,
    gate_weight: torch.Tensor,
    ape: torch.Tensor,
    state_cache: torch.Tensor,
    cache_block_table: torch.Tensor,
    start_pos: torch.Tensor,
    *,
    norm_weight: torch.Tensor | None = None,
    norm_bias: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    cmp_ratio: int = 4,
    norm_eps: float = 1e-6,
) -> torch.Tensor:
    del gate_weight, ape, start_pos, norm_weight, norm_bias, cu_seqlens, norm_eps
    if cmp_ratio <= 0:
        raise ValueError(f"cmp_ratio must be positive, got {cmp_ratio}.")
    logical_capacity = cache_block_table.shape[1] * state_cache.shape[1]
    pool_capacity = (logical_capacity + cmp_ratio - 1) // cmp_ratio
    return torch.empty(
        (cache_block_table.shape[0], pool_capacity, wk.shape[0]),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )


direct_register_custom_op(
    op_name="glm5_next_key_pool",
    op_func=glm5_next_key_pool,
    mutates_args=["state_cache"],
    fake_impl=glm5_next_key_pool_fake,
    dispatch_key="PrivateUse1",
)
