# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full GLM5 Next PoolKeyIndexer CPU reference (adapted from the golden).

The op selects the top-k token positions for every query row with a pooling
mechanism on the key side:

  1. Pool-level scores (per batch, float32):
       dot[i, n, j] = dot(query[i, n, :], pool_key[j, :]) / sqrt(headDim)
       score[i, j]  = sum_n weights[i, n] * relu(dot[i, n, j])
  2. Causal visibility (mask_mode=3, rightDownCausal): query row ``i`` may
     only see the first ``(gpos + 1) // pool_size`` pools.
  3. Pool-level TopK: keep ``topk // pool_size`` pools with the highest scores.
  4. Index expansion: a selected pool ``p`` expands to the token indices
     ``p * pool_size, ..., p * pool_size + pool_size - 1``.
  5. Tail append: the batch's last incomplete pool contributes up to
     ``pool_tail_k`` tokens, capped by the top-k window in causal mode.

The computation follows ``pool_key_indexer_reference.py`` (the CPU golden of
the CANN ``pool_key_indexer`` operator).  The output is padded to
``query.shape[0]`` rows so the shape stays fixed under CUDA-graph capture.
"""

from __future__ import annotations

import math

import torch
from vllm.utils.torch_utils import direct_register_custom_op

_NEG_INF = float("-inf")


def _as_int_list(x):
    """Convert an int tensor / list / scalar to a list of ints."""
    if x is None:
        return None
    if torch.is_tensor(x):
        return [int(v) for v in x.detach().cpu().reshape(-1).tolist()]
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    return [int(x)]


def _prefix_to_counts(prefix):
    """Convert prefix sums to per-batch counts."""
    p = _as_int_list(prefix)
    return [p[0]] + [p[i] - p[i - 1] for i in range(1, len(p))]


def _split_query(query, weights, layout_q, actual_seq_q):
    """Split query/weights into per-batch (Sb, N1, D) / (Sb, N1) float32 lists.

    Returns (q_list, w_list, counts, batch, s1_dim):
      - counts: valid query token count per batch
      - s1_dim: S1 dim of the BSND output (query.shape[1], incl. invalid rows)
    """
    if layout_q == "BSND":
        batch, s1_dim, n1, d = query.shape
        if actual_seq_q is not None:
            counts = _as_int_list(actual_seq_q)
        else:
            counts = [s1_dim] * batch
        q_list = [query[b, :c].to(torch.float32) for b, c in enumerate(counts)]
        w_list = [weights[b, :c].to(torch.float32) for b, c in enumerate(counts)]
        return q_list, w_list, counts, batch, s1_dim
    if layout_q == "TND":
        t1, n1, d = query.shape
        if actual_seq_q is None:
            raise ValueError("layout_q='TND' requires actual_seq_q (prefix sums)")
        counts = _prefix_to_counts(actual_seq_q)
        q_list, w_list, start = [], [], 0
        for c in counts:
            q_list.append(query[start : start + c].to(torch.float32))
            w_list.append(weights[start : start + c].to(torch.float32))
            start += c
        return q_list, w_list, counts, len(counts), t1
    raise ValueError(f"unsupported layout_q: {layout_q}")


def _split_key(pool_key, layout_k, actual_seq_k, block_table):
    """Split pool_key into per-batch (Pb, D) float32 lists.

    - BSND:    pool_key (B, S2, N2, D); actual_seq_k optional (pools per batch)
    - TND:     pool_key (T2, N2, D);    actual_seq_k required (pool prefix sums)
    - PA_BBND: pool_key (blockNum, blockSize, N2, D), one pool per block row;
               requires block_table (B, maxBlocks) and actual_seq_k
               (pools per batch); logical pool p lives in physical block
               block_table[b][p // blockSize] at row p % blockSize;
               empty slots (-1) are skipped.
    """
    if layout_k == "BSND":
        batch, s2, n2, d = pool_key.shape
        if actual_seq_k is not None:
            counts = _as_int_list(actual_seq_k)
        else:
            counts = [s2] * batch
        return [pool_key[b, :c, 0, :].to(torch.float32) for b, c in enumerate(counts)]
    if layout_k == "TND":
        t2, n2, d = pool_key.shape
        if actual_seq_k is None:
            raise ValueError("layout_k='TND' requires actual_seq_k (prefix sums)")
        counts = _prefix_to_counts(actual_seq_k)
        keys, start = [], 0
        for c in counts:
            keys.append(pool_key[start : start + c, 0, :].to(torch.float32))
            start += c
        return keys
    if layout_k == "PA_BBND":
        if block_table is None:
            raise ValueError("layout_k='PA_BBND' requires block_table")
        if actual_seq_k is None:
            raise ValueError("layout_k='PA_BBND' requires actual_seq_k (pools per batch)")
        block_num, block_size, n2, d = pool_key.shape
        bt = block_table.detach().cpu()
        batch, max_blocks = bt.shape
        counts = _as_int_list(actual_seq_k)
        keys = []
        for b in range(batch):
            need, rows, pos = counts[b], [], 0
            for t in range(max_blocks):
                if pos >= need:
                    break
                blk = int(bt[b, t])
                if blk < 0:
                    continue
                take = min(block_size, need - pos)
                rows.append(pool_key[blk, :take, 0, :].to(torch.float32))
                pos += take
            if rows:
                keys.append(torch.cat(rows, dim=0))
            else:
                keys.append(pool_key.new_zeros((0, d)).to(torch.float32))
        return keys
    raise ValueError(f"unsupported layout_k: {layout_k}")


def _compute_batch(q_b, w_b, k_b, tail_k, topk, pool_size, mask_mode):
    """Pool-level top-k selection and index expansion for one batch.

    Args:
      q_b: (S, N1, D) float32;  w_b: (S, N1) float32;  k_b: (P, D) float32;
      tail_k: valid token count of the trailing incomplete pool.

    Returns:
      token_idx: (S, topk + pool_size - 1) int32, invalid entries -1
      pool_val:  (S, topk // pool_size) float32,   invalid entries -inf
    """
    s_len, n1, d = q_b.shape
    p_len = k_b.shape[0]
    sparse_count = topk // pool_size
    out_len = topk + pool_size - 1
    l_orig = p_len * pool_size + tail_k

    token_idx = torch.full((s_len, out_len), -1, dtype=torch.int32, device=q_b.device)
    pool_val = torch.full((s_len, sparse_count), _NEG_INF, dtype=torch.float32, device=q_b.device)
    if s_len == 0 or p_len == 0:
        return token_idx, pool_val

    # Pool scores: score[i, j] = sum_n w[i, n] * relu(q[i, n] . k[j] / sqrt(d))
    scale = 1.0 / math.sqrt(d)
    dot = torch.einsum("snd,pd->snp", q_b, k_b) * scale  # (S, N1, P)
    scores = torch.einsum("sn,snp->sp", w_b, dot.clamp_min(0.0))  # (S, P)

    arange_pool = torch.arange(pool_size, dtype=torch.int64, device=q_b.device)
    for i in range(s_len):
        # Causal pool visibility (rightDownCausal): floor division.
        if mask_mode == 3:
            valid = (l_orig - s_len + i + 1) // pool_size
            valid = max(0, min(valid, p_len))
        else:
            valid = p_len
        select = min(valid, sparse_count)
        if select > 0:
            # Pool-level top-k (descending scores, ascending index on ties).
            order = torch.argsort(scores[i, :valid], descending=True, stable=True)
            picked = order[:select]
            # Expand selected pools to token indices, filling in order.
            tokens = picked.to(torch.int64).unsqueeze(1) * pool_size + arange_pool
            flat = tokens.reshape(-1)
            token_idx[i, : flat.numel()] = flat.to(torch.int32)
            pool_val[i, :select] = scores[i, picked]

        # Tail append (from the topk column on, capacity pool_size - 1).
        if pool_size > 1 and tail_k > 0:
            if mask_mode == 0:
                visible = tail_k
            else:
                gpos = l_orig - s_len + i
                visible = max(0, min(tail_k, gpos - topk + 1))
            if visible > 0:
                tail_tokens = torch.arange(
                    l_orig - tail_k,
                    l_orig - tail_k + visible,
                    dtype=torch.int32,
                    device=q_b.device,
                )
                token_idx[i, topk : topk + visible] = tail_tokens

    return token_idx, pool_val


def glm5_next_pool_key_indexer(
    query: torch.Tensor,
    pool_key: torch.Tensor,
    weights: torch.Tensor,
    pool_tail_k: torch.Tensor,
    *,
    actual_seq_q: torch.Tensor | None = None,
    actual_seq_k: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    layout_q: str = "TND",
    layout_k: str = "PA_BBND",
    topk: int = 128,
    pool_size: int = 16,
    mask_mode: int = 3,
    return_value: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU reference implementation of the pool_key_indexer op.

    Non-CPU inputs are executed on CPU (golden semantics) to avoid NPU kernel
    dtype/index-form restrictions; outputs are moved back to the input device.
    """
    if query.device.type != "cpu":
        indices, values = _glm5_next_pool_key_indexer_impl(
            query.cpu(),
            pool_key.cpu(),
            weights.cpu(),
            pool_tail_k.cpu(),
            actual_seq_q=actual_seq_q.cpu() if actual_seq_q is not None else None,
            actual_seq_k=actual_seq_k.cpu() if actual_seq_k is not None else None,
            block_table=block_table.cpu() if block_table is not None else None,
            layout_q=layout_q,
            layout_k=layout_k,
            topk=topk,
            pool_size=pool_size,
            mask_mode=mask_mode,
            return_value=return_value,
        )
        return indices.to(query.device), values.to(query.device)
    return _glm5_next_pool_key_indexer_impl(
        query,
        pool_key,
        weights,
        pool_tail_k,
        actual_seq_q=actual_seq_q,
        actual_seq_k=actual_seq_k,
        block_table=block_table,
        layout_q=layout_q,
        layout_k=layout_k,
        topk=topk,
        pool_size=pool_size,
        mask_mode=mask_mode,
        return_value=return_value,
    )


def _glm5_next_pool_key_indexer_impl(
    query: torch.Tensor,
    pool_key: torch.Tensor,
    weights: torch.Tensor,
    pool_tail_k: torch.Tensor,
    *,
    actual_seq_q: torch.Tensor | None = None,
    actual_seq_k: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    layout_q: str = "TND",
    layout_k: str = "PA_BBND",
    topk: int = 128,
    pool_size: int = 16,
    mask_mode: int = 3,
    return_value: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU reference implementation of the pool_key_indexer op.

    Args:
      query: Query tensor, dtype float16/bfloat16/float32 (computed in float32).
        - layout_q="TND": shape (T1, N1, D), T1 total query tokens over batches;
          CUDA-graph padded rows are allowed, skipped, and filled with -1.
      pool_key: Pool-level key tensor (one pool key per row), dtype matches query:
        - layout_k="PA_BBND": shape (blockNum, blockSize, N2, D), one pool per
          block row; requires block_table and actual_seq_k; supports a
          non-contiguous (strided) 0 axis.
      weights: Per-head weights, shape (T1, N1) (TND).
      pool_tail_k: Valid token count of each batch's trailing incomplete pool,
        shape (B,), range [0, pool_size-1].
      actual_seq_q: Required for layout_q="TND"; prefix sums, length B, last
        entry equals T1.
      actual_seq_k: Required for layout_k="PA_BBND"; pools per batch (not
        prefix sums).
      block_table: Required for layout_k="PA_BBND"; shape
        (B, maxBlockNumPerSeq), int32 physical block ids, -1 for empty slots.
      layout_q: Query layout, "BSND" or "TND".
      layout_k: Pool-key layout, "BSND", "TND" or "PA_BBND".
      topk: Number of expanded tokens to keep; topk % pool_size == 0.
      pool_size: Tokens per pool, range [1, 128].
      mask_mode: 0 = no mask; 3 = rightDownCausal.
      return_value: True returns values; False returns an empty tensor.

    Returns:
      (indices, values):
        indices: int32; TND shape (T1, topk+pool_size-1), invalid entries -1;
          output rows padded to query.shape[0] (CUDA-graph padding rows -1).
        values: float32; return_value=True TND shape (T1, topk//pool_size),
          invalid entries -inf; return_value=False an empty tensor (0,).
    """
    if layout_q not in ("BSND", "TND"):
        raise ValueError(f"unsupported layout_q: {layout_q}")
    if layout_k not in ("BSND", "TND", "PA_BBND"):
        raise ValueError(f"unsupported layout_k: {layout_k}")
    if mask_mode not in (0, 3):
        raise ValueError(f"unsupported mask_mode: {mask_mode}")
    if pool_size < 1:
        raise ValueError(f"pool_size must be >= 1, got {pool_size}")
    if topk % pool_size != 0:
        raise ValueError(f"topk({topk}) must be divisible by pool_size({pool_size})")
    if pool_key.shape[-2] != 1:
        raise ValueError(f"pool_key N2 must be 1, got {pool_key.shape[-2]}")
    if query.shape[-1] != pool_key.shape[-1]:
        raise ValueError(f"headDim mismatch: query {query.shape[-1]} vs pool_key {pool_key.shape[-1]}")

    q_list, w_list, q_counts, batch, s1_dim = _split_query(query, weights, layout_q, actual_seq_q)
    k_list = _split_key(pool_key, layout_k, actual_seq_k, block_table)
    if len(k_list) != batch:
        raise ValueError(f"batch mismatch: query {batch} vs pool_key {len(k_list)}")
    tail_list = _as_int_list(pool_tail_k) if pool_tail_k is not None else [0] * batch
    if len(tail_list) != batch:
        raise ValueError(f"pool_tail_k length {len(tail_list)} != batch {batch}")

    idx_rows, val_rows = [], []
    for b in range(batch):
        token_idx, pool_val = _compute_batch(q_list[b], w_list[b], k_list[b], tail_list[b], topk, pool_size, mask_mode)
        idx_rows.append(token_idx)
        val_rows.append(pool_val)

    out_len = topk + pool_size - 1
    sparse_count = topk // pool_size
    if layout_q == "BSND":
        indices = torch.full((batch, s1_dim, out_len), -1, dtype=torch.int32, device=query.device)
        values = torch.full(
            (batch, s1_dim, sparse_count), _NEG_INF, dtype=torch.float32, device=query.device
        )
        for b in range(batch):
            c = q_counts[b]
            indices[b, :c] = idx_rows[b]
            values[b, :c] = val_rows[b]
    else:  # TND: concat all rows and pad to query.shape[0] for graph-stable shapes
        total_rows = query.shape[0]
        indices = torch.full((total_rows, out_len), -1, dtype=torch.int32, device=query.device)
        values = torch.full(
            (total_rows, sparse_count), _NEG_INF, dtype=torch.float32, device=query.device
        )
        if idx_rows:
            actual_rows = sum(q_counts)
            indices[:actual_rows] = torch.cat(idx_rows, dim=0)
            values[:actual_rows] = torch.cat(val_rows, dim=0)

    if not return_value:
        values = torch.empty(0, dtype=torch.float32, device=query.device)
    return indices, values


def glm5_next_pool_key_indexer_fake(
    query: torch.Tensor,
    pool_key: torch.Tensor,
    weights: torch.Tensor,
    pool_tail_k: torch.Tensor,
    *,
    actual_seq_q: torch.Tensor | None = None,
    actual_seq_k: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    layout_q: str = "TND",
    layout_k: str = "PA_BBND",
    topk: int = 128,
    pool_size: int = 16,
    mask_mode: int = 3,
    return_value: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    del pool_key, weights, pool_tail_k, actual_seq_q, actual_seq_k, block_table
    del layout_q, layout_k, mask_mode
    if topk <= 0 or pool_size <= 0 or topk % pool_size:
        raise ValueError(f"Invalid GLM5 Next pool indexer topk/pool attrs: {topk}, {pool_size}.")
    indices = torch.empty(
        (query.shape[0], topk + pool_size - 1),
        dtype=torch.int32,
        device=query.device,
    )
    if return_value:
        values = torch.empty(
            (query.shape[0], topk // pool_size),
            dtype=torch.float32,
            device=query.device,
        )
    else:
        values = torch.empty(0, dtype=torch.float32, device=query.device)
    return indices, values


direct_register_custom_op(
    op_name="glm5_next_pool_key_indexer",
    op_func=glm5_next_pool_key_indexer,
    mutates_args=[],
    fake_impl=glm5_next_pool_key_indexer_fake,
    dispatch_key="PrivateUse1",
)
