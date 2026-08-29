# SPDX-License-Identifier: Apache-2.0
"""Graph-safe device ops for GLM-5 Next KPool attention."""

import torch
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ops.triton.glm5_next_kpool_compress import (
    glm5_next_kpool_compress_and_write_cache_triton,
)
from vllm_ascend.ops.triton.glm5_next_lightning_indexer import (
    glm5_next_lightning_indexer_triton,
)


def kpool_compress(
    kv_cache: torch.Tensor,
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
) -> None:
    glm5_next_kpool_compress_and_write_cache_triton(kv_cache, slot_k, slot_score, ape, loc)


def kpool_compress_fake(
    kv_cache: torch.Tensor,
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
) -> None:
    return None


def lightning_indexer(
    query: torch.Tensor,
    indexer_cache: torch.Tensor,
    weights: torch.Tensor,
    cum_query_lens: torch.Tensor,
    indexer_seq_lens: torch.Tensor,
    indexer_block_table: torch.Tensor,
    positions: torch.Tensor,
    *,
    index_topk: int,
    index_kpool: int,
    max_pool_seq_len: int,
) -> torch.Tensor:
    return glm5_next_lightning_indexer_triton(
        query,
        indexer_cache,
        weights,
        cum_query_lens,
        indexer_seq_lens,
        indexer_block_table,
        positions,
        index_topk=index_topk,
        index_kpool=index_kpool,
        max_pool_seq_len=max_pool_seq_len,
    )


def lightning_indexer_fake(
    query: torch.Tensor,
    indexer_cache: torch.Tensor,
    weights: torch.Tensor,
    cum_query_lens: torch.Tensor,
    indexer_seq_lens: torch.Tensor,
    indexer_block_table: torch.Tensor,
    positions: torch.Tensor,
    *,
    index_topk: int,
    index_kpool: int,
    max_pool_seq_len: int,
) -> torch.Tensor:
    del indexer_cache, weights, cum_query_lens, indexer_seq_lens
    del indexer_block_table, positions, max_pool_seq_len
    return torch.empty(
        (query.shape[0], 1, index_topk + index_kpool - 1),
        dtype=torch.int32,
        device=query.device,
    )


direct_register_custom_op(
    op_name="glm5_next_kpool_compress_and_write_cache",
    op_func=kpool_compress,
    mutates_args=["kv_cache"],
    fake_impl=kpool_compress_fake,
    dispatch_key="PrivateUse1",
)
direct_register_custom_op(
    op_name="glm5_next_lightning_indexer",
    op_func=lightning_indexer,
    mutates_args=[],
    fake_impl=lightning_indexer_fake,
    dispatch_key="PrivateUse1",
)
