from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm_ascend.device.device_op import A5DeviceAdaptor, BaseDeviceAdaptor


def test_glm5_sparse_attention_device_contract_is_a5_only():
    assert not BaseDeviceAdaptor.supports_sharedkv_indexer_kpool_mla()
    assert A5DeviceAdaptor.supports_sharedkv_indexer_kpool_mla()
    assert A5DeviceAdaptor.get_sparse_attention_metadata_kwargs_indexer_kpool_mla(
        torch.device("cpu")
    ) == {"device": "cpu"}
    with mock.patch.object(
        torch.ops._C_ascend,
        "npu_sparse_flash_mla_metadata",
        create=True,
    ) as metadata_op:
        assert A5DeviceAdaptor.get_sparse_attention_metadata_op_indexer_kpool_mla() is metadata_op
    # The existing DSA contract must retain its 64-dimensional RoPE.
    assert A5DeviceAdaptor.get_dsa_sparse_attn_base_kwargs()["rope_head_dim"] == 64


def test_base_glm5_sparse_attention_delegates_to_small_op_path():
    """BaseDeviceAdaptor routes Indexer KPool MLA through npu_sparse_flash_attention."""
    expected = torch.randn((1, 8, 512), dtype=torch.bfloat16)
    packed_kv_cache = torch.zeros((1, 128, 1, 512), dtype=torch.bfloat16)
    block_table = torch.zeros((1, 1), dtype=torch.int32)
    ql_nope = torch.zeros_like(expected)
    q_pe = torch.empty((1, 8, 0), dtype=torch.bfloat16)
    topk_indices = torch.zeros((1, 1, 515), dtype=torch.int32)
    query_lens = torch.tensor([1], dtype=torch.int32)
    key_lens = torch.tensor([1], dtype=torch.int32)

    with mock.patch.object(
        torch.ops._C_ascend,
        "npu_sparse_flash_attention",
        return_value=expected,
        create=True,
    ) as sparse_op:
        result = BaseDeviceAdaptor.execute_sparse_attention_indexer_kpool_mla(
            SimpleNamespace(scale=0.125),
            ql_nope,
            q_pe,
            packed_kv_cache,
            topk_indices,
            SimpleNamespace(block_table=block_table, num_actual_tokens=1),
            query_lens,
            key_lens,
        )

    assert result is expected
    sparse_op.assert_called_once()
    call_kwargs = sparse_op.call_args.kwargs
    assert call_kwargs["key"] is packed_kv_cache
    assert call_kwargs["value"] is packed_kv_cache
    assert call_kwargs["sparse_indices"] is topk_indices
    assert call_kwargs["scale_value"] == 0.125
    assert call_kwargs["sparse_block_size"] == 1
    assert call_kwargs["block_table"] is block_table
    assert call_kwargs["actual_seq_lengths_query"] is query_lens
    assert call_kwargs["actual_seq_lengths_kv"] is key_lens
    assert call_kwargs["query_rope"] is None
    assert call_kwargs["key_rope"] is None
    assert call_kwargs["layout_query"] == "TND"
    assert call_kwargs["layout_kv"] == "PA_BSND"
    assert call_kwargs["sparse_mode"] == 3
    assert call_kwargs["attention_mode"] == 2
    assert call_kwargs["return_softmax_lse"] is False
    # query must be cat(ql_nope, q_pe) contiguous
    query = call_kwargs["query"]
    assert query.shape == (1, 8, 512)
    assert query.is_contiguous()


def test_base_glm5_sparse_attention_pads_actual_token_indices_to_query_rows():
    """Eager MTP draft: topk_indices carry num_actual_tokens rows, aligned with
    the leading query rows; they must be zero-padded to the padded query length."""
    num_tokens = 76
    num_actual_tokens = 2
    expected = torch.randn((num_tokens, 8, 512), dtype=torch.bfloat16)
    packed_kv_cache = torch.zeros((1, 128, 1, 512), dtype=torch.bfloat16)
    block_table = torch.zeros((num_actual_tokens, 1), dtype=torch.int32)
    ql_nope = torch.zeros_like(expected)
    q_pe = torch.empty((num_tokens, 8, 0), dtype=torch.bfloat16)
    # per-actual-token topk_indices: shape [num_actual_tokens, 1, K]
    topk_indices = torch.arange(num_actual_tokens * 515, dtype=torch.int32).reshape(num_actual_tokens, 1, 515)
    # cumulative query lengths: 2 requests with 1 real draft token each
    cum_query_lens = torch.tensor([1, 2], dtype=torch.int32)
    key_lens = torch.tensor([40, 40], dtype=torch.int32)

    with mock.patch.object(
        torch.ops._C_ascend,
        "npu_sparse_flash_attention",
        return_value=expected,
        create=True,
    ) as sparse_op:
        result = BaseDeviceAdaptor.execute_sparse_attention_indexer_kpool_mla(
            SimpleNamespace(scale=0.125),
            ql_nope,
            q_pe,
            packed_kv_cache,
            topk_indices,
            SimpleNamespace(block_table=block_table, num_actual_tokens=num_actual_tokens),
            cum_query_lens,
            key_lens,
        )

    assert result is expected
    sparse_op.assert_called_once()
    call_kwargs = sparse_op.call_args.kwargs
    # sparse_indices must cover all padded query rows: [num_tokens, 1, 515]
    sparse_indices = call_kwargs["sparse_indices"]
    assert sparse_indices.shape == (num_tokens, 1, 515)
    # the leading real rows keep their per-token indices
    assert (sparse_indices[:num_actual_tokens] == topk_indices).all()
    # padded rows are in-bounds zeros and are never consumed by the kernel
    assert (sparse_indices[num_actual_tokens:] == 0).all()


def test_base_glm5_sparse_attention_rejects_unaligned_indices():
    """topk_indices that match neither query rows nor actual tokens must raise."""
    num_tokens = 76
    expected = torch.randn((num_tokens, 8, 512), dtype=torch.bfloat16)
    packed_kv_cache = torch.zeros((1, 128, 1, 512), dtype=torch.bfloat16)
    block_table = torch.zeros((2, 1), dtype=torch.int32)
    ql_nope = torch.zeros_like(expected)
    q_pe = torch.empty((num_tokens, 8, 0), dtype=torch.bfloat16)
    # 3 rows: neither the padded query length (76) nor num_actual_tokens (2)
    topk_indices = torch.zeros((3, 1, 515), dtype=torch.int32)
    cum_query_lens = torch.tensor([1, 2], dtype=torch.int32)
    key_lens = torch.tensor([40, 40], dtype=torch.int32)

    with pytest.raises(RuntimeError, match="topk_indices rows"):
        BaseDeviceAdaptor.execute_sparse_attention_indexer_kpool_mla(
            SimpleNamespace(scale=0.125),
            ql_nope,
            q_pe,
            packed_kv_cache,
            topk_indices,
            SimpleNamespace(block_table=block_table, num_actual_tokens=2),
            cum_query_lens,
            key_lens,
        )


def test_a5_glm5_sparse_attention_uses_non_quantized_sharedkv():
    query = torch.zeros((1, 8, 512), dtype=torch.bfloat16)
    kv = torch.zeros((1, 128, 1, 512), dtype=torch.bfloat16)
    topk_indices = torch.zeros((1, 1, 515), dtype=torch.int32)
    block_table = torch.zeros((1, 1), dtype=torch.int32)
    query_lens = torch.tensor([1], dtype=torch.int32)
    key_lens = torch.tensor([1], dtype=torch.int32)
    sas_metadata = torch.zeros(1024, dtype=torch.int32)
    sas_sinks = torch.ones(8, dtype=torch.float32)
    metadata = SimpleNamespace(
        block_table=block_table,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        sas_metadata=sas_metadata,
        sas_sinks=sas_sinks,
    )
    expected = torch.ones_like(query)

    with mock.patch.object(
        torch.ops._C_ascend,
        "npu_sparse_flash_mla",
        return_value=(expected, torch.empty(0)),
        create=True,
    ) as sparse_op:
        result = A5DeviceAdaptor.execute_sparse_attention_indexer_kpool_mla(
            SimpleNamespace(scale=0.125, qk_rope_head_dim=0, kv_lora_rank=512),
            query,
            torch.empty((1, 8, 0), dtype=torch.bfloat16),
            kv,
            topk_indices,
            metadata,
            query_lens,
            key_lens,
            block_table=block_table,
        )

    assert result is expected
    sparse_op.assert_called_once()
    call_args, kwargs = sparse_op.call_args
    assert call_args == (query,)
    # GLM-5 passes its only KV cache as the original (ori) KV — no compressed path.
    assert kwargs["ori_kv"] is kv
    assert kwargs["cmp_kv"] is None
    assert kwargs["ori_sparse_indices"] is topk_indices
    assert kwargs["cmp_sparse_indices"] is None
    assert kwargs["ori_block_table"] is block_table
    assert kwargs["cmp_block_table"] is None
    assert kwargs["cu_seqlens_q"] is metadata.query_start_loc
    assert kwargs["seqused_ori_kv"] is key_lens
    assert kwargs["seqused_cmp_kv"] is None
    assert kwargs["metadata"] is sas_metadata
    assert kwargs["sinks"] is sas_sinks
    assert kwargs["ori_mask_mode"] == 3
    assert kwargs["layout_kv"] == "PA_BBND"
    assert kwargs["topk_value_mode"] == 1
    assert kwargs["seqused_q"] is None
    assert "kv_quant_mode" not in kwargs


def test_npu_flash_attention_uses_fusion_attention_for_fp32():
    query = torch.randn(5, 4, 64, dtype=torch.float32)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    seq_lens_cpu = torch.tensor([2, 3], dtype=torch.int32)
    expected = torch.randn_like(query)

    with (
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu.npu_fusion_attention",
            return_value=(expected,),
        ) as mock_fusion_attention,
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu._npu_flash_attention_unpad",
            create=True,
        ) as mock_flash_attention,
    ):
        output = BaseDeviceAdaptor.npu_flash_attention(
            query=query,
            key=key,
            value=value,
            seq_lens_cpu=seq_lens_cpu,
            head_num=4,
            scale_value=0.125,
            num_kv_heads=4,
        )

    assert output is expected
    mock_flash_attention.assert_not_called()
    mock_fusion_attention.assert_called_once()
    call_kwargs = mock_fusion_attention.call_args.kwargs
    assert call_kwargs["query"] is query
    assert call_kwargs["key"] is key
    assert call_kwargs["value"] is value
    assert call_kwargs["actual_seq_qlen"] == [2, 5]
    assert all(isinstance(seq_len, int) for seq_len in call_kwargs["actual_seq_qlen"])
    assert call_kwargs["actual_seq_kvlen"] is call_kwargs["actual_seq_qlen"]
    assert call_kwargs["head_num"] == 4
    assert call_kwargs["scale"] == 0.125
    assert call_kwargs["input_layout"] == "TND"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_npu_flash_attention_uses_unpad_attention_for_low_precision(dtype):
    query = torch.randn(5, 4, 64, dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    seq_lens_cpu = torch.tensor([2, 3], dtype=torch.int32)

    def fake_flash_attention(*, query, key, value, seq_len, scale_value, num_heads, num_kv_heads, out):
        out.copy_(query + 1)

    with (
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu.npu_fusion_attention",
        ) as mock_fusion_attention,
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu._npu_flash_attention_unpad",
            side_effect=fake_flash_attention,
            create=True,
        ) as mock_flash_attention,
    ):
        output = BaseDeviceAdaptor.npu_flash_attention(
            query=query,
            key=key,
            value=value,
            seq_lens_cpu=seq_lens_cpu,
            head_num=4,
            scale_value=0.125,
            num_kv_heads=4,
        )

    mock_fusion_attention.assert_not_called()
    mock_flash_attention.assert_called_once()
    call_kwargs = mock_flash_attention.call_args.kwargs
    assert call_kwargs["query"] is query
    assert call_kwargs["key"] is key
    assert call_kwargs["value"] is value
    assert call_kwargs["seq_len"] is seq_lens_cpu
    assert call_kwargs["num_heads"] == 4
    assert call_kwargs["num_kv_heads"] == 4
    assert call_kwargs["scale_value"] == 0.125
    torch.testing.assert_close(output, query + 1)


def test_a5_npu_flash_attention_uses_python_sequence_lengths():
    query = torch.randn(5, 4, 64, dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    seq_lens_cpu = torch.tensor([2, 3], dtype=torch.int32)
    expected = torch.randn_like(query)

    with mock.patch(
        "vllm_ascend.device.device_op.torch_npu.npu_fusion_attention",
        return_value=(expected,),
    ) as mock_fusion_attention:
        output = A5DeviceAdaptor.npu_flash_attention(
            query=query,
            key=key,
            value=value,
            seq_lens_cpu=seq_lens_cpu,
            head_num=4,
            scale_value=0.125,
            num_kv_heads=4,
        )

    assert output is expected
    call_kwargs = mock_fusion_attention.call_args.kwargs
    assert call_kwargs["actual_seq_qlen"] == [2, 5]
    assert all(isinstance(seq_len, int) for seq_len in call_kwargs["actual_seq_qlen"])
    assert call_kwargs["actual_seq_kvlen"] is call_kwargs["actual_seq_qlen"]
