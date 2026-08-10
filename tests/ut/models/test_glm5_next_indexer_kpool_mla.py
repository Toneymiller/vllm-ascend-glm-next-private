# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import vllm.v1.core.kv_cache_utils as upstream_kv_cache_utils
from transformers import AutoConfig
from vllm import ModelRegistry
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.models.config import HybridAttentionMambaModelConfig
from vllm.transformers_utils.model_arch_config_convertor import (
    MODEL_ARCH_CONFIG_CONVERTORS,
)
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.kv_cache_interface import (
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker import mamba_utils
from vllm.v1.worker.utils import select_common_block_size

from vllm_ascend.attention.indexer_kpool_mla_v1 import (
    AscendIndexerKPoolBackend,
    AscendIndexerKPoolMetadataBuilder,
    AscendIndexerKPoolMLABackend,
    AscendIndexerKPoolMLAImpl,
    AscendIndexerKPoolMLAMetadataBuilder,
    AscendIndexerKPoolStateBackend,
    AscendIndexerKPoolStateMetadataBuilder,
)
from vllm_ascend.attention.sfa_v1 import AscendSFAMetadataBuilder
from vllm_ascend.core.kv_cache_interface import (
    AscendIndexerKPoolStateSpec,
    format_indexer_kpool_slot_mapping,
)
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.models import register_model as register_ascend_models
from vllm_ascend.models.glm5_next import (
    GLM5_TRANSFORMERS_INTERNAL_WEIGHTS_MAPPER,
    AscendGlm5NextCompressorStateCache,
    AscendGlm5NextGatedRMSNormParams,
    AscendGlm5NextIndexer,
    AscendGlm5NextIndexerKPoolCache,
    AscendSparseAttnIndexerKpool,
    SparseAttnIndexerKpool,
)
from vllm_ascend.ops.glm5_next_kpool_compress import glm5_next_kpool_compress_and_write_cache
from vllm_ascend.ops.glm5_next_lightning_indexer import glm5_next_lightning_indexer
from vllm_ascend.ops.indexer_kpool_mla import (
    AscendIndexerKPoolMLAAttention,
    IndexerKPoolMLACacheLayer,
    _collect_cache_metadata,
)
from vllm_ascend.ops.triton.kda.kda import fused_kda_gate
from vllm_ascend.patch.platform.patch_glm5_next_config import (
    Glm5NextModelArchConfigConvertor,
)
from vllm_ascend.patch.platform.patch_kv_cache_utils import (
    _create_glm5_attention_groups,
    _create_mamba_groups,
    _get_glm5_cache_layout,
    _get_kv_cache_config_deepseek_v4,
    _max_memory_usage_bytes_from_groups,
    _max_memory_usage_pages,
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
)
from vllm_ascend.patch.platform.patch_mamba_config import (
    GLM5_KERNEL_BLOCK_SIZE,
    _get_mamba_target_page_size,
    _is_glm5_next_model,
)
from vllm_ascend.patch.worker.patch_process_weights_after_loading import (
    _is_ascend_attention,
)
from vllm_ascend.transformers_utils.configs.glm5_next import (
    Glm5NextConfig,
    Glm5NextTextConfig,
    Glm5NextVisionConfig,
)


def test_indexer_kpool_mla_specs_allocate_one_physical_vector_per_role():
    mla_spec = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        model_version="glm5_next",
    )
    indexer_spec = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        compress_ratio=4,
        model_version="glm5_next",
    )
    state_spec = AscendIndexerKPoolStateSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=256,
        dtype=torch.bfloat16,
        sliding_window=4,
        cache_role="indexer_state",
    )

    assert mla_spec.page_size_bytes == 128 * (512 + 64) * 2
    assert indexer_spec.storage_block_size == 32
    assert indexer_spec.page_size_bytes == 32 * 128 * 2
    assert state_spec.page_size_bytes == 4 * 256 * 2
    assert state_spec.sliding_window == 4


def test_glm5_allocator_patches_the_installed_upstream_entrypoint():
    allocator_names = (
        "_get_kv_cache_config_deepseek_v4",
        "_get_kv_cache_config_packed",
    )
    installed_names = [
        name for name in allocator_names if hasattr(upstream_kv_cache_utils, name)
    ]

    assert installed_names
    for name in installed_names:
        assert (
            getattr(upstream_kv_cache_utils, name)
            is _get_kv_cache_config_deepseek_v4
        )


def test_indexer_kpool_mla_cache_roles_expose_v023_prefill_backend_sentinel():
    """Only the executable MLA cache passes through MLACommonMetadataBuilder."""
    assert IndexerKPoolMLACacheLayer.cache_role == "kv"
    assert IndexerKPoolMLACacheLayer.prefill_backend is None
    assert not hasattr(AscendGlm5NextIndexerKPoolCache, "prefill_backend")


def test_indexer_kpool_mla_cache_spec_stays_bf16_512_for_sharedkv():
    layer = object.__new__(IndexerKPoolMLACacheLayer)
    layer.kv_cache_dtype = "bfloat16"
    layer.block_size = 128
    layer.head_size = 512

    spec = layer.get_kv_cache_spec(SimpleNamespace(model_config=None))

    assert spec.block_size == 128
    assert spec.head_size == 512
    assert spec.dtype == torch.bfloat16


def test_indexer_kpool_mla_supports_uniform_decode_aclgraph():
    for speculative_config in (
        None,
        SimpleNamespace(num_speculative_tokens=5),
    ):
        assert (
            AscendIndexerKPoolMLAMetadataBuilder.get_cudagraph_support(
                SimpleNamespace(speculative_config=speculative_config),
                SimpleNamespace(),
            )
            is AttentionCGSupport.UNIFORM_BATCH
        )


def test_indexer_kpool_mla_a5_metadata_refreshes_graph_stable_buffer():
    builder = object.__new__(AscendIndexerKPoolMLAMetadataBuilder)
    builder._sas_metadata_buffer = torch.zeros(1024, dtype=torch.int32)
    builder._spec_sas_metadata_buffers = None
    builder._seqused_q = torch.empty(0, dtype=torch.int32)
    builder.model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(
            num_attention_heads=32,
            kv_lora_rank=512,
            index_topk=512,
            index_kpool=4,
        )
    )
    builder.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=4))
    common_metadata = SimpleNamespace(
        num_reqs=2,
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        seq_lens=torch.tensor([17, 33], dtype=torch.int32),
    )
    generated_metadata = torch.arange(1024, dtype=torch.int32)
    metadata_op = MagicMock(return_value=generated_metadata)
    base_metadata = SimpleNamespace(cache_role="", sas_metadata=None)

    with (
        patch.object(
            AscendSFAMetadataBuilder,
            "_build",
            return_value=base_metadata,
        ),
        patch.object(
            DeviceOperator,
            "supports_sharedkv_indexer_kpool_mla",
            return_value=True,
        ),
        patch.object(
            DeviceOperator,
            "get_sparse_attention_metadata_kwargs_indexer_kpool_mla",
            return_value={"device": "cpu"},
        ),
        patch.object(
            DeviceOperator,
            "get_sparse_attention_metadata_op_indexer_kpool_mla",
            return_value=metadata_op,
        ),
    ):
        metadata = builder._build(common_metadata)

    assert metadata.cache_role == "kv"
    assert metadata.sas_metadata is builder._sas_metadata_buffer
    assert metadata.sas_metadata.data_ptr() == builder._sas_metadata_buffer.data_ptr()
    assert metadata.query_start_loc is common_metadata.query_start_loc
    torch.testing.assert_close(metadata.sas_metadata, generated_metadata)
    metadata_op.assert_called_once()
    kwargs = metadata_op.call_args.kwargs
    assert kwargs["num_heads_q"] == 8
    assert kwargs["head_dim"] == 512
    assert kwargs["cmp_topk"] == 515
    assert kwargs["cmp_ratio"] == 1
    assert not kwargs["has_ori_kv"]
    assert kwargs["has_cmp_kv"]


def test_indexer_kpool_cache_only_builders_do_not_disable_uniform_decode_aclgraph():
    for builder_cls in (
        AscendIndexerKPoolMetadataBuilder,
        AscendIndexerKPoolStateMetadataBuilder,
    ):
        assert (
            builder_cls.get_cudagraph_support(
                SimpleNamespace(),
                SimpleNamespace(),
            )
            is AttentionCGSupport.UNIFORM_BATCH
        )


def test_indexer_kpool_mla_participates_in_post_load_weight_processing():
    wrapper = object.__new__(AscendIndexerKPoolMLAAttention)
    assert _is_ascend_attention(wrapper)


def test_indexer_kpool_mla_collects_metadata_by_exact_cache_layer_name():
    prefix = "model.layers.0.self_attn"
    cache_layers = (
        SimpleNamespace(layer_name=f"{prefix}.attn"),
        SimpleNamespace(prefix=f"{prefix}.indexer.compressor.state_cache"),
        SimpleNamespace(prefix=f"{prefix}.indexer.k_cache"),
    )
    expected = tuple(SimpleNamespace(cache_role=role) for role in ("kv", "indexer_state", "indexer"))
    metadata = {
        f"{prefix}.attn": expected[0],
        f"{prefix}.indexer.compressor.state_cache": expected[1],
        f"{prefix}.indexer.k_cache": expected[2],
        # 同一 attention 前缀下的其他 metadata 不能传给组合实现。
        f"{prefix}.unrelated_cache": SimpleNamespace(cache_role="kv"),
    }

    wrapper = SimpleNamespace(prefix=prefix, cache_layers=cache_layers)
    assert _collect_cache_metadata(wrapper, metadata) == expected


def test_indexer_kpool_state_uses_independent_backend_for_four_token_pages():
    assert AscendGlm5NextCompressorStateCache.get_attn_backend(None) is AscendIndexerKPoolStateBackend
    assert select_common_block_size(4, [AscendIndexerKPoolStateBackend]) == 4
    assert AscendIndexerKPoolStateBackend.get_kv_cache_shape(8, 4, 1, 256) == (8, 4, 256)


def test_indexer_kpool_cache_uses_minimal_independent_metadata_builder():
    spec = MLAAttentionSpec(
        block_size=384,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        compress_ratio=4,
        model_version="glm5_next",
    )
    builder = AscendIndexerKPoolMetadataBuilder(
        spec,
        ["model.layers.0.self_attn.indexer.k_cache"],
        SimpleNamespace(
            scheduler_config=SimpleNamespace(
                max_num_batched_tokens=16,
                max_num_seqs=4,
            ),
            model_config=SimpleNamespace(max_model_len=768),
        ),
        torch.device("cpu"),
    )
    common_metadata = SimpleNamespace(
        num_reqs=1,
        num_input_tokens=4,
        positions=torch.tensor([380, 381, 382, 383]),
        slot_mapping=torch.tensor([7 * 384 + offset for offset in range(380, 384)]),
        seq_lens=torch.tensor([384], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([384], dtype=torch.int32),
        seq_lens_cpu=None,
        block_table_tensor=torch.tensor(
            [[21, 22, 23, 6, 7, 8]],
            dtype=torch.int32,
        ),
    )

    metadata = builder.build(0, common_metadata)

    assert not issubclass(AscendIndexerKPoolMetadataBuilder, AscendSFAMetadataBuilder)
    assert AscendGlm5NextIndexerKPoolCache.get_attn_backend(None) is AscendIndexerKPoolBackend
    assert (
        select_common_block_size(
            384,
            [AscendIndexerKPoolMLABackend, AscendIndexerKPoolBackend],
        )
        == 128
    )
    assert metadata.block_size == 96
    assert metadata.block_table.tolist() == [[7, 2]]
    assert metadata.slot_mapping.tolist() == [-1, -1, -1, 7 * 96 + 95]
    assert metadata.seq_lens.tolist() == [96]
    assert metadata.seq_lens_cpu.tolist() == [96]

    first_slot_mapping_ptr = metadata.slot_mapping.data_ptr()
    first_seq_lens_ptr = metadata.seq_lens.data_ptr()
    first_block_table_ptr = metadata.block_table.data_ptr()
    common_metadata.positions = torch.tensor([4, 5, 6, 7])
    common_metadata.slot_mapping = torch.tensor(
        [3 * 384 + offset for offset in range(4, 8)]
    )
    common_metadata.seq_lens = torch.tensor([8], dtype=torch.int32)
    common_metadata._seq_lens_cpu = torch.tensor([8], dtype=torch.int32)
    common_metadata.block_table_tensor = torch.tensor([[9, 10, 11]], dtype=torch.int32)

    replay_metadata = builder.build(0, common_metadata)

    assert replay_metadata.slot_mapping.data_ptr() == first_slot_mapping_ptr
    assert replay_metadata.seq_lens.data_ptr() == first_seq_lens_ptr
    assert replay_metadata.block_table.data_ptr() == first_block_table_ptr
    assert replay_metadata.slot_mapping.tolist() == [-1, -1, -1, 3 * 96 + 1]
    assert replay_metadata.seq_lens.tolist() == [2]
    assert replay_metadata.block_table.tolist() == [[3]]


def test_glm5_latest_config_schema_drives_attention_and_mlp_layout():
    config = Glm5NextTextConfig(
        num_hidden_layers=4,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "deepseek_sparse_attention",
        ],
        mlp_layer_types=["dense", "dense", "dense", "sparse"],
        linear_head_dim=128,
        linear_num_heads=64,
        linear_conv_kernel_dim=4,
        linear_lower_bound=-5.0,
        qk_rope_head_dim=0,
        index_kpool=4,
    )

    assert config.model_type == "glm5_next_text"
    assert config.layers_block_type == [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "attention",
    ]
    assert config.mlp_layer_types == ["dense", "dense", "dense", "sparse"]
    assert config.linear_head_dim == 128
    assert config.linear_num_heads == 64
    assert config.linear_conv_kernel_dim == 4
    assert config.linear_lower_bound == -5.0
    assert config.qk_rope_head_dim == 0
    assert config.index_kpool == 4


def test_glm5_model_arch_config_uses_mla_cache_dimensions():
    text_config = Glm5NextTextConfig(
        head_dim=0,
        kv_lora_rank=512,
        qk_rope_head_dim=0,
    )
    config = Glm5NextConfig(text_config=text_config.to_dict())
    convertor_cls = MODEL_ARCH_CONFIG_CONVERTORS["glm5_next"]
    convertor = convertor_cls(config, config.text_config)
    model_arch_config = convertor.convert()

    assert convertor_cls is Glm5NextModelArchConfigConvertor
    assert convertor.is_deepseek_mla()
    assert convertor.get_head_size() == 512
    assert model_arch_config.is_deepseek_mla
    assert model_arch_config.head_size == 512


def test_glm5_kda_page_is_padded_to_contiguous_main_mla_page():
    target_page_size = _get_mamba_target_page_size(
        is_glm5_next=True,
        attn_page_size=393216,
        mamba_raw_size=271360,
        conv_block_page_size=9216,
    )

    assert target_page_size == 393216


def test_glm5_logical_block_makes_main_mla_page_contiguous():
    logical_block_size = 384
    physical_page_size = 393216
    mla_payload_size = (
        logical_block_size
        * 1
        * 512
        * torch.bfloat16.itemsize
    )

    assert GLM5_KERNEL_BLOCK_SIZE == 128
    assert logical_block_size % GLM5_KERNEL_BLOCK_SIZE == 0
    assert mla_payload_size == physical_page_size


@pytest.mark.parametrize("model_type", ["glm5_next", "glm5_next_text"])
def test_glm5_model_type_detection_accepts_outer_and_text_configs(
    model_type: str,
):
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type=model_type),
        hf_text_config=SimpleNamespace(model_type=model_type),
    )

    assert _is_glm5_next_model(model_config)


@patch(
    "vllm_ascend.patch.platform.patch_mamba_config."
    "MambaModelConfig.verify_and_update_config"
)
@patch(
    "vllm_ascend.patch.platform.patch_mamba_config."
    "ModelRegistry.resolve_model_cls"
)
def test_glm5_mamba_config_aligns_prefix_cache_to_contiguous_main_page(
    mock_resolve_model_cls,
    mock_mamba_verify,
):
    class FakeGlm5Model:
        @classmethod
        def get_mamba_state_shape_from_config(cls, vllm_config):
            del vllm_config
            return (3, 1536), (4, 128, 128)

        @classmethod
        def get_mamba_state_dtype_from_config(cls, vllm_config):
            del vllm_config
            return torch.bfloat16, torch.float32

    mock_resolve_model_cls.return_value = FakeGlm5Model, None
    cache_config = SimpleNamespace(
        block_size=128,
        cache_dtype="auto",
        mamba_page_size_padded=None,
        enable_prefix_caching=True,
        mamba_cache_mode="align",
        mamba_block_size=None,
    )
    model_config = SimpleNamespace(
        architecture="Glm5NextForCausalLM",
        dtype=torch.bfloat16,
        hf_config=SimpleNamespace(model_type="glm5_next"),
        hf_text_config=SimpleNamespace(
            kv_lora_rank=512,
            qk_rope_head_dim=0,
        ),
        max_model_len=131072,
        use_mla=True,
        get_num_kv_heads=lambda parallel_config: 1,
    )
    vllm_config = SimpleNamespace(
        cache_config=cache_config,
        model_config=model_config,
        parallel_config=SimpleNamespace(),
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=False,
        ),
        speculative_config=None,
        kv_transfer_config=None,
    )

    HybridAttentionMambaModelConfig.verify_and_update_config(vllm_config)

    mock_mamba_verify.assert_called_once_with(vllm_config)
    assert cache_config.block_size == 384
    assert cache_config.mamba_page_size_padded == 393216
    assert cache_config.mamba_block_size == 384


def test_glm5_mamba_groups_use_top_level_mamba_spec():
    mamba_specs = {
        f"layer.{layer_idx}": MambaSpec(
            block_size=256,
            shapes=((8, 128, 128), (6, 3, 128)),
            dtypes=(torch.bfloat16, torch.float32),
            page_size_padded=271360,
        )
        for layer_idx in (0, 3)
    }

    groups = _create_mamba_groups(
        mamba_specs,
        [["layer.0", "layer.3"]],
    )

    assert len(groups) == 1
    assert isinstance(groups[0].kv_cache_spec, MambaSpec)
    assert groups[0].kv_cache_spec == mamba_specs["layer.0"]
    assert groups[0].layer_names == ["layer.0", "layer.3"]


def test_glm5_auto_config_reads_registered_nested_text_config(tmp_path):
    config_dict = {
        "model_type": "glm5_next",
        "architectures": ["Glm5NextForConditionalGeneration"],
        "text_config": {
            "model_type": "glm5_next_text",
            "num_hidden_layers": 4,
            "num_experts_per_tok": 8,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "deepseek_sparse_attention",
            ],
            "mlp_layer_types": ["dense", "dense", "dense", "sparse"],
            "index_kpool": 4,
            "index_topk": 2048,
        },
        "vision_config": {
            "model_type": "glm_ocr_vision",
            "depth": 24,
            "hidden_size": 1024,
            "num_heads": 16,
            "patch_size": 14,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 4096,
            "projection_intermediate_size": 10240,
        },
        "image_token_id": 154854,
        "video_token_id": 154855,
        "image_start_token_id": 154830,
        "image_end_token_id": 154831,
        "video_start_token_id": 154832,
        "video_end_token_id": 154833,
    }
    (tmp_path / "config.json").write_text(json.dumps(config_dict))

    config = AutoConfig.from_pretrained(tmp_path)

    assert isinstance(config, Glm5NextConfig)
    assert isinstance(config.text_config, Glm5NextTextConfig)
    assert isinstance(config.vision_config, Glm5NextVisionConfig)
    assert config.model_type == "glm5_next"
    assert config.architectures == ["Glm5NextForConditionalGeneration"]
    assert config.text_config.model_type == "glm5_next_text"
    assert config.text_config.num_experts_per_token == 8
    assert config.text_config.index_kpool == 4
    assert config.text_config.index_topk == 2048
    assert config.vision_config.depth == 24
    assert config.vision_config.hidden_size == 1024
    assert config.vision_config.num_heads == 16
    assert config.vision_config.patch_size == 14
    assert config.vision_config.spatial_merge_size == 2
    assert config.vision_config.temporal_patch_size == 2
    assert config.vision_config.out_hidden_size == 4096
    assert config.vision_config.projection_intermediate_size == 10240
    assert config.image_token_id == 154854
    assert config.video_token_id == 154855
    assert config.image_start_token_id == 154830
    assert config.image_end_token_id == 154831
    assert config.video_start_token_id == 154832
    assert config.video_end_token_id == 154833


def test_glm5_conditional_generation_architecture_uses_text_only_ascend_model(
    monkeypatch,
):
    registered_models = {}
    monkeypatch.setattr(
        ModelRegistry,
        "register_model",
        lambda architecture, model_cls: registered_models.__setitem__(
            architecture,
            model_cls,
        ),
    )

    register_ascend_models()

    assert registered_models["Glm5NextForConditionalGeneration"] == (
        "vllm_ascend.models.glm5_next:AscendGlm5NextForCausalLM"
    )
    assert registered_models["Glm5NextForConditionalGeneration"] == registered_models["Glm5NextForCausalLM"]


def test_glm5_transformers_internal_weight_names_are_mapped():
    names = GLM5_TRANSFORMERS_INTERNAL_WEIGHTS_MAPPER.apply_list(
        [
            "layers.0.self_attn.forget_gate.A_log",
            "layers.0.self_attn.forget_gate.dt_bias",
            "layers.0.self_attn.o_norm.weight",
            "layers.38.attn_hc.fn",
            "layers.38.attn_hc.base",
            "layers.38.attn_hc.scale",
            "layers.38.ffn_hc.fn",
            "layers.38.ffn_hc.base",
            "layers.38.ffn_hc.scale",
        ]
    )

    assert names == [
        "layers.0.self_attn.A_log",
        "layers.0.self_attn.dt_bias",
        "layers.0.self_attn.o_norm.weight",
        "layers.38.hc_attn_fn",
        "layers.38.hc_attn_base",
        "layers.38.hc_attn_scale",
        "layers.38.hc_ffn_fn",
        "layers.38.hc_ffn_base",
        "layers.38.hc_ffn_scale",
    ]


def test_glm5_gated_rms_norm_matches_transformers_state_dict_name():
    o_norm = AscendGlm5NextGatedRMSNormParams(hidden_size=8)

    assert list(dict(o_norm.named_parameters())) == ["weight"]
    assert list(o_norm.state_dict()) == ["weight"]
    assert torch.count_nonzero(o_norm.bias) == 0


def test_glm5_kda_gate_exposes_bounded_gate_parameters():
    parameters = inspect.signature(fused_kda_gate).parameters
    assert parameters["safe_gate"].default is False
    assert parameters["lower_bound"].default == -5.0


def test_indexer_kpool_mla_nope_rope_single_is_identity():
    hidden = torch.randn(2, 3)
    result = AscendIndexerKPoolMLAImpl.rope_single(
        SimpleNamespace(qk_rope_head_dim=0),
        hidden,
        torch.empty(2, 1, 1, 0),
        torch.empty(2, 1, 1, 0),
    )

    assert result is hidden


def test_indexer_kpool_mla_nope_packs_only_latent_cache_values():
    op = SimpleNamespace(
        kv_lora_rank=4,
        qk_rope_head_dim=0,
    )
    kv_c = torch.arange(12, dtype=torch.bfloat16).reshape(3, 1, 4)
    k_pe = torch.empty(3, 1, 0, dtype=torch.bfloat16)

    result = AscendIndexerKPoolMLAImpl._pack_mla_cache_values(
        op,
        kv_c,
        k_pe,
        num_tokens=3,
    )

    assert result.shape == (3, 4)
    assert torch.equal(result, kv_c.reshape(3, 4))


def test_indexer_kpool_mla_state_block_table_maps_request_tail_pages():
    op = SimpleNamespace(head_dim=1)
    raw_state_cache = torch.zeros(48, dtype=torch.bfloat16)
    state_cache = torch.as_strided(
        raw_state_cache,
        size=(4, 4, 2),
        stride=(12, 2, 1),
    )
    state_cache[2] = torch.tensor([[20.0, 200.0], [21.0, 201.0], [22.0, 202.0], [23.0, 203.0]])
    state_cache[3] = torch.tensor([[30.0, 300.0], [31.0, 301.0], [32.0, 302.0], [33.0, 303.0]])
    metadata = SimpleNamespace(
        block_size=4,
        # Both requests gather absolute logical positions 4..7, but their
        # current tail pages are owned by different physical blocks. The old
        # page entry can be released/null without changing the logical index.
        block_table=torch.tensor([[0, 2], [1, 3]], dtype=torch.int32),
    )

    result = SparseAttnIndexerKpool._gather_compressor_state(
        op,
        state_cache,
        metadata,
        end_positions=torch.tensor([7, 7]),
        request_ids=torch.tensor([0, 1]),
        index_kpool=4,
    )

    assert result[:, :, 0].tolist() == [
        [20.0, 21.0, 22.0, 23.0],
        [30.0, 31.0, 32.0, 33.0],
    ]


def _make_glm5_cache_groups(
    *,
    include_mtp: bool,
    num_speculative_tokens: int = 0,
):
    mla_layer_indices = list(range(3, 44, 4))
    if include_mtp:
        mla_layer_indices.append(45)
    mamba_layer_indices = [idx for idx in range(45) if idx % 4 != 3]
    specs = {}
    for layer_idx in mla_layer_indices:
        prefix = f"model.layers.{layer_idx}.self_attn"
        specs[f"{prefix}.attn"] = MLAAttentionSpec(
            block_size=384,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
            model_version="glm5_next",
        )
        specs[f"{prefix}.indexer.k_cache"] = MLAAttentionSpec(
            block_size=384,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            compress_ratio=16,
            model_version="glm5_next",
        )
        specs[f"{prefix}.indexer.compressor.state_cache"] = AscendIndexerKPoolStateSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=256,
            dtype=torch.bfloat16,
            sliding_window=16,
            cache_role="indexer_state",
            model_version="glm5_next",
        )
    for layer_idx in mamba_layer_indices:
        specs[f"model.layers.{layer_idx}.mamba"] = MambaSpec(
            block_size=384,
            shapes=(
                (3 + num_speculative_tokens, 1536),
                (4, 128, 128),
            ),
            dtypes=(torch.bfloat16, torch.float32),
            num_speculative_blocks=num_speculative_tokens,
        )

    # Match NPUModelRunner.get_kv_cache_spec: attention/cache-role specs are
    # collected first and all Mamba specs are appended afterward. Grouping
    # must recover model order rather than depend on this insertion order.
    first_mamba_pos = next(
        idx
        for idx, spec in enumerate(specs.values())
        if isinstance(spec, MambaSpec)
    )
    assert all(
        isinstance(spec, MambaSpec)
        for spec in list(specs.values())[first_mamba_pos:]
    )
    groups = get_kv_cache_groups(SimpleNamespace(), specs)
    return specs, groups


def test_glm5_main_and_indexer_share_one_full_history_group():
    _, groups = _make_glm5_cache_groups(include_mtp=False)
    layout = _get_glm5_cache_layout(groups)
    assert layout is not None

    assert len(groups) == 5
    assert groups[0] is layout.full_group
    assert groups[1] is layout.state_group
    assert len(layout.full_group.layer_names) == 22
    assert len(layout.state_group.layer_names) == 11
    assert [len(group.layer_names) for group in layout.mamba_groups] == [12, 11, 11]
    assert all(
        isinstance(group.kv_cache_spec, MambaSpec)
        for group in layout.mamba_groups
    )
    assert [group.layer_names for group in layout.mamba_groups] == [
        [f"model.layers.{layer_idx}.mamba" for layer_idx in range(0, 45, 4)],
        [f"model.layers.{layer_idx}.mamba" for layer_idx in range(1, 45, 4)],
        [f"model.layers.{layer_idx}.mamba" for layer_idx in range(2, 45, 4)],
    ]

    full_specs = layout.full_group.kv_cache_spec.kv_cache_specs
    full_ratios = [full_specs[name].compress_ratio for name in layout.full_group.layer_names]
    assert full_ratios == [ratio for _ in range(11) for ratio in (1, 16)]
    assert layout.full_group.kv_cache_spec.block_size == 384
    assert layout.state_group.kv_cache_spec.block_size == 16
    assert {
        full_specs[name].storage_block_size
        for name in layout.indexer_names
    } == {24}

    layer_to_group_id = {
        layer_name: group_id
        for group_id, group in enumerate(groups)
        for layer_name in group.layer_names
    }
    for layer_idx in range(3, 44, 4):
        prefix = f"model.layers.{layer_idx}.self_attn"
        assert layer_to_group_id[f"{prefix}.attn"] == 0
        assert layer_to_group_id[f"{prefix}.indexer.k_cache"] == 0
        assert layer_to_group_id[f"{prefix}.indexer.compressor.state_cache"] == 1
    assert {
        layer_to_group_id[name]
        for name in layer_to_group_id
        if name.endswith(".mamba")
    } == {2, 3, 4}

    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=1024),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
    )
    # One FullAttentionManager/block table reserves three 384-token IDs for
    # both physical caches instead of reserving three IDs per cache role.
    assert layout.full_group.kv_cache_spec.max_memory_usage_pages(vllm_config) == 3
    first_main, first_indexer = layout.full_group.layer_names[:2]
    assert [
        full_specs[name].max_memory_usage_bytes(vllm_config)
        // full_specs[name].page_size_bytes
        for name in (first_main, first_indexer)
    ] == [3, 3]


@pytest.mark.parametrize(
    ("compress_ratio", "state_block_size", "state_window", "error"),
    [
        (10, 10, 10, "must be divisible"),
        (4, 8, 8, "must equal the paired compression ratio"),
        (4, 4, 8, "must equal the paired compression ratio"),
    ],
)
def test_glm5_combined_group_validates_compression_geometry(
    compress_ratio: int,
    state_block_size: int,
    state_window: int,
    error: str,
):
    specs = {
        "model.layers.0.self_attn.attn": MLAAttentionSpec(
            block_size=384,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
            model_version="glm5_next",
        ),
        "model.layers.0.self_attn.indexer.k_cache": MLAAttentionSpec(
            block_size=384,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            compress_ratio=compress_ratio,
            model_version="glm5_next",
        ),
        "model.layers.0.self_attn.indexer.compressor.state_cache": AscendIndexerKPoolStateSpec(
            block_size=state_block_size,
            num_kv_heads=1,
            head_size=256,
            dtype=torch.bfloat16,
            sliding_window=state_window,
            cache_role="indexer_state",
            model_version="glm5_next",
        ),
    }

    with pytest.raises(ValueError, match=error):
        _create_glm5_attention_groups(specs)


def test_glm5_target_allocator_uses_twelve_large_and_eleven_small_tensors():
    specs, groups = _make_glm5_cache_groups(include_mtp=False)
    layout = _get_glm5_cache_layout(groups)
    assert layout is not None
    assert [len(group.layer_names) for group in layout.mamba_groups] == [12, 11, 11]
    assert layout.main_slot_count == 12
    assert layout.small_slot_count == 11
    assert layout.main_page_size == max(
        spec.real_page_size_bytes
        for spec in specs.values()
        if isinstance(spec, (MLAAttentionSpec, MambaSpec)) and getattr(spec, "compress_ratio", 1) == 1
    )
    assert layout.main_page_size == 384 * 512 * 2
    assert layout.small_page_size == 16 * 256 * 2

    bytes_per_block = 12 * layout.main_page_size + 11 * layout.small_page_size
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(num_gpu_blocks_override=None),
    )
    config = get_kv_cache_config_from_groups(
        vllm_config,
        groups,
        available_memory=bytes_per_block * 3,
    )
    num_blocks = config.num_blocks
    tensors = config.kv_cache_tensors

    assert num_blocks == 3
    assert config.kv_cache_groups is groups
    mamba_group_ids, mamba_spec = mamba_utils.get_mamba_groups(config)
    assert mamba_group_ids == [2, 3, 4]
    assert isinstance(mamba_spec, MambaSpec)
    assert config.has_mamba_layers
    assert config.needs_kv_cache_zeroing
    assert len(tensors) == 23
    assert all(tensor.size == layout.main_page_size * 3 for tensor in tensors[:12])
    assert all(tensor.size == layout.small_page_size * 3 for tensor in tensors[12:])
    assert tensors[0].shared_by == [
        "model.layers.3.self_attn.attn",
        "model.layers.0.mamba",
        "model.layers.1.mamba",
        "model.layers.2.mamba",
    ]
    assert tensors[11].shared_by == ["model.layers.44.mamba"]
    assert tensors[12].shared_by == [
        "model.layers.3.self_attn.indexer.k_cache",
        "model.layers.3.self_attn.indexer.compressor.state_cache",
    ]


def test_glm5_combined_layout_adds_mtp_indexer_state_small_page():
    _, groups = _make_glm5_cache_groups(
        include_mtp=True,
        num_speculative_tokens=5,
    )
    layout = _get_glm5_cache_layout(groups)
    assert layout is not None
    assert layout.main_slot_count == 12
    assert layout.small_slot_count == 12

    bytes_per_block = 12 * layout.main_page_size + 12 * layout.small_page_size
    num_blocks, tensors = _get_kv_cache_config_deepseek_v4(
        SimpleNamespace(cache_config=SimpleNamespace(num_gpu_blocks_override=None)),
        groups,
        available_memory=bytes_per_block * 2,
    )

    assert num_blocks == 2
    assert len(tensors) == 24
    assert tensors[11].shared_by == [
        "model.layers.45.self_attn.attn",
        "model.layers.44.mamba",
    ]
    assert tensors[-1].shared_by == [
        "model.layers.45.self_attn.indexer.k_cache",
        "model.layers.45.self_attn.indexer.compressor.state_cache",
    ]


def test_glm5_target_mtp5_keeps_kda_padded_to_main_page():
    specs, groups = _make_glm5_cache_groups(
        include_mtp=False,
        num_speculative_tokens=5,
    )
    layout = _get_glm5_cache_layout(groups)
    assert layout is not None

    assert layout.main_slot_count == 12
    assert layout.small_slot_count == 11
    assert layout.main_page_size == 384 * 512 * 2
    assert {
        spec.num_speculative_blocks
        for spec in specs.values()
        if isinstance(spec, MambaSpec)
    } == {5}


def test_indexer_kpool_mla_standalone_mtp_allocator_uses_two_page_classes():
    specs = {
        "layer.attn": MLAAttentionSpec(
            block_size=384,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
            model_version="glm5_next",
        ),
        "layer.indexer.k_cache": MLAAttentionSpec(
            block_size=384,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            compress_ratio=4,
            model_version="glm5_next",
        ),
        "layer.indexer.compressor.state_cache": AscendIndexerKPoolStateSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=256,
            dtype=torch.bfloat16,
            sliding_window=4,
            cache_role="indexer_state",
            model_version="glm5_next",
        ),
    }
    groups = get_kv_cache_groups(SimpleNamespace(), specs)
    assert len(groups) == 2
    assert groups[0].layer_names == ["layer.attn", "layer.indexer.k_cache"]
    assert groups[1].layer_names == ["layer.indexer.compressor.state_cache"]
    main_page_size = specs["layer.attn"].page_size_bytes
    small_page_size = specs["layer.indexer.k_cache"].page_size_bytes
    assert small_page_size == specs["layer.indexer.compressor.state_cache"].page_size_bytes
    bytes_per_block = main_page_size + small_page_size
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(num_gpu_blocks_override=None),
    )

    num_blocks, tensors = _get_kv_cache_config_deepseek_v4(
        vllm_config,
        groups,
        available_memory=bytes_per_block * 3,
    )

    assert num_blocks == 3
    assert len(tensors) == 2
    assert tensors[0].shared_by == ["layer.attn"]
    assert tensors[0].size == main_page_size * num_blocks
    assert tensors[1].shared_by == [
        "layer.indexer.k_cache",
        "layer.indexer.compressor.state_cache",
    ]
    assert tensors[1].size == small_page_size * num_blocks


def test_glm5_memory_accounting_counts_combined_full_group_once():
    _, groups = _make_glm5_cache_groups(include_mtp=False)
    layout = _get_glm5_cache_layout(groups)
    assert layout is not None
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=1024),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
    )
    blocks_needed = sum(
        _max_memory_usage_pages(vllm_config, group.kv_cache_spec)
        for group in groups
    )
    full_blocks = layout.full_group.kv_cache_spec.max_memory_usage_pages(vllm_config)
    bytes_per_block = 12 * layout.main_page_size + 11 * layout.small_page_size

    assert _max_memory_usage_bytes_from_groups(vllm_config, groups) == (blocks_needed * bytes_per_block)
    assert _max_memory_usage_bytes_from_groups(vllm_config, groups) < (
        (blocks_needed + full_blocks) * bytes_per_block
    )


def test_indexer_kpool_mla_compressed_slot_mapping_only_writes_completed_pools():
    slots = torch.tensor([0, 14, 15, 16, 127, 128, 143, -1])
    positions = torch.tensor([0, 14, 15, 16, 127, 128, 143, 15])

    result = format_indexer_kpool_slot_mapping(
        slots,
        positions,
        logical_block_size=128,
        compress_ratio=16,
    )

    assert result.tolist() == [-1, -1, 0, -1, 7, -1, 8, -1]


def test_indexer_kpool_mla_compressed_slot_mapping_rejects_partial_block_pool():
    with pytest.raises(ValueError, match="must be divisible"):
        format_indexer_kpool_slot_mapping(
            torch.tensor([0]),
            torch.tensor([0]),
            logical_block_size=128,
            compress_ratio=10,
        )


def test_indexer_kpool_mla_expand_pools_to_tokens_matches_vllm_contract():
    pool_ids = torch.tensor(
        [
            [2, -1],
            [0, 1],
        ],
        dtype=torch.int32,
    )

    result = AscendSparseAttnIndexerKpool.expand_pools_to_tokens(
        pool_ids,
        pool_ids >= 0,
        topk=8,
        pool_size=4,
    )

    assert result.dtype == torch.int32
    assert result.tolist() == [
        [8, 9, 10, 11, -1, -1, -1, -1],
        [0, 1, 2, 3, 4, 5, 6, 7],
    ]


def test_indexer_kpool_mla_append_tail_to_topk_uses_per_query_lengths():
    history = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 15],
        ],
        dtype=torch.int32,
    )

    result = AscendSparseAttnIndexerKpool.append_tail_to_topk(
        history,
        seq_lens=torch.tensor([11, 16], dtype=torch.int32),
        pool_lens=torch.tensor([2, 4], dtype=torch.int32),
        pool_size=4,
    )

    assert result.tolist() == [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [8, 9, 10, 11, 12, 13, 14, 15, -1, -1, -1],
    ]


def test_indexer_kpool_mla_kpool_topk_requires_exact_pool_budget():
    with pytest.raises(ValueError, match="must be divisible"):
        AscendSparseAttnIndexerKpool.history_group_budget_for_topk(
            topk=10,
            pool_size=4,
        )


def test_indexer_kpool_mla_kpool_topk_budget_is_cached_during_initialization():
    op = AscendSparseAttnIndexerKpool(
        k_cache=SimpleNamespace(),
        quant_block_size=128,
        scale_fmt=None,
        topk_tokens=2048,
        head_dim=128,
        max_model_len=4096,
        max_total_seq_len=4096,
        topk_indices_buffer=None,
        state_cache=SimpleNamespace(compress_ratio=4),
        attn_layer_name="layer.attn",
    )

    assert op.pool_topk == 512


@patch("torch.ops._C_ascend.npu_lightning_indexer", create=True)
def test_indexer_kpool_mla_decode_topk_uses_graph_compatible_ascend_op(mock_lightning_indexer):
    expected = torch.tensor(
        [
            [[2, -1]],
            [[1, 0]],
        ],
        dtype=torch.int32,
    )
    mock_lightning_indexer.return_value = (expected, torch.empty(0))
    query = torch.zeros((2, 1, 4), dtype=torch.bfloat16)
    key = torch.zeros((3, 2, 1, 4), dtype=torch.bfloat16)
    weights = torch.ones((2, 1), dtype=torch.bfloat16)
    query_lens = torch.tensor([1, 2], dtype=torch.int32)
    key_lens = torch.tensor([3, 2], dtype=torch.int32)
    block_table = torch.tensor([[0, 1], [2, 0]], dtype=torch.int32)

    result = AscendSparseAttnIndexerKpool.indexer_kpool_topk_decode(
        query,
        key,
        weights,
        query_lens,
        key_lens,
        block_table,
        sparse_count=2,
    )

    torch.testing.assert_close(result, expected.squeeze(1))
    mock_lightning_indexer.assert_called_once_with(
        query=query,
        key=key,
        weights=weights,
        actual_seq_lengths_query=query_lens,
        actual_seq_lengths_key=key_lens,
        block_table=block_table,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=2,
        sparse_mode=3,
    )


@patch("vllm_ascend.attention.indexer_kpool_mla_v1.get_forward_context")
@patch("torch_npu.npu_scatter_nd_update_", create=True)
def test_indexer_kpool_mla_state_write_maps_negative_slots_to_restored_sentinel(
    mock_scatter,
    mock_get_forward_context,
):
    mock_get_forward_context.return_value = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.FULL)
    raw_cache = torch.zeros(80, dtype=torch.bfloat16)
    cache = torch.as_strided(
        raw_cache,
        size=(2, 4, 8),
        stride=(40, 8, 1),
    )
    slots = torch.tensor([6, -1], dtype=torch.int64)
    values = torch.ones((2, 1, 8), dtype=torch.bfloat16)

    AscendIndexerKPoolMLAImpl._store_indexer_cache(
        None,
        cache,
        slots,
        values,
    )

    mock_scatter.assert_not_called()
    assert cache[1, 2].tolist() == [1.0] * 8
    assert cache[0, 0].tolist() == [0.0] * 8
    assert raw_cache[32:40].tolist() == [0.0] * 8


@patch("vllm_ascend.attention.indexer_kpool_mla_v1.get_forward_context")
@patch("torch_npu.npu_scatter_nd_update_", create=True)
def test_indexer_kpool_mla_eager_state_write_uses_paged_indices(
    mock_scatter,
    mock_get_forward_context,
):
    mock_get_forward_context.return_value = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.NONE)
    raw_cache = torch.zeros(80, dtype=torch.bfloat16)
    cache = torch.as_strided(
        raw_cache,
        size=(2, 4, 8),
        stride=(40, 8, 1),
    )
    values = torch.arange(16, dtype=torch.bfloat16).view(2, 1, 8)

    AscendIndexerKPoolMLAImpl._store_indexer_cache(
        None,
        cache,
        torch.tensor([6, -1], dtype=torch.int64),
        values,
    )

    mock_scatter.assert_called_once()
    args = mock_scatter.call_args.args
    assert args[0] is cache
    torch.testing.assert_close(
        args[1],
        torch.tensor([[1, 2]], dtype=torch.int64),
    )
    torch.testing.assert_close(args[2], values[:1, 0])


@patch("vllm_ascend.attention.indexer_kpool_mla_v1.get_forward_context")
@patch("torch_npu.npu_scatter_nd_update_", create=True)
def test_indexer_kpool_mla_paged_write_preserves_physical_page_stride(
    mock_scatter,
    mock_get_forward_context,
):
    mock_get_forward_context.return_value = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.FULL)
    raw_cache = torch.zeros(16, dtype=torch.bfloat16)
    cache = torch.as_strided(
        raw_cache,
        size=(2, 2, 1, 2),
        stride=(8, 2, 2, 1),
    )
    values = torch.tensor(
        [[[1.0, 2.0]], [[9.0, 9.0]]],
        dtype=torch.bfloat16,
    )

    AscendIndexerKPoolMLAImpl._scatter_paged_cache(
        cache,
        torch.tensor([3, -1], dtype=torch.int64),
        values,
        block_size=2,
    )

    mock_scatter.assert_not_called()
    torch.testing.assert_close(cache[1, 1], values[0])
    torch.testing.assert_close(cache[0, 0], torch.zeros_like(cache[0, 0]))
    assert raw_cache[4:8].tolist() == [0.0] * 4


def test_glm5_indexer_paged_write_preserves_physical_page_stride():
    raw_cache = torch.zeros(16, dtype=torch.bfloat16)
    cache = torch.as_strided(
        raw_cache,
        size=(2, 2, 1, 2),
        stride=(8, 2, 2, 1),
    )
    values = torch.tensor(
        [[[3.0, 4.0]], [[9.0, 9.0]]],
        dtype=torch.bfloat16,
    )

    AscendSparseAttnIndexerKpool._scatter_paged_cache(
        cache,
        torch.tensor([2, -1], dtype=torch.int64),
        values,
        block_size=2,
    )

    torch.testing.assert_close(cache[1, 0], values[0])
    torch.testing.assert_close(cache[0, 0], torch.zeros_like(cache[0, 0]))
    assert raw_cache[4:8].tolist() == [0.0] * 4


@patch("vllm_ascend.models.glm5_next.get_forward_context")
@patch("torch.ops.vllm.glm5_next_lightning_indexer", create=True)
@patch(
    "torch.ops.vllm.glm5_next_kpool_compress_and_write_cache",
    create=True,
)
def test_glm5_indexer_eager_mtp_ignores_padded_input_rows(
    mock_compress,
    mock_lightning_indexer,
    mock_get_forward_context,
):
    expected = torch.tensor([[[0, 1, 2, 3]]], dtype=torch.int32)
    mock_lightning_indexer.return_value = expected

    state_cache = torch.zeros((1, 4, 4), dtype=torch.bfloat16)
    indexer_cache = torch.zeros((1, 1, 1, 2), dtype=torch.bfloat16)
    state_layer = SimpleNamespace(
        prefix="layer.indexer.state",
        compress_ratio=4,
        kv_cache=state_cache,
    )
    indexer_layer = SimpleNamespace(
        prefix="layer.indexer.k_cache",
        kv_cache=indexer_cache,
    )
    state_metadata = SimpleNamespace(
        slot_mapping=torch.tensor([3, -1, -1, -1, -1, -1, -1]),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        block_size=4,
    )
    indexer_metadata = SimpleNamespace(
        slot_mapping=torch.tensor([0, -1, -1, -1, -1, -1, -1]),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        seq_lens=torch.tensor([1], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([1], dtype=torch.int32),
    )
    attn_metadata = SimpleNamespace(
        num_actual_tokens=1,
        cum_query_lens=torch.tensor([1], dtype=torch.int32),
        seq_lens=torch.tensor([4], dtype=torch.int32),
    )
    mock_get_forward_context.return_value = SimpleNamespace(
        virtual_engine=0,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        attn_metadata={
            state_layer.prefix: state_metadata,
            indexer_layer.prefix: indexer_metadata,
            "layer.attn": attn_metadata,
        },
    )
    op = AscendSparseAttnIndexerKpool(
        k_cache=indexer_layer,
        quant_block_size=128,
        scale_fmt=None,
        topk_tokens=4,
        head_dim=2,
        max_model_len=16,
        max_total_seq_len=16,
        topk_indices_buffer=None,
        state_cache=state_layer,
        attn_layer_name="layer.attn",
    )

    result = op.forward_ascend(
        torch.empty((7, 1)),
        torch.zeros((7, 1, 2), dtype=torch.bfloat16),
        torch.arange(14, dtype=torch.bfloat16).view(7, 2),
        torch.ones((7, 1), dtype=torch.bfloat16),
        gate_score=torch.zeros((7, 2), dtype=torch.bfloat16),
        compress_ape=torch.zeros((4, 2), dtype=torch.float32),
        index_kpool=4,
        positions=torch.arange(7, dtype=torch.int64),
    )

    torch.testing.assert_close(result, expected)
    assert mock_compress.call_args.args[1].shape[0] == 1
    assert mock_compress.call_args.args[4].shape[0] == 1
    assert mock_lightning_indexer.call_args.args[0].shape[0] == 1
    assert mock_lightning_indexer.call_args.args[6].shape[0] == 1


@patch("vllm_ascend.models.glm5_next.get_forward_context")
@patch("torch.ops.vllm.glm5_next_lightning_indexer", create=True)
@patch("torch_npu.npu_scatter_nd_update_", create=True)
def test_indexer_kpool_mla_full_decode_avoids_dynamic_topk_and_cpu_length(
    mock_scatter,
    mock_lightning_indexer,
    mock_get_forward_context,
):
    expected = torch.tensor(
        [
            [[0, 1, 2, 3, -1, -1, -1]],
            [[-1, -1, -1, -1, 0, -1, -1]],
        ],
        dtype=torch.int32,
    )
    mock_lightning_indexer.return_value = expected

    state_cache = torch.zeros((1, 4, 4), dtype=torch.bfloat16)
    indexer_cache = torch.zeros((1, 1, 1, 2), dtype=torch.bfloat16)
    state_layer = SimpleNamespace(
        prefix="layer.indexer.state",
        compress_ratio=4,
        kv_cache=state_cache,
    )
    indexer_layer = SimpleNamespace(
        prefix="layer.indexer.k_cache",
        kv_cache=indexer_cache,
    )
    state_metadata = SimpleNamespace(
        slot_mapping=torch.tensor([3, -1], dtype=torch.int64),
        block_table=torch.tensor([[0], [0]], dtype=torch.int32),
        block_size=4,
    )
    indexer_metadata = SimpleNamespace(
        slot_mapping=torch.tensor([0, -1], dtype=torch.int64),
        block_table=torch.tensor([[0], [0]], dtype=torch.int32),
        seq_lens=torch.tensor([1, 0], dtype=torch.int32),
        seq_lens_cpu=SimpleNamespace(
            max=lambda: pytest.fail("full decode must not read CPU max sequence length")
        ),
    )
    attn_metadata = SimpleNamespace(
        cum_query_lens=torch.tensor([1, 2], dtype=torch.int32),
        seq_lens=torch.tensor([4, 1], dtype=torch.int32),
    )
    mock_get_forward_context.return_value = SimpleNamespace(
        virtual_engine=0,
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        attn_metadata={
            state_layer.prefix: state_metadata,
            indexer_layer.prefix: indexer_metadata,
            "layer.attn": attn_metadata,
        },
    )
    op = AscendSparseAttnIndexerKpool(
        k_cache=indexer_layer,
        quant_block_size=128,
        scale_fmt=None,
        topk_tokens=4,
        head_dim=2,
        max_model_len=16,
        max_total_seq_len=16,
        topk_indices_buffer=None,
        state_cache=state_layer,
        attn_layer_name="layer.attn",
    )
    with patch.object(
        op,
        "indexer_kpool_topk_pytorch",
        side_effect=AssertionError("full decode must not use the dynamic PyTorch top-k"),
    ):
        result = op.forward_ascend(
            torch.empty((2, 1)),
            torch.zeros((2, 1, 2), dtype=torch.bfloat16),
            torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.bfloat16),
            torch.ones((2, 1), dtype=torch.bfloat16),
            gate_score=torch.zeros((2, 2), dtype=torch.bfloat16),
            compress_ape=torch.zeros((4, 2), dtype=torch.float32),
            index_kpool=4,
            positions=torch.tensor([3, 0], dtype=torch.int64),
        )

    torch.testing.assert_close(result, expected)
    assert indexer_cache[0, 0, 0, 0] > 0
    assert mock_lightning_indexer.call_count == 1
    assert mock_lightning_indexer.call_args.kwargs["index_topk"] == 4
    assert mock_lightning_indexer.call_args.kwargs["index_kpool"] == 4
    assert mock_lightning_indexer.call_args.kwargs["max_pool_seq_len"] == 1
    mock_scatter.assert_not_called()


def test_indexer_kpool_mla_indexer_small_ops_use_bfloat16_cache_contract():
    assert list(inspect.signature(AscendSparseAttnIndexerKpool.cp_gather_indexer_k_cache).parameters) == [
        "kv_cache",
        "dst_k",
        "block_table",
        "cu_seq_lens",
    ]
    assert list(inspect.signature(AscendSparseAttnIndexerKpool.bf16_mqa_logits).parameters) == [
        "query",
        "key",
        "weights",
        "cu_seqlen_ks",
        "cu_seqlen_ke",
        "clean_logits",
    ]
    assert list(inspect.signature(AscendSparseAttnIndexerKpool.top_k_per_row_prefill).parameters) == [
        "logits",
        "cu_seqlen_ks",
        "cu_seqlen_ke",
        "raw_topk_indices",
        "num_rows",
        "stride0",
        "stride1",
        "topk_tokens",
    ]


def test_glm5_indexer_class_keeps_upstream_forward_contracts():
    expected_op_forward = [
        "self",
        "hidden_states",
        "q_quant",
        "k",
        "weights",
        "gate_score",
        "compress_ape",
        "index_kpool",
        "positions",
    ]
    assert list(inspect.signature(SparseAttnIndexerKpool.forward_native).parameters) == expected_op_forward
    assert list(inspect.signature(SparseAttnIndexerKpool.forward_ascend).parameters) == expected_op_forward
    assert list(inspect.signature(AscendGlm5NextIndexer.forward).parameters) == [
        "self",
        "hidden_states",
        "qr",
        "positions",
        "rotary_emb",
    ]


def test_indexer_kpool_mla_bf16_mqa_logits_matches_weighted_query_key_product():
    query = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [2.0, 0.0]],
        ],
        dtype=torch.bfloat16,
    )
    weights = torch.tensor(
        [
            [1.0, 2.0],
            [1.0, -1.0],
        ],
        dtype=torch.bfloat16,
    )
    key = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 2.0],
        ],
        dtype=torch.bfloat16,
    )
    cu_seqlen_ks = torch.tensor([0, 1], dtype=torch.int32)
    cu_seqlen_ke = torch.tensor([3, 4], dtype=torch.int32)

    result = AscendSparseAttnIndexerKpool.bf16_mqa_logits(
        query,
        key,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        clean_logits=False,
    )

    weighted_query = (query * weights.unsqueeze(-1)).sum(dim=1)
    expected = (weighted_query @ key.T).float()
    torch.testing.assert_close(result, expected)


def test_indexer_kpool_mla_top_k_per_row_prefill_returns_sequence_relative_indices():
    logits = torch.tensor(
        [
            [1.0, 5.0, 3.0, 100.0, 100.0],
            [100.0, 100.0, 2.0, 7.0, 4.0],
        ],
        dtype=torch.float32,
    )
    cu_seqlen_ks = torch.tensor([0, 2], dtype=torch.int32)
    cu_seqlen_ke = torch.tensor([3, 5], dtype=torch.int32)
    output = torch.empty((2, 2), dtype=torch.int32)

    AscendSparseAttnIndexerKpool.top_k_per_row_prefill(
        logits,
        cu_seqlen_ks,
        cu_seqlen_ke,
        output,
        logits.shape[0],
        logits.stride(0),
        logits.stride(1),
        2,
    )

    assert set(output[0].tolist()) == {1, 2}
    assert set(output[1].tolist()) == {1, 2}


def test_indexer_kpool_mla_indexer_kpool_topk_matches_weighted_mqa_on_paged_cache():
    logical_keys = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 2.0],
            [3.0, -1.0],
        ],
        dtype=torch.bfloat16,
    )
    # Compressed logical pages [0, 1, 2] map to physical pages [2, 0, 1].
    key_cache = torch.zeros((3, 2, 1, 2), dtype=torch.bfloat16)
    block_table = torch.tensor([[2, 0, 1]], dtype=torch.int32)
    for pool_id, logical_key in enumerate(logical_keys):
        logical_page, offset = divmod(pool_id, 2)
        physical_page = int(block_table[0, logical_page])
        key_cache[physical_page, offset, 0] = logical_key

    query = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [2.0, 0.0]],
        ],
        dtype=torch.bfloat16,
    )
    weights = torch.tensor(
        [
            [1.0, 2.0],
            [1.0, -1.0],
        ],
        dtype=torch.bfloat16,
    )

    result = AscendSparseAttnIndexerKpool.indexer_kpool_topk_pytorch(
        query=query,
        key=key_cache,
        weights=weights,
        actual_seq_lengths_query=torch.tensor([2], dtype=torch.int32),
        actual_seq_lengths_key=torch.tensor([5], dtype=torch.int32),
        block_table=block_table,
        query_positions=torch.tensor([5, 9], dtype=torch.int64),
        sparse_count=2,
        pool_size=2,
        max_key_seq_len=5,
        query_chunk_size=1,
        key_chunk_size=2,
    )

    # Query 0 sees only pools [0, 3), while query 1 sees all five pools.
    assert set(result[0].tolist()) == {1, 2}
    assert set(result[1].tolist()) == {1, 3}


def test_indexer_kpool_mla_indexer_kpool_topk_pads_when_history_is_short():
    result = AscendSparseAttnIndexerKpool.indexer_kpool_topk_pytorch(
        query=torch.tensor([[[1.0, 0.0]]], dtype=torch.bfloat16),
        key=torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]], dtype=torch.bfloat16),
        weights=torch.ones((1, 1), dtype=torch.bfloat16),
        actual_seq_lengths_query=torch.tensor([1], dtype=torch.int32),
        actual_seq_lengths_key=torch.tensor([2], dtype=torch.int32),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        query_positions=torch.tensor([1], dtype=torch.int64),
        sparse_count=4,
        pool_size=2,
        max_key_seq_len=2,
        query_chunk_size=1,
        key_chunk_size=2,
    )

    assert result.shape == (1, 4)
    assert result[result >= 0].tolist() == [0]


def test_glm5_next_lightning_indexer_matches_reference_chain():
    index_topk = 4
    index_kpool = 2
    logical_keys = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 2.0],
            [3.0, -1.0],
        ],
        dtype=torch.bfloat16,
    )
    indexer_cache = torch.zeros((3, 2, 1, 2), dtype=torch.bfloat16)
    indexer_block_table = torch.tensor([[2, 0, 1]], dtype=torch.int32)
    for pool_id, logical_key in enumerate(logical_keys):
        logical_page, offset = divmod(pool_id, 2)
        physical_page = int(indexer_block_table[0, logical_page])
        indexer_cache[physical_page, offset, 0] = logical_key
    indexer_cache_before = indexer_cache.clone()

    query = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [2.0, 0.0]],
        ],
        dtype=torch.bfloat16,
    )
    weights = torch.tensor(
        [
            [1.0, 2.0],
            [1.0, -1.0],
        ],
        dtype=torch.bfloat16,
    )
    cum_query_lens = torch.tensor([2], dtype=torch.int32)
    indexer_seq_lens = torch.tensor([5], dtype=torch.int32)
    positions = torch.tensor([4, 8], dtype=torch.int64)

    pool_ids = AscendSparseAttnIndexerKpool.indexer_kpool_topk_pytorch(
        query=query,
        key=indexer_cache,
        weights=weights,
        actual_seq_lengths_query=cum_query_lens,
        actual_seq_lengths_key=indexer_seq_lens,
        block_table=indexer_block_table,
        query_positions=positions,
        sparse_count=index_topk // index_kpool,
        pool_size=index_kpool,
        max_key_seq_len=5,
    )
    expected = AscendSparseAttnIndexerKpool.expand_pools_to_tokens(
        pool_ids,
        pool_ids >= 0,
        index_topk,
        index_kpool,
    )
    query_seq_lens = positions.to(torch.int32) + 1
    pool_lens = torch.div(
        query_seq_lens,
        index_kpool,
        rounding_mode="floor",
    )
    expected = AscendSparseAttnIndexerKpool.append_tail_to_topk(
        expected,
        query_seq_lens,
        pool_lens,
        index_kpool,
    ).unsqueeze(1)

    result = glm5_next_lightning_indexer(
        query,
        indexer_cache,
        weights,
        cum_query_lens,
        indexer_seq_lens,
        indexer_block_table,
        positions,
        index_topk=index_topk,
        index_kpool=index_kpool,
        max_pool_seq_len=5,
    )

    assert result.dtype == torch.int32
    assert result.shape == (2, 1, index_topk + index_kpool - 1)
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(indexer_cache, indexer_cache_before)


def test_glm5_next_lightning_indexer_fallback_aligns_cache_block_chunks():
    index_topk = 2
    index_kpool = 2
    max_pool_seq_len = 2050
    cache_block_size = 96
    head_dim = 2
    num_pages = (max_pool_seq_len + cache_block_size - 1) // cache_block_size

    query = torch.tensor([[[1.0, 0.0]]], dtype=torch.bfloat16)
    weights = torch.ones((1, 1), dtype=torch.bfloat16)
    indexer_cache = torch.zeros(
        (num_pages, cache_block_size, 1, head_dim),
        dtype=torch.bfloat16,
    )
    indexer_cache[0, 0, 0, 0] = 1.0
    indexer_block_table = torch.arange(num_pages, dtype=torch.int32).unsqueeze(0)
    cum_query_lens = torch.tensor([1], dtype=torch.int32)
    indexer_seq_lens = torch.tensor([max_pool_seq_len], dtype=torch.int32)
    positions = torch.tensor([max_pool_seq_len * index_kpool - 1], dtype=torch.int64)

    result = glm5_next_lightning_indexer(
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

    assert result.shape == (1, 1, index_topk + index_kpool - 1)
    assert result[0, 0].tolist() == [0, 1, -1]


def test_indexer_kpool_mla_kpool_compress_returns_bfloat16_without_quant_scale():
    indexer_cache = torch.zeros((1, 2, 1, 2), dtype=torch.bfloat16)
    slot_k = torch.tensor(
        [[[1.0, 3.0], [3.0, 1.0]]],
        dtype=torch.bfloat16,
    )
    slot_score = torch.zeros_like(slot_k)
    compress_ape = torch.zeros((2, 2), dtype=torch.float32)

    compressed_k = AscendSparseAttnIndexerKpool.kpool_compress_and_write_cache(
        indexer_cache,
        slot_k,
        slot_score,
        compress_ape,
        torch.tensor([0], dtype=torch.int64),
        pool_size=2,
        head_dim=2,
        return_compressed=True,
        write_cache=False,
    )

    assert isinstance(compressed_k, torch.Tensor)
    assert compressed_k.dtype == torch.bfloat16
    torch.testing.assert_close(
        compressed_k,
        torch.tensor([[2.0, 2.0]], dtype=torch.bfloat16),
    )


def test_glm5_next_kpool_compress_op_writes_paged_cache_like_reference():
    kv_cache = torch.full((2, 3, 1, 4), -7.0, dtype=torch.bfloat16)
    slot_k_storage = torch.arange(3 * 2 * 8, dtype=torch.float32).reshape(3, 2, 8)
    slot_score_storage = torch.arange(3 * 2 * 8, dtype=torch.float32).reshape(3, 2, 8) * 0.03
    ape_storage = torch.arange(2 * 8, dtype=torch.float32).reshape(2, 8) * 0.02
    slot_k = (slot_k_storage[:, :, ::2] * 0.125).to(torch.bfloat16)
    slot_score = slot_score_storage[:, :, ::2].to(torch.bfloat16)
    compress_ape = ape_storage[:, ::2]
    loc = torch.tensor([0, 3, 5], dtype=torch.int64)

    expected_compressed = AscendSparseAttnIndexerKpool.kpool_compress_and_write_cache(
        kv_cache.clone(),
        slot_k,
        slot_score,
        compress_ape,
        loc,
        pool_size=2,
        head_dim=4,
        return_compressed=True,
        write_cache=False,
    )
    assert expected_compressed is not None
    expected_cache = kv_cache.clone()
    block_ids = torch.div(loc, kv_cache.shape[1], rounding_mode="floor")
    block_offsets = torch.remainder(loc, kv_cache.shape[1])
    expected_cache[block_ids, block_offsets, 0, :] = expected_compressed

    result = glm5_next_kpool_compress_and_write_cache(
        kv_cache,
        slot_k,
        slot_score,
        compress_ape,
        loc,
    )

    assert result is None
    torch.testing.assert_close(kv_cache, expected_cache, rtol=1e-2, atol=1e-2)


def test_indexer_kpool_mla_sparse_attention_pytorch_matches_golden_semantics():
    # Logical token order is [0, 1, 2, 3], while the physical pages are
    # deliberately shuffled by block_table=[1, 0].
    logical_kv = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, -1.0],
        ],
        dtype=torch.float32,
    )
    raw_cache = torch.empty(16, dtype=torch.float32)
    packed_cache = torch.as_strided(
        raw_cache,
        size=(2, 2, 1, 3),
        stride=(8, 3, 3, 1),
    )
    packed_cache[1, :, 0, :] = logical_kv[:2]
    packed_cache[0, :, 0, :] = logical_kv[2:]
    latent_view, rope_view = packed_cache.split([2, 1], dim=-1)
    assert not latent_view.is_contiguous()
    assert not rope_view.is_contiguous()

    ql_nope = torch.tensor(
        [
            [[1.0, 0.0]],
            [[0.0, 1.0]],
            [[9.0, 9.0]],  # graph-padding row
        ],
        dtype=torch.float32,
    )
    q_pe = torch.tensor(
        [
            [[1.0]],
            [[-1.0]],
            [[9.0]],
        ],
        dtype=torch.float32,
    )
    topk_indices = torch.tensor(
        [
            [[3, 2, 0, -1]],
            [[3, 2, 0, -1]],
            [[0, 1, 2, 3]],
        ],
        dtype=torch.int32,
    )

    result = AscendIndexerKPoolMLAImpl._sparse_attention_pytorch(
        ql_nope=ql_nope,
        q_pe=q_pe,
        packed_kv_cache=packed_cache,
        topk_indices=topk_indices,
        block_table=torch.tensor([[1, 0]], dtype=torch.int32),
        actual_seq_lengths_query=torch.tensor([2], dtype=torch.int32),
        actual_seq_lengths_key=torch.tensor([4], dtype=torch.int32),
        scale=0.5,
        num_actual_tokens=2,
        query_chunk_size=1,
    )

    expected = torch.zeros_like(ql_nope)
    query_full = torch.cat([ql_nope[:2], q_pe[:2]], dim=-1)
    # Golden sparse-mode 3 threshold:
    # row 0: 4 - 2 + 0 + 1 = 3 -> token 3 is causally masked.
    # row 1: 4 - 2 + 1 + 1 = 4 -> token 3 is visible.
    selected_per_query = ([2, 0], [3, 2, 0])
    for query_idx, selected in enumerate(selected_per_query):
        selected_tensor = logical_kv[selected]
        scores = (query_full[query_idx, 0] @ selected_tensor.transpose(0, 1)) * 0.5
        probabilities = torch.softmax(scores, dim=-1)
        expected[query_idx, 0] = probabilities @ selected_tensor[:, :2]

    torch.testing.assert_close(result, expected)


def test_indexer_kpool_mla_delegates_sparse_attention_to_device_operator():
    impl = object.__new__(AscendIndexerKPoolMLAImpl)
    torch.nn.Module.__init__(impl)
    impl.kv_lora_rank = 2
    impl.qk_rope_head_dim = 0
    impl.scale = 0.5
    packed_cache = torch.zeros((1, 1, 1, 2), dtype=torch.bfloat16)
    impl._indexer_kpool_mla_caches = {"kv": packed_cache}
    expected = torch.ones((1, 1, 2), dtype=torch.bfloat16)
    block_table = torch.zeros((1, 1), dtype=torch.int32)
    metadata = SimpleNamespace(
        block_table=block_table,
        num_actual_tokens=1,
        sas_metadata=None,
    )
    ql_nope = torch.zeros((1, 1, 2), dtype=torch.bfloat16)
    q_pe = torch.empty((1, 1, 0), dtype=torch.bfloat16)
    kv_cache = (
        packed_cache[..., :2],
        packed_cache[..., 2:2],
    )
    topk_indices = torch.zeros((1, 1, 4), dtype=torch.int32)
    query_lens = torch.tensor([1], dtype=torch.int32)
    key_lens = torch.tensor([1], dtype=torch.int32)

    with patch.object(
        DeviceOperator,
        "execute_sparse_attention_indexer_kpool_mla",
        return_value=expected,
    ) as execute_sparse_attention:
        result = impl._execute_sparse_flash_attention_process(
            ql_nope,
            q_pe,
            kv_cache,
            topk_indices,
            metadata,
            query_lens,
            key_lens,
        )

    assert result is expected
    execute_sparse_attention.assert_called_once_with(
        impl,
        ql_nope,
        q_pe,
        packed_cache,
        topk_indices,
        metadata,
        query_lens,
        key_lens,
        block_table=block_table,
        sparse_mode=3,
        return_lse=False,
    )


def test_indexer_kpool_mla_sparse_attention_pytorch_maps_each_request_block_table():
    packed_cache = torch.tensor(
        [
            [[[3.0, 0.0, 0.0]]],
            [[[1.0, 0.0, 0.0]]],
        ],
        dtype=torch.float32,
    )
    ql_nope = torch.tensor(
        [
            [[1.0, 0.0]],
            [[1.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    q_pe = torch.zeros((2, 1, 1), dtype=torch.float32)

    result = AscendIndexerKPoolMLAImpl._sparse_attention_pytorch(
        ql_nope=ql_nope,
        q_pe=q_pe,
        packed_kv_cache=packed_cache,
        topk_indices=torch.zeros((2, 1, 1), dtype=torch.int32),
        block_table=torch.tensor(
            [
                [1],
                [0],
            ],
            dtype=torch.int32,
        ),
        actual_seq_lengths_query=torch.tensor([1, 2], dtype=torch.int32),
        actual_seq_lengths_key=torch.tensor([1, 1], dtype=torch.int32),
        scale=1.0,
        num_actual_tokens=2,
    )

    assert result[:, 0].tolist() == [
        [1.0, 0.0],
        [3.0, 0.0],
    ]
