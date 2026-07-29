# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import vllm.v1.core.kv_cache_utils as upstream_kv_cache_utils
from transformers import AutoConfig
from vllm import ModelRegistry
from vllm.config.compilation import CUDAGraphMode
from vllm.transformers_utils.model_arch_config_convertor import (
    MODEL_ARCH_CONFIG_CONVERTORS,
)
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.utils import select_common_block_size

from vllm_ascend.attention.indexer_kpool_mla_v1 import (
    AscendIndexerKPoolBackend,
    AscendIndexerKPoolMetadataBuilder,
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
    _align_glm5_cache_specs,
    _create_uniform_mamba_groups,
    _get_glm5_cache_layout,
    _get_kv_cache_config_deepseek_v4,
    _max_memory_usage_bytes_from_groups,
)
from vllm_ascend.patch.platform.patch_mamba_config import (
    _get_mamba_target_page_size,
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
        block_size=128,
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
            )
        ),
        torch.device("cpu"),
    )
    common_metadata = SimpleNamespace(
        num_reqs=1,
        num_input_tokens=4,
        positions=torch.tensor([0, 1, 2, 3]),
        slot_mapping=torch.tensor([7 * 128 + offset for offset in range(4)]),
        seq_lens=torch.tensor([4], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([4], dtype=torch.int32),
        seq_lens_cpu=None,
        block_table_tensor=torch.tensor([[7]], dtype=torch.int32),
    )

    metadata = builder.build(0, common_metadata)

    assert not issubclass(AscendIndexerKPoolMetadataBuilder, AscendSFAMetadataBuilder)
    assert AscendGlm5NextIndexerKPoolCache.get_attn_backend(None) is AscendIndexerKPoolBackend
    assert select_common_block_size(128, [AscendIndexerKPoolBackend]) == 128
    assert metadata.block_size == 32
    assert metadata.slot_mapping.tolist() == [-1, -1, -1, 7 * 32]
    assert metadata.seq_lens.tolist() == [1]
    assert metadata.seq_lens_cpu.tolist() == [1]
    assert metadata.block_table.tolist() == [[7]]

    first_slot_mapping_ptr = metadata.slot_mapping.data_ptr()
    first_seq_lens_ptr = metadata.seq_lens.data_ptr()
    common_metadata.positions = torch.tensor([4, 5, 6, 7])
    common_metadata.slot_mapping = torch.tensor([9 * 128 + offset for offset in range(4, 8)])
    common_metadata.seq_lens = torch.tensor([8], dtype=torch.int32)
    common_metadata._seq_lens_cpu = torch.tensor([8], dtype=torch.int32)
    common_metadata.block_table_tensor = torch.tensor([[9]], dtype=torch.int32)

    replay_metadata = builder.build(0, common_metadata)

    assert replay_metadata.slot_mapping.data_ptr() == first_slot_mapping_ptr
    assert replay_metadata.seq_lens.data_ptr() == first_seq_lens_ptr
    assert replay_metadata.slot_mapping.tolist() == [-1, -1, -1, 9 * 32 + 1]
    assert replay_metadata.seq_lens.tolist() == [2]


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


def test_glm5_kda_page_covers_ssm_and_conv_states():
    target_page_size = _get_mamba_target_page_size(
        is_glm5_next=True,
        attn_page_size=262144,
        mamba_raw_size=271360,
        conv_block_page_size=9216,
    )

    assert target_page_size == 271360


def test_glm5_mamba_groups_use_uniform_type_wrapper():
    mamba_specs = {
        f"layer.{layer_idx}": MambaSpec(
            block_size=256,
            shapes=((8, 128, 128), (6, 3, 128)),
            dtypes=(torch.bfloat16, torch.float32),
            page_size_padded=271360,
        )
        for layer_idx in (0, 3)
    }

    groups = _create_uniform_mamba_groups(
        mamba_specs,
        [["layer.0", "layer.3"]],
    )

    assert len(groups) == 1
    assert isinstance(groups[0].kv_cache_spec, UniformTypeKVCacheSpecs)
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
    state_cache = torch.zeros((4, 4, 2), dtype=torch.bfloat16)
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


def _uniform_group(specs, *names):
    group_specs = {name: specs[name] for name in names}
    uniform_spec = UniformTypeKVCacheSpecs.from_specs(group_specs)
    assert uniform_spec is not None
    return KVCacheGroupSpec(list(names), uniform_spec)


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
            block_size=128,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
            model_version="glm5_next",
        )
        specs[f"{prefix}.indexer.k_cache"] = MLAAttentionSpec(
            block_size=128,
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
            block_size=128,
            shapes=(
                (3 + num_speculative_tokens, 1536),
                (4, 128, 128),
            ),
            dtypes=(torch.bfloat16, torch.float32),
            num_speculative_blocks=num_speculative_tokens,
        )

    _align_glm5_cache_specs(specs)
    mla_names = [name for name in specs if name.endswith(".attn")]
    indexer_names = [name for name in specs if name.endswith(".indexer.k_cache")]
    state_names = [name for name in specs if name.endswith(".state_cache")]
    groups = [
        _uniform_group(specs, *mla_names),
        _uniform_group(specs, *indexer_names),
        _uniform_group(specs, *state_names),
    ]
    if mamba_layer_indices:
        mamba_specs = {name: spec for name, spec in specs.items() if isinstance(spec, MambaSpec)}
        mamba_names = [
            [f"model.layers.{layer_idx}.mamba" for layer_idx in mamba_layer_indices if layer_idx % 4 == group_idx]
            for group_idx in range(3)
        ]
        groups.extend(_create_uniform_mamba_groups(mamba_specs, mamba_names))
    return specs, groups


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
    assert layout.main_page_size == 271360
    assert layout.small_page_size == 16 * 256 * 2

    bytes_per_block = 12 * layout.main_page_size + 11 * layout.small_page_size
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(num_gpu_blocks_override=None),
    )
    num_blocks, tensors = _get_kv_cache_config_deepseek_v4(
        vllm_config,
        groups,
        available_memory=bytes_per_block * 3,
    )

    assert num_blocks == 3
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


def test_glm5_target_mtp5_keeps_slot_counts_and_expands_kda_page():
    specs, groups = _make_glm5_cache_groups(
        include_mtp=False,
        num_speculative_tokens=5,
    )
    layout = _get_glm5_cache_layout(groups)
    assert layout is not None

    assert layout.main_slot_count == 12
    assert layout.small_slot_count == 11
    assert layout.main_page_size == 286720
    assert {
        spec.num_speculative_blocks
        for spec in specs.values()
        if isinstance(spec, MambaSpec)
    } == {5}


def test_indexer_kpool_mla_standalone_mtp_allocator_uses_two_page_classes():
    specs = {
        "layer.attn": MLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
            model_version="glm5_next",
        ),
        "layer.indexer.k_cache": MLAAttentionSpec(
            block_size=128,
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
    _align_glm5_cache_specs(specs)
    groups = [
        KVCacheGroupSpec(
            list(group_specs),
            UniformTypeKVCacheSpecs(
                block_size=next(iter(group_specs.values())).block_size,
                kv_cache_specs=group_specs,
            ),
        )
        for group_specs in (
            {"layer.attn": specs["layer.attn"]},
            {"layer.indexer.k_cache": specs["layer.indexer.k_cache"]},
            {"layer.indexer.compressor.state_cache": specs["layer.indexer.compressor.state_cache"]},
        )
    ]
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


def test_glm5_memory_accounting_charges_shared_pool_for_every_group():
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
    blocks_needed = sum(group.kv_cache_spec.max_memory_usage_pages(vllm_config) for group in groups)
    bytes_per_block = 12 * layout.main_page_size + 11 * layout.small_page_size

    assert _max_memory_usage_bytes_from_groups(vllm_config, groups) == (blocks_needed * bytes_per_block)


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
    cache = torch.zeros((1, 4, 8), dtype=torch.bfloat16)
    slots = torch.tensor([2, -1], dtype=torch.int64)
    values = torch.ones((2, 1, 8), dtype=torch.bfloat16)

    AscendIndexerKPoolMLAImpl._store_indexer_cache(
        None,
        cache,
        slots,
        values,
    )

    mock_scatter.assert_not_called()
    assert cache.view(-1, 8)[2].tolist() == [1.0] * 8
    assert cache.view(-1, 8)[0].tolist() == [0.0] * 8


@patch("vllm_ascend.models.glm5_next.get_forward_context")
@patch("torch.ops._C_ascend.npu_lightning_indexer", create=True)
@patch("torch_npu.npu_scatter_nd_update_", create=True)
def test_indexer_kpool_mla_full_decode_avoids_dynamic_topk_and_cpu_length(
    mock_scatter,
    mock_lightning_indexer,
    mock_get_forward_context,
):
    mock_lightning_indexer.return_value = (
        torch.tensor([[[0]], [[-1]]], dtype=torch.int32),
        torch.empty(0),
    )

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
        seq_lens_cpu=SimpleNamespace(max=lambda: pytest.fail("full decode must not read CPU max sequence length")),
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

    assert result.shape == (2, 1, 7)
    assert result[0, 0].tolist() == [0, 1, 2, 3, -1, -1, -1]
    assert result[1, 0].tolist() == [-1, -1, -1, -1, 0, -1, -1]
    assert indexer_cache[0, 0, 0, 0] > 0
    assert mock_lightning_indexer.call_count == 1
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
    packed_cache = torch.empty((2, 2, 1, 3), dtype=torch.float32)
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
