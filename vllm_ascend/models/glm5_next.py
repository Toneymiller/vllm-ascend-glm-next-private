from collections.abc import Iterable
from typing import ClassVar

import torch
import torch_npu
from einops import rearrange
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CacheConfig,
    ParallelConfig,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    GateLinear,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    KimiGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from vllm.model_executor.models.deepseek_v2 import yarn_get_mscale
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
    sequence_parallel_chunk,
)
from vllm.models.deepseek_v4.compressor import CompressorStateCache
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.mla.indexer import (
    get_max_prefill_buffer_size,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.core.kv_cache_interface import AscendIndexerKPoolStateSpec
from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionBackend
from vllm_ascend.ops.indexer_kpool_mla import (
    AscendIndexerKPoolMLAAttention,
    IndexerKPoolMLAModules,
)
from vllm_ascend.ops.kimi_kda_state import kimi_kda_state_shape
from vllm_ascend.ops.triton.kda.kda import (
    chunk_kda,
    fused_kda_gate,
    fused_recurrent_kda_fwd,
    rms_norm_gated,
)
from vllm_ascend.transformers_utils.configs.glm5_next import Glm5NextTextConfig
from vllm_ascend.utils import uses_global_inputs_embeds

INDEXER_KPOOL_HEAD_DIM = 128
INDEXER_KPOOL_QUERY_CHUNK_SIZE = 16
INDEXER_KPOOL_KEY_CHUNK_SIZE = 2048
MTP_ROT_WEIGHT_NAME = "rot.weight"

GLM5_TRANSFORMERS_INTERNAL_WEIGHTS_MAPPER = WeightsMapper(
    orig_to_new_substr={
        ".self_attn.forget_gate.A_log": ".self_attn.A_log",
        ".self_attn.forget_gate.dt_bias": ".self_attn.dt_bias",
        ".self_attn.forget_gate.f_a_proj": ".self_attn.f_a_proj",
        ".self_attn.forget_gate.f_b_proj": ".self_attn.f_b_proj",
        ".attn_hc.fn": ".hc_attn_fn",
        ".attn_hc.base": ".hc_attn_base",
        ".attn_hc.scale": ".hc_attn_scale",
        ".ffn_hc.fn": ".hc_ffn_fn",
        ".ffn_hc.base": ".hc_ffn_base",
        ".ffn_hc.scale": ".hc_ffn_scale",
    }
)

GLM5_PACKED_MODULES_MAPPING = {
    "gate_up_proj": ["gate_proj", "up_proj"],
    "experts": [
        "experts.0.gate_proj",
        "experts.0.up_proj",
        "experts.0.down_proj",
    ],
    "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
}


def get_spec_layer_idx_from_weight_name(config, weight_name: str) -> int | None:
    num_mtp_layers = getattr(config, "num_nextn_predict_layers", 0)
    if num_mtp_layers <= 0:
        return None
    first_mtp_layer = config.num_hidden_layers
    if weight_name == MTP_ROT_WEIGHT_NAME:
        return first_mtp_layer
    for layer_idx in range(first_mtp_layer, first_mtp_layer + num_mtp_layers):
        if f"layers.{layer_idx}." in weight_name:
            return layer_idx
    return None


def _expand_mhc_residual_streams(
    hidden_states: torch.Tensor,
    num_streams: int,
) -> torch.Tensor:
    streams = hidden_states.unsqueeze(1).expand(-1, num_streams, -1)
    return streams.contiguous().view(hidden_states.shape[0], -1)


class AscendGlm5NextGatedRMSNormParams(nn.Module):
    """保存 gated RMSNorm 参数，并匹配 Transformers checkpoint 命名。"""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # Transformers 只保存 o_norm.weight；Ascend kernel 需要显式零 bias。
        # non-persistent buffer 不参与 checkpoint 和严格参数完整性检查。
        self.register_buffer(
            "bias",
            torch.zeros(hidden_size),
            persistent=False,
        )


def _get_indexer_kpool_mla_backend() -> type[AttentionBackend]:
    # 延迟导入，避免模型检查阶段形成：GLM5 -> SFA -> device_op -> ops
    # -> fused_moe -> device_op 的循环导入。
    from vllm_ascend.attention.indexer_kpool_mla_v1 import (
        AscendIndexerKPoolMLABackend,
    )

    return AscendIndexerKPoolMLABackend


def _get_indexer_kpool_state_backend() -> type[AttentionBackend]:
    # 与 MLA backend 一样延迟导入，避免模型检查阶段形成循环导入。
    from vllm_ascend.attention.indexer_kpool_mla_v1 import (
        AscendIndexerKPoolStateBackend,
    )

    return AscendIndexerKPoolStateBackend


def _get_indexer_kpool_backend() -> type[AttentionBackend]:
    # Cache-only backend 同样延迟导入，避免模型检查阶段形成循环导入。
    from vllm_ascend.attention.indexer_kpool_mla_v1 import (
        AscendIndexerKPoolBackend,
    )

    return AscendIndexerKPoolBackend


def resolve_kda_config(config) -> dict:
    """Normalize legacy and latest flat KDA configuration fields."""
    legacy_config = getattr(config, "linear_attn_config", None)
    if legacy_config is not None:
        return legacy_config
    return {
        "head_dim": getattr(config, "linear_head_dim", 128),
        "num_heads": getattr(config, "linear_num_heads", 64),
        "short_conv_kernel_size": getattr(
            config,
            "linear_conv_kernel_dim",
            4,
        ),
        "lower_bound": getattr(config, "linear_lower_bound", -5.0),
    }


def _is_kda_layer(config, layer_idx: int) -> bool:
    if hasattr(config, "is_kda_layer"):
        return config.is_kda_layer(layer_idx)
    kda_layers = (getattr(config, "linear_attn_config", None) or {}).get("kda_layers", [])
    if kda_layers:
        return layer_idx in kda_layers
    layer_types = getattr(config, "layer_types", None)
    return (
        layer_types is not None
        and layer_idx < len(layer_types)
        and layer_types[layer_idx] == "linear_attention"
    )


def _is_moe(config) -> bool:
    if hasattr(config, "is_moe"):
        return config.is_moe
    return getattr(config, "n_routed_experts", None) is not None


def _pad_nope_kv_a_weight(
    config: Glm5NextTextConfig,
    name: str,
    loaded_weight: torch.Tensor,
) -> torch.Tensor:
    """Pad the checkpoint's latent-only KV projection for NoPE models."""
    rope_dim = getattr(config, "qk_rope_head_dim", 0)
    kv_lora_rank = getattr(config, "kv_lora_rank", None)
    if (
        not getattr(config, "mla_nope", False)
        or rope_dim <= 0
        or kv_lora_rank is None
        or not name.endswith(".kv_a_proj_with_mqa.weight")
        or loaded_weight.shape[0] != kv_lora_rank
    ):
        return loaded_weight
    padding = torch.zeros(
        rope_dim,
        *loaded_weight.shape[1:],
        dtype=loaded_weight.dtype,
        device=loaded_weight.device,
    )
    return torch.cat([loaded_weight, padding], dim=0)


class AscendGlm5NextIndexerKPoolCache(nn.Module, AttentionLayerBase):
    """One independently allocated physical cache in GLM-5 Indexer KPool MLA."""

    def __init__(
        self,
        *,
        head_dim: int,
        dtype: torch.dtype,
        cache_role: str,
        cache_config: CacheConfig,
        prefix: str,
        compress_ratio: int = 1,
    ) -> None:
        super().__init__()
        if cache_config.block_size % compress_ratio:
            raise ValueError(
                "Indexer KPool MLA cache block size "
                f"{cache_config.block_size} must be divisible by "
                f"compress ratio {compress_ratio}."
            )
        self.head_dim = head_dim
        self.dtype = dtype
        self.cache_role = cache_role
        self.cache_config = cache_config
        self.compress_ratio = compress_ratio
        self.prefix = prefix
        self.kv_cache = [
            torch.tensor([]) for _ in range(get_current_vllm_config().parallel_config.pipeline_parallel_size)
        ]
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return MLAAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            cache_dtype_str=None,
            compress_ratio=self.compress_ratio,
            model_version="glm5_next",
        )

    def get_attn_backend(self):
        return _get_indexer_kpool_backend()

    def forward(self): ...


class AscendGlm5NextCompressorStateCache(CompressorStateCache):
    """GLM-5 kpool tail 状态，每个 token 保存一个 ``[K, gate]`` 向量。

    这是每个请求占用一个 page 的滑动 tail，不是按 position 取模的张量
    环形队列。state block table 仍用于把每个请求的绝对逻辑 pool 位置
    映射到 allocator 管理的物理 page。
    """

    def __init__(
        self,
        *,
        state_dim: int,
        dtype: torch.dtype,
        compress_ratio: int,
        cache_config: CacheConfig,
        prefix: str,
    ) -> None:
        nn.Module.__init__(self)
        if dtype != torch.bfloat16:
            raise ValueError(f"GLM-5 compressor state must use bfloat16, got {dtype}.")
        self.state_dim = state_dim
        self.dtype = dtype
        self.prefix = prefix
        self.compress_ratio = compress_ratio
        self.sliding_window = compress_ratio
        self.block_size = compress_ratio
        self.cache_config = cache_config
        self.cache_role = "indexer_state"
        self.kv_cache = [
            torch.tensor([]) for _ in range(get_current_vllm_config().parallel_config.pipeline_parallel_size)
        ]
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return AscendIndexerKPoolStateSpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.state_dim,
            dtype=self.dtype,
            sliding_window=self.sliding_window,
            cache_dtype_str=None,
            model_version="glm5_next",
            cache_role=self.cache_role,
        )

    def get_attn_backend(self):
        return _get_indexer_kpool_state_backend()

    def forward(self): ...


class AscendSparseAttnIndexerKpool(nn.Module):
    """Ascend implementation of vLLM's ``SparseAttnIndexerKpool``.

    The public constructor and forward arguments follow the upstream custom-op
    layer. CUDA FP8/Triton kernels are replaced by the pure-BF16 Indexer
    implementations owned by this class.
    """

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str | None,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor | None,
        skip_k_cache_insert: bool = False,
        use_fp4_cache: bool = False,
        *,
        state_cache: AscendGlm5NextCompressorStateCache,
        attn_layer_name: str,
    ) -> None:
        super().__init__()
        self.k_cache = k_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.skip_k_cache_insert = skip_k_cache_insert
        self.use_fp4_cache = use_fp4_cache
        self.state_cache = state_cache
        self.attn_layer_name = attn_layer_name
        # top-k token 数和 pool 大小均为模型静态配置，只需在初始化时计算一次。
        self.pool_topk = self.history_group_budget_for_topk(
            self.topk_tokens,
            self.state_cache.compress_ratio,
        )

    @staticmethod
    def _bound_cache(layer) -> torch.Tensor | tuple[torch.Tensor, ...]:
        context = get_forward_context()
        virtual_engine = getattr(context, "virtual_engine", 0) or 0
        cache = layer.kv_cache
        if isinstance(cache, (list, tuple)):
            cache = cache[virtual_engine]
        if isinstance(cache, (list, tuple)):
            if len(cache) == 1:
                cache = cache[0]
            elif all(isinstance(tensor, torch.Tensor) for tensor in cache):
                return tuple(cache)
        if not isinstance(cache, torch.Tensor):
            raise TypeError(f"GLM-5 Indexer cache {type(layer).__name__} is not bound.")
        return cache

    @staticmethod
    def _scatter_rows_graph_safe(
        cache_rows: torch.Tensor,
        slots: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Scatter fixed-shape rows while treating invalid slots as no-ops."""
        valid = (slots >= 0) & (slots < cache_rows.shape[0])
        safe_slots = torch.where(valid, slots, torch.zeros_like(slots))
        row_mask = valid.view(-1, *([1] * (values.ndim - 1)))
        row_zero = cache_rows[0].clone()
        safe_values = torch.where(row_mask, values, row_zero.unsqueeze(0))
        row_zero_mask = valid & (slots == 0)
        update_zero = torch.where(
            row_zero_mask.view(-1, *([1] * (values.ndim - 1))),
            values,
            torch.zeros_like(values),
        ).sum(dim=0)
        expected_zero = torch.where(row_zero_mask.any(), update_zero, row_zero)
        # Avoid the scatter-nd update API in ACLGraph capture. Padded rows use
        # row zero as a fixed-shape sentinel and it is restored immediately.
        cache_rows[safe_slots] = safe_values
        cache_rows[0].copy_(expected_zero)

    @staticmethod
    def _scatter_paged_cache(
        cache: torch.Tensor,
        slots: torch.Tensor,
        values: torch.Tensor,
        block_size: int,
    ) -> None:
        if cache.shape[1] != block_size:
            raise ValueError(
                f"Cache block size mismatch: expected {block_size}, got {cache.shape[1]}."
            )
        values = values.reshape(values.shape[0], *cache.shape[2:])
        valid = (slots >= 0) & (slots < cache.shape[0] * block_size)
        safe_slots = torch.where(valid, slots, torch.zeros_like(slots))
        block_ids = torch.div(
            safe_slots,
            block_size,
            rounding_mode="floor",
        )
        block_offsets = torch.remainder(safe_slots, block_size)
        row_mask = valid.view(-1, *([1] * (values.ndim - 1)))
        row_zero = cache[0, 0].clone()
        safe_values = torch.where(row_mask, values, row_zero.unsqueeze(0))
        row_zero_mask = valid & (slots == 0)
        update_zero = torch.where(
            row_zero_mask.view(-1, *([1] * (values.ndim - 1))),
            values,
            torch.zeros_like(values),
        ).sum(dim=0)
        expected_zero = torch.where(
            row_zero_mask.any(),
            update_zero,
            row_zero,
        )
        cache[block_ids, block_offsets] = safe_values
        cache[0, 0].copy_(expected_zero)

    def _gather_compressor_state(
        self,
        state_cache: torch.Tensor,
        state_metadata,
        end_positions: torch.Tensor,
        request_ids: torch.Tensor,
        index_kpool: int,
    ) -> torch.Tensor:
        offsets = torch.arange(
            index_kpool - 1,
            -1,
            -1,
            device=end_positions.device,
        )
        logical = end_positions[:, None] - offsets[None, :]
        safe_logical = logical.clamp_min(0)
        pages = torch.div(
            safe_logical,
            state_metadata.block_size,
            rounding_mode="floor",
        ).clamp_max(state_metadata.block_table.shape[1] - 1)
        page_offsets = torch.remainder(safe_logical, state_metadata.block_size)
        physical_blocks = state_metadata.block_table[
            request_ids[:, None],
            pages,
        ].clamp(min=0, max=state_cache.shape[0] - 1)
        return state_cache[
            physical_blocks.long(),
            page_offsets,
        ]

    @staticmethod
    def indexer_kpool_topk_decode(
        query: torch.Tensor,
        key: torch.Tensor,
        weights: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        block_table: torch.Tensor,
        sparse_count: int,
    ) -> torch.Tensor:
        """Select compressed pools with the graph-compatible Ascend op."""
        pool_ids, _ = torch.ops._C_ascend.npu_lightning_indexer(
            query=query,
            key=key,
            weights=weights,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
            block_table=block_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=sparse_count,
            sparse_mode=3,
        )
        return pool_ids.squeeze(1)

    @staticmethod
    def history_group_budget_for_topk(topk: int, pool_size: int) -> int:
        """计算 pool 级 top-k 数量，并校验 token 预算能够整除。"""
        if topk % pool_size:
            raise ValueError(f"topk ({topk}) must be divisible by pool_size ({pool_size}).")
        return topk // pool_size

    @staticmethod
    def expand_pools_to_tokens(
        group_ids: torch.Tensor,
        group_valid: torch.Tensor,
        topk: int,
        pool_size: int,
        page_table: torch.Tensor | None = None,
        topk_offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """将选中的 pool id 展开为 token id。"""
        if group_ids.ndim != 2:
            raise ValueError(f"group_ids must be 2-D, got shape {group_ids.shape}.")
        if group_valid.shape != group_ids.shape:
            raise ValueError(
                f"group_valid must have the same shape as group_ids, got {group_valid.shape} and {group_ids.shape}."
            )
        group_budget = AscendSparseAttnIndexerKpool.history_group_budget_for_topk(
            topk,
            pool_size,
        )
        if group_ids.shape[1] != group_budget:
            raise ValueError(f"group_ids width must be {group_budget}, got {group_ids.shape[1]}.")
        if page_table is not None and topk_offsets is not None:
            raise ValueError("page_table and topk_offsets are mutually exclusive.")

        offsets = torch.arange(
            pool_size,
            device=group_ids.device,
            dtype=torch.int64,
        )
        token_ids = group_ids.to(torch.int64).unsqueeze(-1) * pool_size + offsets
        token_ids = token_ids.reshape(group_ids.shape[0], topk)
        valid = group_valid.unsqueeze(-1).expand(-1, -1, pool_size).reshape(group_ids.shape[0], topk)

        if page_table is not None:
            if page_table.ndim != 2:
                raise ValueError(f"page_table must be 2-D, got shape {page_table.shape}.")
            safe_ids = token_ids.clamp(min=0, max=page_table.shape[1] - 1)
            output = torch.gather(page_table, dim=1, index=safe_ids).to(torch.int32)
        elif topk_offsets is not None:
            if topk_offsets.ndim == 2:
                if topk_offsets.shape[1] != 1:
                    raise ValueError("2-D topk_offsets must have a singleton last dimension.")
                topk_offsets = topk_offsets.squeeze(1)
            output = (token_ids + topk_offsets.to(torch.int64).unsqueeze(1)).to(torch.int32)
        else:
            output = token_ids.to(torch.int32)

        return torch.where(valid, output, torch.full_like(output, -1))

    @staticmethod
    def append_tail_to_topk(
        topk_result: torch.Tensor,
        seq_lens: torch.Tensor,
        pool_lens: torch.Tensor,
        pool_size: int,
        page_table: torch.Tensor | None = None,
        topk_offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """把每个 query 尚未组成完整 pool 的因果 tail 追加到 top-k。"""
        if topk_result.dtype != torch.int32:
            raise TypeError(f"topk_result must use int32, got {topk_result.dtype}.")
        if topk_result.ndim != 2:
            raise ValueError(f"topk_result must be 2-D, got shape {topk_result.shape}.")
        if seq_lens.ndim != 1 or pool_lens.ndim != 1:
            raise ValueError("seq_lens and pool_lens must both be 1-D.")
        if seq_lens.shape != pool_lens.shape:
            raise ValueError(
                f"seq_lens and pool_lens must have the same shape, got {seq_lens.shape} and {pool_lens.shape}."
            )
        if topk_result.shape[0] != seq_lens.shape[0]:
            raise ValueError(
                f"topk_result rows must match seq_lens, got {topk_result.shape[0]} and {seq_lens.shape[0]}."
            )
        if page_table is not None and topk_offsets is not None:
            raise ValueError("page_table and topk_offsets are mutually exclusive.")

        tail_width = pool_size - 1
        if tail_width == 0:
            return topk_result

        rows, history_width = topk_result.shape
        output_width = history_width + tail_width
        columns = torch.arange(
            output_width,
            device=topk_result.device,
        )[None, :]
        is_history = columns < history_width
        tail_offsets = columns - history_width

        pool_lens = pool_lens.to(torch.int32)
        seq_lens = seq_lens.to(torch.int32)
        tail_start = pool_lens * pool_size
        tail_count = seq_lens - tail_start
        is_tail = (tail_offsets >= 0) & (tail_offsets < tail_count[:, None])

        safe_history = torch.minimum(
            columns,
            torch.full_like(columns, history_width - 1),
        ).expand(rows, output_width)
        history_values = torch.gather(topk_result, 1, safe_history)

        tail_raw = tail_start[:, None] + tail_offsets
        tail_values = tail_raw.to(torch.int32)
        if page_table is not None:
            if page_table.ndim != 2:
                raise ValueError(f"page_table must be 2-D, got shape {page_table.shape}.")
            safe_tail = tail_raw.clamp(min=0, max=page_table.shape[1] - 1)
            tail_values = torch.gather(page_table, 1, safe_tail).to(torch.int32)
        elif topk_offsets is not None:
            if topk_offsets.ndim == 2:
                if topk_offsets.shape[1] != 1:
                    raise ValueError("2-D topk_offsets must have a singleton last dimension.")
                topk_offsets = topk_offsets.squeeze(1)
            tail_values = (tail_raw + topk_offsets.to(torch.int64).unsqueeze(1)).to(torch.int32)

        output = torch.where(is_history, history_values, -1)
        return torch.where(is_tail, tail_values, output)

    @staticmethod
    def cp_gather_indexer_k_cache(
        kv_cache: torch.Tensor,
        dst_k: torch.Tensor,
        block_table: torch.Tensor,
        cu_seq_lens: torch.Tensor,
    ) -> None:
        """从分页 BF16 Indexer K cache 收集连续的逻辑 K。"""
        paged_k = kv_cache
        if paged_k.ndim != 4 or paged_k.shape[2] != 1:
            raise ValueError(f"Paged indexer K must be [blocks,block,1,dim], got {paged_k.shape}.")
        if paged_k.dtype != torch.bfloat16:
            raise TypeError(f"Paged indexer K must use bfloat16, got {paged_k.dtype}.")
        if dst_k.ndim != 2 or dst_k.shape[1] != paged_k.shape[-1]:
            raise ValueError(f"dst_k must be [total_k,{paged_k.shape[-1]}], got {dst_k.shape}.")
        if dst_k.dtype != torch.bfloat16:
            raise TypeError(f"dst_k must use bfloat16, got {dst_k.dtype}.")
        if block_table.ndim != 2:
            raise ValueError(f"block_table must be 2-D, got {block_table.shape}.")
        if cu_seq_lens.shape != (block_table.shape[0] + 1,):
            raise ValueError(
                "cu_seq_lens must contain one boundary per block-table row, "
                f"got {cu_seq_lens.shape} and {block_table.shape[0]} rows."
            )
        if block_table.shape[1] == 0 and dst_k.shape[0]:
            raise ValueError("block_table has no columns for a non-empty gather.")
        if dst_k.shape[0] == 0:
            return

        output_rows = torch.arange(
            dst_k.shape[0],
            dtype=cu_seq_lens.dtype,
            device=dst_k.device,
        )
        request_ids = torch.bucketize(
            output_rows,
            cu_seq_lens[1:],
            right=True,
        )
        logical_indices = output_rows - cu_seq_lens[request_ids]
        cache_block_size = paged_k.shape[1]
        logical_pages = torch.div(
            logical_indices,
            cache_block_size,
            rounding_mode="floor",
        )
        page_offsets = torch.remainder(logical_indices, cache_block_size)
        physical_blocks = block_table[
            request_ids,
            logical_pages,
        ].to(torch.int64)
        safe_physical_blocks = physical_blocks.clamp(
            min=0,
            max=paged_k.shape[0] - 1,
        )
        gathered_k = paged_k[
            safe_physical_blocks,
            page_offsets,
            0,
            :,
        ]
        dst_k.copy_(gathered_k)

    @staticmethod
    def bf16_mqa_logits(
        query: torch.Tensor,
        key: torch.Tensor,
        weights: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        clean_logits: bool,
    ) -> torch.Tensor:
        """使用 BF16 Q/K 计算经过多头权重归约的 Indexer logits。"""
        if query.ndim != 3:
            raise ValueError(f"Indexer query must be [M,H,D], got {query.shape}.")
        if query.dtype != torch.bfloat16:
            raise TypeError(f"Indexer query must use bfloat16, got {query.dtype}.")
        if weights.shape != query.shape[:2]:
            raise ValueError(f"Indexer weights must be [M,H], got {weights.shape} for query {query.shape}.")
        if key.ndim != 2 or key.shape[1] != query.shape[2]:
            raise ValueError(f"Indexer K must be [N,D] with D={query.shape[2]}, got {key.shape}.")
        if key.dtype != torch.bfloat16:
            raise TypeError(f"Indexer K must use bfloat16, got {key.dtype}.")
        if cu_seqlen_ks.shape != (query.shape[0],) or cu_seqlen_ke.shape != cu_seqlen_ks.shape:
            raise ValueError(
                "cu_seqlen_ks/cu_seqlen_ke must have one entry per query, "
                f"got {cu_seqlen_ks.shape}, {cu_seqlen_ke.shape}, and M={query.shape[0]}."
            )

        # sum_h(weight_h * dot(q_h, k)) ==
        # dot(sum_h(weight_h * q_h), k), without an [M,H,N] tensor.
        weighted_q = (query * weights.to(torch.bfloat16).unsqueeze(-1)).sum(dim=1)
        logits = torch.matmul(weighted_q, key.transpose(0, 1)).float()
        if clean_logits:
            columns = torch.arange(
                logits.shape[1],
                dtype=cu_seqlen_ks.dtype,
                device=logits.device,
            )
            valid = (columns[None, :] >= cu_seqlen_ks[:, None]) & (columns[None, :] < cu_seqlen_ke[:, None])
            logits = logits.masked_fill(
                ~valid,
                torch.finfo(logits.dtype).min,
            )
        return logits

    @staticmethod
    def top_k_per_row_prefill(
        logits: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        raw_topk_indices: torch.Tensor,
        num_rows: int,
        stride0: int,
        stride1: int,
        topk_tokens: int,
    ) -> None:
        """使用与上游算子相同的接口执行逐行 top-k。"""
        if logits.ndim != 2:
            raise ValueError(f"Indexer logits must be 2-D, got {logits.shape}.")
        if num_rows != logits.shape[0]:
            raise ValueError(f"num_rows={num_rows} does not match logits rows {logits.shape[0]}.")
        if stride0 != logits.stride(0) or stride1 != logits.stride(1):
            raise ValueError(
                f"Explicit logits strides do not match the tensor: {(stride0, stride1)} vs {logits.stride()}."
            )
        if cu_seqlen_ks.shape != (num_rows,) or cu_seqlen_ke.shape != (num_rows,):
            raise ValueError("cu_seqlen_ks/cu_seqlen_ke must have one entry per logits row.")
        if raw_topk_indices.shape[0] < num_rows or raw_topk_indices.shape[1] < topk_tokens:
            raise ValueError(f"raw_topk_indices {raw_topk_indices.shape} cannot hold [{num_rows}, {topk_tokens}].")
        if topk_tokens <= 0:
            raise ValueError(f"topk_tokens must be positive, got {topk_tokens}.")

        columns = torch.arange(
            logits.shape[1],
            dtype=cu_seqlen_ks.dtype,
            device=logits.device,
        )
        valid = (columns[None, :] >= cu_seqlen_ks[:, None]) & (columns[None, :] < cu_seqlen_ke[:, None])
        scores = logits.masked_fill(
            ~valid,
            torch.finfo(logits.dtype).min,
        )
        if scores.shape[1] < topk_tokens:
            scores = torch.nn.functional.pad(
                scores,
                (0, topk_tokens - scores.shape[1]),
                value=torch.finfo(logits.dtype).min,
            )
        values, absolute_indices = torch.topk(
            scores,
            k=topk_tokens,
            dim=1,
            largest=True,
            sorted=False,
        )
        relative_indices = absolute_indices - cu_seqlen_ks[:, None]
        selected_valid = values != torch.finfo(logits.dtype).min
        relative_indices = torch.where(
            selected_valid,
            relative_indices,
            torch.full_like(relative_indices, -1),
        )
        raw_topk_indices[:num_rows, :topk_tokens].copy_(relative_indices.to(torch.int32))

    @staticmethod
    def indexer_kpool_topk_pytorch(
        query: torch.Tensor,
        key: torch.Tensor,
        weights: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        block_table: torch.Tensor,
        query_positions: torch.Tensor,
        sparse_count: int,
        pool_size: int,
        max_key_seq_len: int,
        query_chunk_size: int = INDEXER_KPOOL_QUERY_CHUNK_SIZE,
        key_chunk_size: int = INDEXER_KPOOL_KEY_CHUNK_SIZE,
    ) -> torch.Tensor:
        """在分页 BF16 K cache 上执行 MQA logits 和逐行 top-k。"""
        if query.ndim != 3:
            raise ValueError(f"Indexer query must be [T,H,D], got {query.shape}.")
        if weights.shape != query.shape[:2]:
            raise ValueError(
                f"Indexer weights must match query T/H dimensions, got {weights.shape} and {query.shape[:2]}."
            )
        if key.ndim != 4 or key.shape[2] != 1:
            raise ValueError(f"Indexer K cache must be [blocks,block,1,dim], got {key.shape}.")
        if key.shape[-1] != query.shape[-1]:
            raise ValueError(f"Indexer query/K dims differ: {query.shape[-1]} and {key.shape[-1]}.")
        if query.dtype != torch.bfloat16 or key.dtype != torch.bfloat16:
            raise TypeError(f"Indexer query/K cache must both use bfloat16, got {query.dtype} and {key.dtype}.")
        if actual_seq_lengths_query.ndim != 1:
            raise ValueError("actual_seq_lengths_query must be cumulative 1-D.")
        if actual_seq_lengths_key.shape != actual_seq_lengths_query.shape:
            raise ValueError(
                "Indexer query/key sequence metadata must have the same "
                f"request count, got {actual_seq_lengths_query.shape} and "
                f"{actual_seq_lengths_key.shape}."
            )
        if block_table.ndim != 2 or block_table.shape[0] != actual_seq_lengths_key.shape[0]:
            raise ValueError(
                "Indexer block-table rows must match request count, got "
                f"{block_table.shape} and {actual_seq_lengths_key.shape}."
            )
        if query_positions.ndim != 1 or query_positions.shape[0] != query.shape[0]:
            raise ValueError(
                "Indexer positions must provide one value per query, got "
                f"{query_positions.shape} and {query.shape[0]} queries."
            )
        if sparse_count <= 0:
            raise ValueError(f"sparse_count must be positive, got {sparse_count}.")
        if pool_size <= 0:
            raise ValueError(f"pool_size must be positive, got {pool_size}.")
        if max_key_seq_len < 0:
            raise ValueError(f"max_key_seq_len must be non-negative, got {max_key_seq_len}.")
        if query_chunk_size <= 0 or key_chunk_size <= 0:
            raise ValueError(
                f"Indexer query/key chunk sizes must be positive, got {query_chunk_size} and {key_chunk_size}."
            )
        if block_table.shape[1] == 0 and max_key_seq_len:
            raise ValueError("Indexer block table has no columns for a non-empty cache.")

        output = torch.full(
            (query.shape[0], sparse_count),
            -1,
            dtype=torch.int32,
            device=query.device,
        )
        if query.shape[0] == 0 or max_key_seq_len == 0:
            return output

        query_ends = actual_seq_lengths_query
        token_ids = torch.arange(
            query.shape[0],
            dtype=query_ends.dtype,
            device=query.device,
        )
        request_ids = torch.bucketize(token_ids, query_ends, right=True)
        request_pool_lens = actual_seq_lengths_key[request_ids].to(torch.int64)
        causal_pool_lens = torch.div(
            query_positions.to(torch.int64) + 1,
            pool_size,
            rounding_mode="floor",
        )
        causal_pool_lens = torch.minimum(causal_pool_lens, request_pool_lens)

        cache_block_size = key.shape[1]
        key_chunk_size = max(
            cache_block_size,
            key_chunk_size // cache_block_size * cache_block_size,
        )
        score_mask_value = torch.finfo(torch.float32).min

        for query_start in range(0, query.shape[0], query_chunk_size):
            query_end = min(query_start + query_chunk_size, query.shape[0])
            chunk_query = query[query_start:query_end]
            chunk_weights = weights[query_start:query_end]
            chunk_request_ids = request_ids[query_start:query_end]
            chunk_pool_lens = causal_pool_lens[query_start:query_end]
            chunk_rows = query_end - query_start
            best_values = torch.full(
                (chunk_rows, sparse_count),
                score_mask_value,
                dtype=torch.float32,
                device=query.device,
            )
            best_indices = torch.full(
                (chunk_rows, sparse_count),
                -1,
                dtype=torch.int64,
                device=query.device,
            )

            for key_start in range(0, max_key_seq_len, key_chunk_size):
                key_end = min(key_start + key_chunk_size, max_key_seq_len)
                keys_per_row = key_end - key_start
                if key_start % cache_block_size:
                    raise ValueError(
                        "Indexer key chunks must start on compressed cache "
                        f"block boundaries, got key_start={key_start} and "
                        f"block_size={cache_block_size}."
                    )
                gather_cu_seq_lens = (
                    torch.arange(
                        chunk_rows + 1,
                        dtype=torch.int32,
                        device=query.device,
                    )
                    * keys_per_row
                )
                gathered_key = torch.empty(
                    (chunk_rows * keys_per_row, query.shape[-1]),
                    dtype=torch.bfloat16,
                    device=query.device,
                )
                first_page = key_start // cache_block_size
                last_page = (key_end + cache_block_size - 1) // cache_block_size
                chunk_block_table = block_table[
                    chunk_request_ids,
                    first_page:last_page,
                ]
                AscendSparseAttnIndexerKpool.cp_gather_indexer_k_cache(
                    key,
                    gathered_key,
                    chunk_block_table,
                    gather_cu_seq_lens,
                )
                cu_seqlen_ks = gather_cu_seq_lens[:-1]
                valid_counts = (chunk_pool_lens - key_start).clamp(min=0, max=keys_per_row).to(torch.int32)
                cu_seqlen_ke = cu_seqlen_ks + valid_counts
                logits = AscendSparseAttnIndexerKpool.bf16_mqa_logits(
                    chunk_query,
                    gathered_key,
                    chunk_weights,
                    cu_seqlen_ks,
                    cu_seqlen_ke,
                    clean_logits=False,
                )
                chunk_topk = torch.empty(
                    (chunk_rows, sparse_count),
                    dtype=torch.int32,
                    device=query.device,
                )
                AscendSparseAttnIndexerKpool.top_k_per_row_prefill(
                    logits,
                    cu_seqlen_ks,
                    cu_seqlen_ke,
                    chunk_topk,
                    chunk_rows,
                    logits.stride(0),
                    logits.stride(1),
                    sparse_count,
                )
                chunk_valid = chunk_topk >= 0
                safe_chunk_topk = chunk_topk.clamp_min(0).to(torch.int64)
                absolute_workspace_indices = cu_seqlen_ks[:, None].to(torch.int64) + safe_chunk_topk
                chunk_values = torch.gather(
                    logits,
                    1,
                    absolute_workspace_indices,
                )
                chunk_values = chunk_values.masked_fill(
                    ~chunk_valid,
                    score_mask_value,
                )
                candidate_indices = torch.where(
                    chunk_valid,
                    safe_chunk_topk + key_start,
                    torch.full_like(safe_chunk_topk, -1),
                )
                merged_values = torch.cat([best_values, chunk_values], dim=1)
                merged_indices = torch.cat([best_indices, candidate_indices], dim=1)
                best_values, selected = torch.topk(
                    merged_values,
                    k=sparse_count,
                    dim=1,
                    largest=True,
                    sorted=False,
                )
                best_indices = torch.gather(merged_indices, 1, selected)

            best_indices = torch.where(
                best_values == score_mask_value,
                torch.full_like(best_indices, -1),
                best_indices,
            )
            output[query_start:query_end] = best_indices.to(torch.int32)

        return output

    @staticmethod
    def kpool_compress_and_write_cache(
        kv_cache: torch.Tensor,
        slot_k: torch.Tensor,
        slot_score: torch.Tensor,
        ape: torch.Tensor,
        loc: torch.Tensor,
        pool_size: int,
        head_dim: int = INDEXER_KPOOL_HEAD_DIM,
        write_mask: torch.Tensor | None = None,
        round_scale: bool = True,
        return_compressed: bool = False,
        write_cache: bool = True,
    ) -> torch.Tensor | None:
        """将 pool 压缩为 BF16 K，并直接写入分页 Indexer cache。"""
        del round_scale
        if slot_k.ndim != 3:
            raise ValueError(f"slot_k must be 3-D, got shape {slot_k.shape}.")
        if slot_k.dtype != torch.bfloat16:
            raise TypeError(f"slot_k must use bfloat16, got {slot_k.dtype}.")
        if slot_score.shape != slot_k.shape:
            raise ValueError(f"slot_score must match slot_k, got {slot_score.shape} and {slot_k.shape}.")
        if ape.dtype != torch.float32:
            raise TypeError(f"ape must use float32, got {ape.dtype}.")
        if ape.shape != slot_k.shape[1:]:
            raise ValueError(f"ape must have shape {slot_k.shape[1:]}, got {ape.shape}.")
        if slot_k.shape[1] != pool_size or slot_k.shape[2] != head_dim:
            raise ValueError(
                f"slot_k shape does not match pool_size/head_dim: {slot_k.shape}, {pool_size}, {head_dim}."
            )
        if loc.dtype != torch.int64:
            raise TypeError(f"loc must use int64, got {loc.dtype}.")
        if loc.shape != (slot_k.shape[0],):
            raise ValueError(f"loc must have shape {(slot_k.shape[0],)}, got {loc.shape}.")
        if not write_cache and not return_compressed:
            raise ValueError("At least one output must be requested.")
        if write_mask is not None:
            if write_mask.shape != (slot_k.shape[0],):
                raise ValueError(f"write_mask must have one value per pool, got {write_mask.shape}.")
            if return_compressed:
                raise ValueError("return_compressed cannot be combined with write_mask.")

        indexer_k_cache = kv_cache
        if indexer_k_cache.dtype != torch.bfloat16:
            raise TypeError(f"Ascend indexer K cache must use bfloat16, got {indexer_k_cache.dtype}.")
        if indexer_k_cache.ndim != 4 or indexer_k_cache.shape[2:] != (1, head_dim):
            raise ValueError(
                "Ascend indexer K cache must have shape "
                f"[blocks,block,1,{head_dim}], got {indexer_k_cache.shape}."
            )

        if slot_k.shape[0] == 0:
            if return_compressed:
                return torch.empty(
                    (0, head_dim),
                    dtype=torch.bfloat16,
                    device=slot_k.device,
                )
            return None

        scores = slot_score.float() + ape.float().unsqueeze(0)
        compressed_k = (torch.softmax(scores, dim=1) * slot_k.float()).sum(dim=1).to(torch.bfloat16)

        if write_cache:
            write_locs = loc
            write_k = compressed_k
            if write_mask is not None:
                selected = write_mask.nonzero().flatten()
                write_locs = write_locs[selected]
                write_k = write_k[selected]
            AscendSparseAttnIndexerKpool._scatter_paged_cache(
                indexer_k_cache,
                write_locs,
                write_k.view(-1, head_dim),
                indexer_k_cache.shape[1],
            )

        if return_compressed:
            return compressed_k
        return None

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        weights: torch.Tensor,
        *,
        wk: torch.Tensor,
        gate_weight: torch.Tensor,
        norm_weight: torch.Tensor | None = None,
        norm_bias: torch.Tensor | None = None,
        compress_ape: torch.Tensor | None = None,
        index_kpool: int = 1,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_native(
            hidden_states,
            q_quant,
            weights,
            wk=wk,
            gate_weight=gate_weight,
            norm_weight=norm_weight,
            norm_bias=norm_bias,
            compress_ape=compress_ape,
            index_kpool=index_kpool,
            positions=positions,
        )

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        weights: torch.Tensor,
        *,
        wk: torch.Tensor,
        gate_weight: torch.Tensor,
        norm_weight: torch.Tensor | None = None,
        norm_bias: torch.Tensor | None = None,
        compress_ape: torch.Tensor | None = None,
        index_kpool: int = 1,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_ascend(
            hidden_states,
            q_quant,
            weights,
            wk=wk,
            gate_weight=gate_weight,
            norm_weight=norm_weight,
            norm_bias=norm_bias,
            compress_ape=compress_ape,
            index_kpool=index_kpool,
            positions=positions,
        )

    def forward_ascend(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        weights: torch.Tensor,
        *,
        wk: torch.Tensor,
        gate_weight: torch.Tensor,
        norm_weight: torch.Tensor | None = None,
        norm_bias: torch.Tensor | None = None,
        compress_ape: torch.Tensor | None = None,
        index_kpool: int = 1,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the full KeyPool compress and PoolKeyIndexer select sequence."""
        if self.use_fp4_cache:
            raise ValueError("Ascend GLM-5 Indexer uses BF16 Q, not FP4.")
        if isinstance(q_quant, tuple):
            q_values, q_scale = q_quant
            if q_scale is not None:
                q_values = q_values * q_scale.unsqueeze(-1).to(q_values.dtype)
        else:
            q_values = q_quant
        if compress_ape is None or positions is None:
            raise ValueError("GLM-5 kpool requires compress_ape and positions.")

        context = get_forward_context()
        metadata = context.attn_metadata
        if not isinstance(metadata, dict):
            raise TypeError("GLM-5 Indexer requires per-layer attention metadata.")
        state_metadata = metadata[self.state_cache.prefix]
        indexer_metadata = metadata[self.k_cache.prefix]
        attn_metadata = metadata[self.attn_layer_name]
        state_cache = self._bound_cache(self.state_cache)
        indexer_cache = self._bound_cache(self.k_cache)
        if not isinstance(state_cache, torch.Tensor):
            raise TypeError("GLM-5 compressor state cache must be one tensor.")
        if not isinstance(indexer_cache, torch.Tensor) or indexer_cache.dtype != torch.bfloat16:
            raise TypeError("GLM-5 indexer cache must be one bfloat16 K tensor.")

        is_full_graph = context.cudagraph_runtime_mode == CUDAGraphMode.FULL
        num_tokens = positions.shape[0] if is_full_graph else min(attn_metadata.num_actual_tokens, positions.shape[0])
        cum_query_lens = attn_metadata.cum_query_lens
        cu_seqlens = torch.cat(
            [
                torch.zeros(1, dtype=cum_query_lens.dtype, device=cum_query_lens.device),
                cum_query_lens,
            ]
        )
        start_pos = positions[:num_tokens][cu_seqlens[:-1].clamp_max(num_tokens - 1)].to(torch.int32)

        pooled_key = torch_npu.key_pool(
            hidden_states[:num_tokens],
            wk,
            gate_weight,
            compress_ape,
            state_cache,
            state_metadata.block_table,
            start_pos,
            norm_weight=norm_weight,
            norm_bias=norm_bias,
            cu_seqlens=cu_seqlens,
            cmp_ratio=index_kpool,
        )

        selected = (
            torch.arange(num_tokens, device=hidden_states.device)
            if is_full_graph
            else (indexer_metadata.slot_mapping[:num_tokens] >= 0).nonzero().flatten()
        )
        if is_full_graph or selected.numel() > 0:
            token_ids = torch.arange(num_tokens, device=hidden_states.device)
            request_ids = torch.bucketize(
                token_ids,
                cu_seqlens,
                right=True,
            ).clamp_max(attn_metadata.seq_lens.shape[0] - 1)
            first_pool = torch.div(
                start_pos,
                index_kpool,
                rounding_mode="floor",
            )
            pool_idx = torch.div(
                positions[:num_tokens][selected],
                index_kpool,
                rounding_mode="floor",
            )
            rows = pooled_key[
                request_ids[selected],
                (pool_idx - first_pool[request_ids[selected]]).clamp(0, pooled_key.shape[1] - 1),
            ]
            self._scatter_paged_cache(
                indexer_cache,
                indexer_metadata.slot_mapping[selected].to(torch.int64),
                rows,
                indexer_cache.shape[1],
            )

        indices, _ = torch_npu.pool_key_indexer(
            q_values[:num_tokens],
            indexer_cache,
            weights[:num_tokens].to(q_values.dtype),
            (attn_metadata.seq_lens - indexer_metadata.seq_lens * index_kpool).to(torch.int32),
            actual_seq_q=cum_query_lens,
            actual_seq_k=indexer_metadata.seq_lens,
            block_table=indexer_metadata.block_table,
            layout_q="TND",
            layout_k="PA_BBND",
            topk=self.topk_tokens,
            pool_size=index_kpool,
            mask_mode=3,
            return_value=False,
        )
        return indices.unsqueeze(1)


class AscendGlm5NextIndexer(nn.Module):
    """GLM-5 indexer weights and its compressed/tail cache ownership."""

    def __init__(
        self,
        config: Glm5NextTextConfig,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig,
        topk_indices_buffer: torch.Tensor | None,
        prefix: str,
    ) -> None:
        super().__init__()
        assert config.index_topk is not None
        assert config.index_n_heads is not None
        assert config.index_head_dim is not None
        assert config.index_kpool is not None
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.index_kpool = config.index_kpool
        self.is_rope_neox_style = not getattr(config, "indexer_rope_interleave", False)
        self.q_lora_rank = q_lora_rank
        self.topk_indices_buffer = topk_indices_buffer
        self.softmax_scale = self.head_dim**-0.5

        self.index_kpool_compress_ape = nn.Parameter(
            torch.zeros(
                self.index_kpool,
                self.head_dim,
                dtype=torch.float32,
            )
        )
        self.index_kpool_compress_gate = nn.Parameter(
            torch.empty(
                self.head_dim,
                hidden_size,
                dtype=torch.bfloat16,
            )
        )
        self.wq_b = ReplicatedLinear(
            q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.wk_weights_proj = MergedColumnParallelLinear(
            hidden_size,
            [self.head_dim, self.n_head],
            bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.wk_weights_proj",
        )
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
        self.state_cache = AscendGlm5NextCompressorStateCache(
            state_dim=2 * self.head_dim,
            dtype=torch.bfloat16,
            compress_ratio=self.index_kpool,
            cache_config=cache_config,
            prefix=f"{prefix}.compressor.state_cache",
        )
        # 压缩后的 Indexer K 直接使用 BF16 存储，不再分配量化 scale cache。
        self.k_cache = AscendGlm5NextIndexerKPoolCache(
            head_dim=self.head_dim,
            dtype=torch.bfloat16,
            cache_role="indexer",
            cache_config=cache_config,
            prefix=f"{prefix}.k_cache",
            compress_ratio=self.index_kpool,
        )
        vllm_config = get_current_vllm_config()
        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_total_seq_len = get_max_prefill_buffer_size(vllm_config)
        self.prefix = prefix
        self.quant_block_size = self.head_dim
        self.scale_fmt = None
        self.indexer_op = AscendSparseAttnIndexerKpool(
            self.k_cache,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            state_cache=self.state_cache,
            attn_layer_name=f"{prefix.removesuffix('.indexer')}.attn",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb,
    ) -> torch.Tensor:
        """Match upstream GLM-5 Indexer forward with BF16 NPU inputs."""
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)

        kw, _ = self.wk_weights_proj(hidden_states)
        weights = kw[:, self.head_dim :]

        if self.rope_dim > 0:
            if rotary_emb is None:
                raise ValueError("GLM-5 Indexer requires rotary embedding when qk_rope_head_dim is non-zero.")
            q_pe, q_nope = torch.split(
                q,
                [self.rope_dim, self.head_dim - self.rope_dim],
                dim=-1,
            )
            q_pe, _ = rotary_emb(
                positions,
                q_pe,
                q_pe.unsqueeze(1),
            )
            q_pe = q_pe.reshape(-1, self.n_head, self.rope_dim)
            q = torch.cat([q_pe, q_nope], dim=-1)
        q = q.to(torch.bfloat16)

        # Upstream folds the FP8 Q scale into weights. Q remains BF16 on
        # Ascend, so only the model-level factors are required.
        weights = weights * (self.softmax_scale * self.n_head**-0.5)
        return self.indexer_op(
            hidden_states,
            q,
            weights,
            wk=self.wk_weights_proj.weight[: self.head_dim],
            gate_weight=self.index_kpool_compress_gate,
            norm_weight=self.k_norm.weight,
            norm_bias=self.k_norm.bias,
            compress_ape=self.index_kpool_compress_ape,
            index_kpool=self.index_kpool,
            positions=positions,
        )


class AscendGlm5NextMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        is_sequence_parallel: bool = False,
        prefix: str = "",
        swiglu_limit: float | None = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.down_proj",
        )
        if swiglu_limit is not None:
            self.act_fn = SiluAndMulWithClamp(swiglu_limit=swiglu_limit)
        else:
            self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class AscendGlm5NextMoE(nn.Module):
    def __init__(
        self,
        config: Glm5NextTextConfig,
        parallel_config: ParallelConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts: int = config.n_routed_experts
        self.n_shared_experts: int = config.n_shared_experts
        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe
        swiglu_limit = getattr(config, "swiglu_limit", None) or 0.0

        if config.n_shared_experts is None:
            self.shared_experts = None
        else:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = AscendGlm5NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                is_sequence_parallel=self.is_sequence_parallel,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
                swiglu_limit=swiglu_limit,
            )

        self.gate = GateLinear(
            config.hidden_size,
            config.n_routed_experts,
            params_dtype=torch.float32,
            out_dtype=torch.float32,
            force_fp32_compute=True,
            prefix=f"{prefix}.gate",
        )
        if getattr(config, "topk_method", None) == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(torch.empty(config.n_routed_experts, dtype=torch.float32))
        else:
            self.gate.e_score_correction_bias = None

        eplb_config = parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size
        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = self.physical_expert_start + self.n_local_physical_experts

        self.experts = FusedMoE(
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=getattr(config, "norm_topk_prob", True),
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=getattr(config, "n_group", 1),
            topk_group=getattr(config, "topk_group", 1),
            prefix=f"{prefix}.experts",
            scoring_func=getattr(config, "scoring_func", "sigmoid"),
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            n_shared_experts=None,
            swiglu_limit=swiglu_limit,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)
        if self.experts.is_internal_router:
            final_hidden_states = self.experts(hidden_states=hidden_states, router_logits=hidden_states)
        else:
            router_logits, _ = self.gate(hidden_states)
            final_hidden_states = self.experts(hidden_states=hidden_states, router_logits=router_logits)
        if self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(final_hidden_states, 0)
            final_hidden_states = final_hidden_states[:num_tokens]
        return final_hidden_states.view(num_tokens, hidden_dim)


class AscendGlm5NextLinearAttention(nn.Module, MambaBase):
    def __init__(
        self,
        config: Glm5NextTextConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.vllm_config = vllm_config
        self.config = config
        kda_config = resolve_kda_config(config)
        self.hidden_size = config.hidden_size
        self.head_dim = kda_config["head_dim"]
        self.num_heads = kda_config["num_heads"]
        tp_size = get_tensor_model_parallel_world_size()
        assert self.num_heads % tp_size == 0
        self.local_num_heads = self.num_heads // tp_size
        self.conv_kernel_size = kda_config.get("short_conv_kernel_size", 4)
        linear_lower_bound = kda_config.get("lower_bound")
        if linear_lower_bound is not None:
            self.kda_safe_gate = True
            self.kda_lower_bound = linear_lower_bound
        else:
            self.kda_safe_gate = kda_config.get("safe_gate", False)
            self.kda_lower_bound = kda_config.get("lower_bound", -5.0)
        self.layer_idx = extract_layer_index(prefix)
        self.rms_norm_eps = config.rms_norm_eps
        self.cache_config = vllm_config.cache_config
        self.model_config = vllm_config.model_config
        self.tp_size = tp_size
        self.tp_rank = get_tensor_model_parallel_rank()
        self.speculative_config = vllm_config.speculative_config
        self.num_spec = self.speculative_config.num_speculative_tokens if self.speculative_config else 0

        projection_size = self.head_dim * self.num_heads
        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            projection_size,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.q_proj",
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size,
            projection_size,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size,
            projection_size,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.v_proj",
        )
        self.f_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.f_a_proj",
        )
        self.f_b_proj = ColumnParallelLinear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.f_b_proj",
        )
        self.dt_bias = nn.Parameter(torch.empty(projection_size // tp_size, dtype=torch.float32))
        from vllm.model_executor.utils import set_weight_attrs

        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        self.b_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_heads,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.b_proj",
        )
        self.q_conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=projection_size,
            bias=False,
            prefix=f"{prefix}.q_conv1d",
        )
        self.k_conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=projection_size,
            bias=False,
            prefix=f"{prefix}.k_conv1d",
        )
        self.v_conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=projection_size,
            bias=False,
            prefix=f"{prefix}.v_conv1d",
        )
        self.q_conv1d.weight.data = self.q_conv1d.weight.data.unsqueeze(1)
        self.k_conv1d.weight.data = self.k_conv1d.weight.data.unsqueeze(1)
        self.v_conv1d.weight.data = self.v_conv1d.weight.data.unsqueeze(1)
        self.A_log = nn.Parameter(torch.empty(1, 1, self.local_num_heads, 1, dtype=torch.float32))

        def a_log_weight_loader(
            param: torch.Tensor,
            loaded_weight: torch.Tensor,
        ):
            if loaded_weight.ndim == 1:
                loaded_weight = loaded_weight.view(1, 1, -1, 1)
            return sharded_weight_loader(2)(param, loaded_weight)

        set_weight_attrs(self.A_log, {"weight_loader": a_log_weight_loader})

        self.g_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.g_a_proj",
        )
        self.g_b_proj = ColumnParallelLinear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.g_b_proj",
        )
        self.o_proj = RowParallelLinear(
            projection_size,
            self.hidden_size,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.o_proj",
        )
        self.o_norm = AscendGlm5NextGatedRMSNormParams(self.head_dim)

        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.GDN_ATTN

    def get_attn_backend(self) -> type[AttentionBackend]:
        return AscendGDNAttentionBackend

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.kda_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.kda_state_shape(
            self.tp_size,
            self.num_heads,
            self.head_dim,
            conv_kernel_size=self.conv_kernel_size,
            num_spec=self.num_spec,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        beta, _ = self.b_proj(hidden_states)
        beta = beta.float().sigmoid()
        g1, _ = self.f_b_proj(self.f_a_proj(hidden_states)[0])
        beta = beta.unsqueeze(0)
        g1 = rearrange(g1, "n (h d) -> 1 n h d", d=self.head_dim)

        g_proj_states, _ = self.g_b_proj(self.g_a_proj(hidden_states)[0])
        gate = rearrange(g_proj_states, "... (h d) -> ... h d", d=self.head_dim)

        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        if attn_metadata_raw is None:
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata = attn_metadata_raw[self.prefix]
        assert isinstance(attn_metadata, GDNAttentionMetadata)

        has_initial_state = attn_metadata.has_initial_state
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor
        num_actual_tokens = attn_metadata.num_actual_tokens

        q = q[:num_actual_tokens]
        k = k[:num_actual_tokens]
        v = v[:num_actual_tokens]
        g1 = g1[:, :num_actual_tokens]
        beta = beta[:, :num_actual_tokens]
        gate = gate[:num_actual_tokens]

        conv_state, recurrent_state = self.kv_cache
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)

        conv_state_q, conv_state_k, conv_state_v = conv_state.chunk(3, dim=-2)

        q_conv_weights = self.q_conv1d.weight.view(self.q_conv1d.weight.size(0), self.q_conv1d.weight.size(2))
        k_conv_weights = self.k_conv1d.weight.view(self.k_conv1d.weight.size(0), self.k_conv1d.weight.size(2))
        v_conv_weights = self.v_conv1d.weight.view(self.v_conv1d.weight.size(0), self.v_conv1d.weight.size(2))

        state_len = self.conv_kernel_size - 1

        if attn_metadata.num_prefills > 0:
            q_t = q.transpose(0, 1).unsqueeze(0)
            k_t = k.transpose(0, 1).unsqueeze(0)
            v_t = v.transpose(0, 1).unsqueeze(0)
            q_out = torch.empty_like(q_t)
            k_out = torch.empty_like(k_t)
            v_out = torch.empty_like(v_t)
            seq_begin_end_idx = [
                (int(non_spec_query_start_loc[i].item()), int(non_spec_query_start_loc[i + 1].item()))
                for i in range(non_spec_query_start_loc.shape[0] - 1)
            ]
            for seq_idx, (bos, eos) in enumerate(seq_begin_end_idx):
                slot = int(non_spec_state_indices_tensor[seq_idx].item())
                for x_t, out_t, w, cs in [
                    (q_t, q_out, q_conv_weights, conv_state_q),
                    (k_t, k_out, k_conv_weights, conv_state_k),
                    (v_t, v_out, v_conv_weights, conv_state_v),
                ]:
                    seq_x = x_t[:, :, bos:eos]
                    if bool(has_initial_state[seq_idx].item()):
                        init = cs[slot, :, :state_len].unsqueeze(0)
                    else:
                        init = torch.zeros(1, w.shape[0], state_len, device=seq_x.device, dtype=seq_x.dtype)
                    conv_input = torch.cat([init, seq_x], dim=-1).to(w.dtype)
                    w_3d = w.unsqueeze(1)
                    seq_res = torch.nn.functional.conv1d(conv_input, w_3d, None, padding=0, groups=w.shape[0])
                    seq_res = torch.nn.functional.silu(seq_res[..., -seq_x.shape[-1] :])
                    out_t[:, :, bos:eos] = seq_res.to(dtype=x_t.dtype)
                    cs[slot, :, :state_len].copy_(conv_input[..., -state_len:].squeeze(0))
            q_conv = q_out.squeeze(0).transpose(0, 1)
            k_conv = k_out.squeeze(0).transpose(0, 1)
            v_conv = v_out.squeeze(0).transpose(0, 1)
        else:
            decode_conv_indices = non_spec_state_indices_tensor[:num_actual_tokens]
            q_conv = torch.empty_like(q)
            k_conv = torch.empty_like(k)
            v_conv = torch.empty_like(v)
            for x_flat, out_flat, w, cs in [
                (q, q_conv, q_conv_weights, conv_state_q),
                (k, k_conv, k_conv_weights, conv_state_k),
                (v, v_conv, v_conv_weights, conv_state_v),
            ]:
                for t in range(num_actual_tokens):
                    slot = int(decode_conv_indices[t].item())
                    x_t = x_flat[t : t + 1, :].unsqueeze(0).transpose(1, 2)
                    cs_t = cs[slot, :, :state_len].unsqueeze(0)
                    conv_input = torch.cat([cs_t, x_t], dim=-1).to(w.dtype)
                    w_3d = w.unsqueeze(1)
                    res = torch.nn.functional.conv1d(conv_input, w_3d, None, padding=0, groups=w.shape[0])
                    res = torch.nn.functional.silu(res[..., -1:])
                    out_flat[t : t + 1, :] = res.squeeze(0).transpose(0, 1).to(dtype=x_flat.dtype)
                    cs[slot, :, :state_len].copy_(conv_input[..., -state_len:].squeeze(0))

        q_conv = rearrange(q_conv, "n (h d) -> 1 n h d", d=self.head_dim)
        k_conv = rearrange(k_conv, "n (h d) -> 1 n h d", d=self.head_dim)
        v_conv = rearrange(v_conv, "n (h d) -> 1 n h d", d=self.head_dim)

        if attn_metadata.num_prefills > 0:
            assert non_spec_state_indices_tensor is not None
            assert has_initial_state is not None
            zero_idx = non_spec_state_indices_tensor[~has_initial_state]
            recurrent_state[zero_idx] = 0
            initial_state = recurrent_state[non_spec_state_indices_tensor].contiguous()
            g = fused_kda_gate(
                rearrange(g1, "1 n h d -> n (h d)"),
                self.A_log,
                self.head_dim,
                g_bias=self.dt_bias,
                safe_gate=self.kda_safe_gate,
                lower_bound=self.kda_lower_bound,
            ).unsqueeze(0)
            core_attn_out, last_recurrent_state = chunk_kda(
                q_conv,
                k_conv,
                v_conv,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=non_spec_query_start_loc,
            )
            recurrent_state[non_spec_state_indices_tensor] = last_recurrent_state
        else:
            assert non_spec_query_start_loc is not None
            g = fused_kda_gate(
                rearrange(g1, "1 n h d -> n (h d)"),
                self.A_log,
                self.head_dim,
                g_bias=self.dt_bias,
                safe_gate=self.kda_safe_gate,
                lower_bound=self.kda_lower_bound,
            ).unsqueeze(0)
            core_attn_out, _ = fused_recurrent_kda_fwd(
                q_conv,
                k_conv,
                v_conv,
                g=g,
                beta=beta,
                scale=self.head_dim**-0.5,
                initial_state=recurrent_state,
                inplace_final_state=True,
                cu_seqlens=non_spec_query_start_loc[: attn_metadata.num_decodes + 1],
                ssm_state_indices=non_spec_state_indices_tensor,
                use_qk_l2norm_in_kernel=True,
            )

        core_attn_out = rearrange(core_attn_out, "1 n h d -> (n h) d", d=self.head_dim)
        gate = rearrange(gate, "n h d -> (n h) d", d=self.head_dim)
        core_attn_out = rms_norm_gated(
            core_attn_out,
            gate,
            self.o_norm.weight,
            self.o_norm.bias,
            activation="sigmoid",
            eps=self.rms_norm_eps,
        )
        core_attn_out = rearrange(core_attn_out, "(n h) d -> n (h d)", d=self.head_dim, h=self.local_num_heads)
        output[:num_actual_tokens], _ = self.o_proj(core_attn_out)


class AscendGlm5NextMLAAttention(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: Glm5NextTextConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        max_position_embeddings: int = 8192,
        cache_config=None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        skip_rope: bool = False,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size
        self.scaling = self.qk_head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings
        self.prefix = prefix

        proj_input_size = hidden_size
        if self.q_lora_rank is not None:
            self.fused_qkv_a_proj = MergedColumnParallelLinear(
                proj_input_size,
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                bias=False,
                quant_config=quant_config,
                disable_tp=True,
                prefix=f"{prefix}.fused_qkv_a_proj",
            )
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
        else:
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                proj_input_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )
            self.q_proj = ColumnParallelLinear(
                proj_input_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        if qk_rope_head_dim > 0 and not skip_rope and config.rope_parameters is not None:
            rope_params = config.rope_parameters
            if rope_params.get("rope_type", "default") != "default":
                rope_params = dict(rope_params)
                rope_params["rope_type"] = (
                    "deepseek_yarn" if rope_params.get("apply_yarn_scaling", True) else "deepseek_llama_scaling"
                )
            self.rotary_emb = get_rope(
                qk_rope_head_dim,
                max_position=max_position_embeddings,
                rope_parameters=rope_params,
                is_neox_style=False,
            )
            if rope_params.get("rope_type") == "deepseek_yarn":
                mscale_all_dim = rope_params.get("mscale_all_dim", False)
                scaling_factor = rope_params.get("factor", 1.0)
                mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
                self.scaling = self.scaling * mscale * mscale
        else:
            self.rotary_emb = None

        if qk_rope_head_dim > 0:
            if config.rope_parameters is None:
                raise ValueError("GLM-5 Indexer requires rope_parameters when qk_rope_head_dim is non-zero.")
            self.indexer_rotary_emb = get_rope(
                qk_rope_head_dim,
                max_position=max_position_embeddings,
                rope_parameters=config.rope_parameters,
                is_neox_style=not getattr(
                    config,
                    "indexer_rope_interleave",
                    False,
                ),
            )
        else:
            self.indexer_rotary_emb = None

        if self.q_lora_rank is None:
            raise ValueError("GLM-5 Indexer KPool MLA requires q_lora_rank.")
        if cache_config is None:
            raise ValueError("GLM-5 Indexer KPool MLA requires cache_config.")
        if getattr(config, "index_topk", None) is None:
            raise ValueError("GLM-5 Indexer KPool MLA requires indexer configuration.")

        self.indexer = AscendGlm5NextIndexer(
            config,
            hidden_size,
            self.q_lora_rank,
            quant_config,
            cache_config,
            topk_indices_buffer,
            f"{prefix}.indexer",
        )
        self.indexer_kpool_mla_attention = AscendIndexerKPoolMLAAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_local_heads,
            scale=self.scaling,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            modules=IndexerKPoolMLAModules(
                fused_qkv_a_proj=self.fused_qkv_a_proj,
                q_a_layernorm=self.q_a_layernorm,
                q_b_proj=self.q_b_proj,
                kv_a_layernorm=self.kv_a_layernorm,
                kv_b_proj=self.kv_b_proj,
                rotary_emb=self.rotary_emb,
                indexer_rotary_emb=self.indexer_rotary_emb,
                o_proj=self.o_proj,
                indexer=self.indexer,
                topk_indices_buffer=topk_indices_buffer,
            ),
            cache_config=cache_config,
            prefix=prefix,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        result = self.indexer_kpool_mla_attention(positions, hidden_states)
        output[:] = result


class AscendGlm5NextDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: Glm5NextTextConfig,
        layer_idx: int,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        is_mtp_layer: bool = False,
    ) -> None:
        super().__init__()
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.is_moe = _is_moe(config)
        self.num_hidden_layers = config.num_hidden_layers
        self.rms_norm_eps = config.rms_norm_eps
        self.num_experts = config.n_routed_experts
        self.is_mtp_layer = is_mtp_layer
        self.mhc = config.mhc
        self.layer_kind = "kda" if _is_kda_layer(config, layer_idx) else "mla"
        self.is_vl_first_layer = bool(
            uses_global_inputs_embeds(vllm_config, "vision_chunk") and self.layer_idx == 0
        )

        if _is_kda_layer(config, layer_idx):
            self.self_attn = KimiGatedDeltaNetAttention(
                config,
                vllm_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            assert config.v_head_dim is not None
            assert config.kv_lora_rank is not None
            self.self_attn = AscendGlm5NextMLAAttention(
                vllm_config=vllm_config,
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                max_position_embeddings=config.max_position_embeddings,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                skip_rope=getattr(config, "mla_nope", False),
                topk_indices_buffer=topk_indices_buffer,
            )

        mlp_layer_types = getattr(config, "mlp_layer_types", None)
        if mlp_layer_types:
            mlp_type = mlp_layer_types[layer_idx] if layer_idx < len(mlp_layer_types) else mlp_layer_types[-1]
        else:
            mlp_type = "sparse" if layer_idx >= config.first_k_dense_replace else "dense"
        if self.is_moe and self.num_experts is not None and mlp_type == "sparse":
            self.mlp = AscendGlm5NextMoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = AscendGlm5NextMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                swiglu_limit=config.swiglu_limit,
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if self.mhc and not self.is_mtp_layer:
            self.hc_mult = config.mhc_num_residual_streams
            self.hc_sinkhorn_iters = config.mhc_sinkhorn_iterations or 20
            self.hc_eps = config.hc_eps or 1e-6
            self.hc_post_mult_value = config.mhc_post_mult_value or 2.0
            n = config.mhc_num_residual_streams
            d_model = n * self.hidden_size
            mix_hc = (2 + n) * n
            self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, d_model, dtype=torch.float32))
            self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
            self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
            self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, d_model, dtype=torch.float32))
            self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
            self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    def hc_pre(self, x, hc_fn, hc_scale, hc_base):
        n = self.hc_mult
        d = self.hidden_size
        x_mhc = x.view(x.shape[0], n, d)
        y, post, comb = torch.ops._C_ascend.npu_hc_pre_v2(
            x_mhc,
            hc_fn,
            hc_scale,
            hc_base,
            n,
            self.hc_sinkhorn_iters,
            self.rms_norm_eps,
            self.hc_eps,
        )
        return y, x_mhc, post, comb

    def hc_post(self, x, residual_mhc, post, comb):
        return torch.ops._C_ascend.npu_hc_post(
            x.unsqueeze(0),
            residual_mhc.unsqueeze(0),
            post.unsqueeze(0),
            comb.unsqueeze(0),
        ).squeeze(0)

    def _make_attention_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """FlashComm1 attention layers emit the local token shard.

        The first multimodal layer still receives the full embedding sequence,
        so its attention output buffer must be sized N/tp (and the residual
        chunked by ``maybe_chunk_residual`` before the add), matching the
        o_proj reduce-scatter layout used by every other layer.
        """
        if self.is_vl_first_layer and _EXTRA_CTX.flash_comm_v1_enabled:
            tp_size = get_tensor_model_parallel_world_size()
            n_out = (hidden_states.shape[0] + tp_size - 1) // tp_size
            return torch.empty(
                (n_out, hidden_states.shape[-1]),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
        return torch.empty_like(hidden_states)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.mhc or self.is_mtp_layer:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            attn_output = self._make_attention_output(hidden_states)
            self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
                output=attn_output,
            )
            residual = torch.ops.vllm.maybe_chunk_residual(attn_output, residual)
            hidden_states = residual + attn_output
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
            return hidden_states, residual

        x = hidden_states
        if self.layer_idx == 0:
            n = self.hc_mult
            x = _expand_mhc_residual_streams(x, n)

        layer_input, residual_mhc, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        hidden_states = self.input_layernorm(layer_input)
        attn_output = self._make_attention_output(hidden_states)
        self.self_attn(hidden_states=hidden_states, positions=positions, output=attn_output)
        residual_mhc = torch.ops.vllm.maybe_chunk_residual(attn_output, residual_mhc)
        post = torch.ops.vllm.maybe_chunk_residual(attn_output, post)
        comb = torch.ops.vllm.maybe_chunk_residual(attn_output, comb)
        hidden_states = self.hc_post(attn_output, residual_mhc, post, comb)

        residual_mhc = hidden_states
        layer_input, residual_mhc_2, post, comb = self.hc_pre(
            hidden_states.view(hidden_states.shape[0], -1),
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
        )
        hidden_states = self.post_attention_layernorm(layer_input)
        x_mlp = self.mlp(hidden_states)
        hidden_states = self.hc_post(x_mlp, residual_mhc_2, post, comb)

        if self.layer_idx == self.num_hidden_layers - 1:
            n = self.hc_mult
            hidden_states = hidden_states.view(hidden_states.shape[0], n, -1).mean(dim=1)

        return hidden_states, None


@support_torch_compile
class AscendGlm5NextModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.vocab_size = config.vocab_size
        self.device = current_platform.device_type
        if getattr(config, "index_topk", None) is not None:
            assert config.index_kpool is not None
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                config.index_topk + config.index_kpool - 1,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            topk_indices_buffer = None

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        def get_layer(prefix: str):
            layer_idx = int(prefix.rsplit(".", 1)[1])
            return AscendGlm5NextDecoderLayer(
                vllm_config=vllm_config,
                config=config,
                layer_idx=layer_idx,
                prefix=prefix,
                topk_indices_buffer=topk_indices_buffer,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})

        hidden_states = self.norm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        mla_params_mapping = [
            (".fused_qkv_a_proj", ".q_a_proj", 0),
            (".fused_qkv_a_proj", ".kv_a_proj_with_mqa", 1),
            (".wk_weights_proj", ".wk", 0),
            (".wk_weights_proj", ".weights_proj", 1),
        ]
        if _is_moe(self.config):
            expert_params_mapping = fused_moe_make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=self.config.n_routed_experts,
            )
        else:
            expert_params_mapping = []
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if get_spec_layer_idx_from_weight_name(self.config, name) is not None:
                continue
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue
            if name.endswith(".conv1d.weight"):
                if loaded_weight.dim() == 3:
                    loaded_weight = loaded_weight.squeeze(1)
                q_w, k_w, v_w = loaded_weight.chunk(3, dim=0)
                for suffix, w in [(".q_conv1d.weight", q_w), (".k_conv1d.weight", k_w), (".v_conv1d.weight", v_w)]:
                    new_name = name.replace(".conv1d.weight", suffix)
                    if new_name in params_dict:
                        param = params_dict[new_name]
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, w.unsqueeze(1))
                        loaded_params.add(new_name)
                continue
            if "A_log" in name and loaded_weight.dim() == 1:
                loaded_weight = loaded_weight.view(1, 1, -1, 1)
            loaded_weight = _pad_nope_kv_a_weight(
                self.config,
                name,
                loaded_weight,
            )
            if name.endswith(".bias") and name not in params_dict:
                continue
            if is_pp_missing_parameter(name, self):
                continue

            handled = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts." in name and name not in params_dict:
                    continue
                name = name.replace(weight_name, param_name)
                if name not in params_dict:
                    break
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                handled = True
                break
            if handled:
                continue
            for param_name, weight_name, shard_id in mla_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name not in params_dict:
                    break
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                handled = True
                break
            if handled:
                continue
            for param_name, weight_name, expert_id, expert_shard_id in expert_params_mapping:
                if weight_name not in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                if is_pp_missing_parameter(mapped_name, self):
                    continue
                param = params_dict[mapped_name]
                weight_loader = param.weight_loader
                weight_loader(
                    param,
                    loaded_weight,
                    mapped_name,
                    expert_id=expert_id,
                    shard_id=expert_shard_id,
                )
                loaded_params.add(mapped_name)
                handled = True
                break
            if handled:
                continue
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params


class AscendGlm5NextForCausalLM(nn.Module, HasInnerState, SupportsPP, MixtureOfExperts, IsHybrid):
    has_inner_state: ClassVar[bool] = True
    is_hybrid: ClassVar[bool] = True
    hf_to_vllm_mapper = GLM5_TRANSFORMERS_INTERNAL_WEIGHTS_MAPPER
    packed_modules_mapping = GLM5_PACKED_MODULES_MAPPING

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.model_config = vllm_config.model_config
        self.vllm_config = vllm_config
        self.config = self.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.quant_config = quant_config
        self.model = AscendGlm5NextModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size,
            scale=logit_scale,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **kwargs,
        )
        return hidden_states

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        return MambaStateDtypeCalculator.kda_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = vllm_config.speculative_config.num_speculative_tokens if vllm_config.speculative_config else 0
        kda_config = resolve_kda_config(hf_config)
        return kimi_kda_state_shape(
            tp_size,
            kda_config["num_heads"],
            kda_config["head_dim"],
            kda_config["short_conv_kernel_size"],
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls):
        return MambaStateCopyFuncCalculator.kda_state_copy_func()

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        skip_prefixes = [MTP_ROT_WEIGHT_NAME]
        if self.config.tie_word_embeddings:
            skip_prefixes.append("lm_head.")

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=skip_prefixes,
        )

        return loader.load_weights(
            weights,
            mapper=self.hf_to_vllm_mapper,
        )
