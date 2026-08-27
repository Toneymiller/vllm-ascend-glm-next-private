# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Common worker-side logic for Mooncake KV transfer connectors."""

import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch_npu  # noqa: F401
from vllm.config import VllmConfig
from vllm.distributed import get_pcp_group
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tp_group,
)
from vllm.logger import logger
from vllm.utils.network_utils import get_ip
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.ascend_config import get_ascend_config, init_ascend_config
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import (
    MooncakeConnectorMetadata,
    MooncakeTransferMetadata,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.stats import (
    MooncakeKVConnectorStats,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.utils import (
    as_kv_cache_tensors,
    collect_configured_register_regions,
)
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import (
    global_te,
)
from vllm_ascend.distributed.kv_transfer.utils.utils import (
    get_transfer_timeout_value,
    validate_register_region_count,
)
from vllm_ascend.distributed.utils import (
    get_decode_context_model_parallel_rank,
    get_decode_context_model_parallel_world_size,
)

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig


@dataclass(frozen=True)
class KVTransferTensorLayout:
    """One tensor's physical transfer geometry.

    ``block_size_scale`` is the number of contiguous transfer units stored in
    one scheduler-visible logical block. ``transfer_unit_tokens`` is expressed
    in raw model tokens, even when the tensor stores compressed rows.
    """

    block_size_scale: int
    block_stride: int
    block_len: int
    block_shape: tuple[int, ...]
    transfer_unit_tokens: int


class MooncakeBaseConnectorWorker:
    """Worker implementation shared by Mooncake transfer modes."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        assert vllm_config.kv_transfer_config is not None

        self.vllm_config = vllm_config
        self.kv_transfer_config = vllm_config.kv_transfer_config
        if self.kv_transfer_config.is_kv_consumer == self.kv_transfer_config.is_kv_producer:
            raise ValueError(
                f"Mooncake worker requires exactly one KV transfer role, got {self.kv_transfer_config.kv_role!r}"
            )
        self.engine_id = engine_id
        self.kv_cache_config = kv_cache_config
        self.block_size = vllm_config.cache_config.block_size
        self.num_blocks = kv_cache_config.num_blocks

        init_ascend_config(vllm_config)
        self.ascend_config = get_ascend_config()
        os.environ["ASCEND_TRANSFER_TIMEOUT"] = str(get_transfer_timeout_value())

        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.tp_group = get_tp_group()
        self.pp_rank = get_pp_group().rank_in_group
        self.pp_size = vllm_config.parallel_config.pipeline_parallel_size
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank_local
        if self.dp_rank is None:
            raise ValueError("Mooncake worker requires a local DP rank")
        self.dp_size = vllm_config.parallel_config.data_parallel_size_local
        pcp_group = get_pcp_group()
        self.pcp_rank = pcp_group.rank_in_group
        self.pcp_size = pcp_group.world_size
        assert self.pcp_size == 1, f"Mooncake temporarily requires prefill context parallel size 1, got {self.pcp_size}"
        self.dcp_size = get_decode_context_model_parallel_world_size()
        self.dcp_rank = get_decode_context_model_parallel_rank() if self.dcp_size > 1 else 0

        self.max_device_id = self.tp_size * self.dp_size * self.pcp_size * self.pp_size
        self.side_channel_host = get_ip()
        self.side_channel_port = (
            self.kv_transfer_config.kv_port
            + vllm_config.parallel_config.data_parallel_rank * self.tp_size * self.pp_size * self.pcp_size
        )
        device_index = (self.pp_rank * self.pcp_size + self.pcp_rank) * self.tp_size + self.tp_rank
        self.handshake_port = self.side_channel_port + device_index

        device_name = str(torch.npu.current_device()) if self.pp_size > 1 else None
        self.engine = global_te.get_transfer_engine(
            self.side_channel_host,
            device_name=device_name,
        )
        self.te_rpc_port = self.engine.get_rpc_port()
        self.xfer_handshake_metadata: KVConnectorHandshakeMetadata | None = None
        self.xfer_stats = MooncakeKVConnectorStats()

        logger.info("Initializing Mooncake worker %s", engine_id)

    def _build_kv_cache_spec_mappings(self) -> None:
        """Flatten group specs and map each layer to its group and spec.

        A regular KV cache group contributes one spec. A uniform-type group
        contributes each distinct inner spec in layer order. Spec uniqueness is
        scoped to a group because different groups have different block tables,
        even when their specs compare equal.
        """
        self.kv_cache_specs: list[KVCacheSpec] = []
        self.layer_name_to_group_index: dict[str, int] = {}
        self.layer_name_to_spec_index: dict[str, int] = {}

        for group_index, group in enumerate(self.kv_cache_config.kv_cache_groups):
            group_spec = group.kv_cache_spec
            group_spec_indices: list[int] = []

            for layer_name in group.layer_names:
                if isinstance(group_spec, UniformTypeKVCacheSpecs):
                    layer_spec = group_spec.kv_cache_specs[layer_name]
                else:
                    layer_spec = group_spec
                spec_index = next(
                    (index for index in group_spec_indices if self.kv_cache_specs[index] == layer_spec),
                    -1,
                )
                if spec_index < 0:
                    spec_index = len(self.kv_cache_specs)
                    self.kv_cache_specs.append(layer_spec)
                    group_spec_indices.append(spec_index)

                self.layer_name_to_group_index[layer_name] = group_index
                self.layer_name_to_spec_index[layer_name] = spec_index

    def _get_layer_spec(self, layer_name: str) -> KVCacheSpec:
        return self.kv_cache_specs[self.layer_name_to_spec_index[layer_name]]

    def _get_layer_cache_role(self, layer_name: str) -> str:
        """Return a stable transfer role derived from the real layer spec."""
        spec = self._get_layer_spec(layer_name)
        cache_role = getattr(spec, "cache_role", None)
        if isinstance(cache_role, str) and cache_role:
            return cache_role

        # GLM-Next's indexer K is intentionally represented as an
        # MLAAttentionSpec so it shares the main MLA block table. Keep the
        # transfer role independent from the spec's TP-routing family.
        if (
            isinstance(spec, MLAAttentionSpec)
            and getattr(spec, "model_version", None) == "glm5_next"
            and getattr(spec, "compress_ratio", 1) > 1
            and layer_name.endswith(".indexer.k_cache")
        ):
            return "indexer_k"
        return "kv"

    def _get_default_tensor_transfer_layout(
        self,
        layer_name: str,
        cache: torch.Tensor,
    ) -> KVTransferTensorLayout:
        if cache.shape[0] % self.num_blocks != 0:
            raise ValueError(
                "Mooncake KV tensor axis 0 is not divisible by num_blocks: "
                f"layer={layer_name!r}, shape={tuple(cache.shape)}, "
                f"num_blocks={self.num_blocks}."
            )

        block_size_scale = cache.shape[0] // self.num_blocks
        if block_size_scale <= 0:
            raise ValueError(
                f"Mooncake KV tensor has invalid block scale for layer {layer_name!r}: "
                f"shape={tuple(cache.shape)}, num_blocks={self.num_blocks}."
            )

        spec = self._get_layer_spec(layer_name)
        if spec.block_size % block_size_scale != 0:
            raise ValueError(
                "Mooncake logical block size is not divisible by the physical "
                f"block scale: layer={layer_name!r}, block_size={spec.block_size}, "
                f"scale={block_size_scale}."
            )

        element_size = cache.element_size()
        block_shape = tuple(cache.shape[1:])
        return KVTransferTensorLayout(
            block_size_scale=block_size_scale,
            block_stride=cache.stride(0) * element_size,
            block_len=math.prod(block_shape) * element_size,
            block_shape=block_shape,
            transfer_unit_tokens=spec.block_size // block_size_scale,
        )

    def _get_glm_indexer_tensor_transfer_layout(
        self,
        layer_name: str,
        cache: torch.Tensor,
        kv_caches: dict[str, torch.Tensor | list[torch.Tensor]],
    ) -> KVTransferTensorLayout | None:
        """Resolve GLM-Next's inner-axis compressed indexer-K layout.

        Main MLA exposes physical transfer units along axis 0. GLM-Next
        indexer-K instead keeps one logical page on axis 0 and packs the
        compressed transfer units into axis 1. Derive its scale from the main
        MLA layer that owns the same scheduler block table, then expose each
        inner-axis chunk as one virtual contiguous Mooncake transfer block.
        """
        spec = self._get_layer_spec(layer_name)
        if self._get_layer_cache_role(layer_name) != "indexer_k":
            return None
        if not isinstance(spec, MLAAttentionSpec) or getattr(spec, "model_version", None) != "glm5_next":
            return None

        group_index = self.layer_name_to_group_index[layer_name]
        anchor_layer = next(
            (
                candidate
                for candidate, candidate_group_index in self.layer_name_to_group_index.items()
                if candidate_group_index == group_index
                and candidate in kv_caches
                and self._get_layer_cache_role(candidate) == "kv"
                and isinstance(self._get_layer_spec(candidate), MLAAttentionSpec)
                and getattr(self._get_layer_spec(candidate), "model_version", None) == "glm5_next"
                and getattr(self._get_layer_spec(candidate), "compress_ratio", 1) == 1
            ),
            None,
        )
        if anchor_layer is None:
            raise ValueError(
                f"GLM-Next indexer layer {layer_name!r} has no main MLA anchor in logical KV group {group_index}."
            )

        anchor_caches = as_kv_cache_tensors(kv_caches[anchor_layer])
        if len(anchor_caches) != 1:
            raise ValueError(
                f"GLM-Next main MLA anchor {anchor_layer!r} must expose one tensor, got {len(anchor_caches)}."
            )
        anchor_cache = anchor_caches[0]
        if anchor_cache.shape[0] % self.num_blocks != 0:
            raise ValueError(
                "GLM-Next main MLA tensor axis 0 is not divisible by num_blocks: "
                f"layer={anchor_layer!r}, shape={tuple(anchor_cache.shape)}, "
                f"num_blocks={self.num_blocks}."
            )

        block_size_scale = anchor_cache.shape[0] // self.num_blocks
        if block_size_scale <= 0 or spec.block_size % block_size_scale != 0:
            raise ValueError(
                "Invalid GLM-Next main MLA transfer geometry: "
                f"group={group_index}, block_size={spec.block_size}, "
                f"scale={block_size_scale}."
            )

        transfer_unit_tokens = spec.block_size // block_size_scale
        compress_ratio = getattr(spec, "compress_ratio", 1)
        if not isinstance(compress_ratio, int) or compress_ratio <= 0 or transfer_unit_tokens % compress_ratio != 0:
            raise ValueError(
                "Invalid GLM-Next indexer compression geometry: "
                f"layer={layer_name!r}, transfer_unit_tokens={transfer_unit_tokens}, "
                f"compress_ratio={compress_ratio}."
            )

        slots_per_unit = transfer_unit_tokens // compress_ratio
        if len(cache.shape) < 2 or cache.shape[0] != self.num_blocks:
            raise ValueError(
                "Unexpected GLM-Next indexer tensor rank/axis 0: "
                f"layer={layer_name!r}, shape={tuple(cache.shape)}, "
                f"expected_axis0={self.num_blocks}."
            )
        expected_axis1 = block_size_scale * slots_per_unit
        if cache.shape[1] != expected_axis1:
            raise ValueError(
                "Unexpected GLM-Next indexer tensor axis 1: "
                f"layer={layer_name!r}, shape={tuple(cache.shape)}, "
                f"expected_axis1={expected_axis1}, scale={block_size_scale}, "
                f"slots_per_unit={slots_per_unit}."
            )

        element_size = cache.element_size()
        unit_shape = (slots_per_unit, *cache.shape[2:])
        block_len = math.prod(unit_shape) * element_size
        block_stride = cache.stride(1) * slots_per_unit * element_size
        page_stride = cache.stride(0) * element_size
        if block_stride != block_len or page_stride != block_size_scale * block_stride:
            raise ValueError(
                "GLM-Next indexer tensor is not inner-block contiguous: "
                f"layer={layer_name!r}, shape={tuple(cache.shape)}, "
                f"stride={tuple(cache.stride())}, page_stride={page_stride}, "
                f"scale={block_size_scale}, block_stride={block_stride}, "
                f"block_len={block_len}."
            )

        return KVTransferTensorLayout(
            block_size_scale=block_size_scale,
            block_stride=block_stride,
            block_len=block_len,
            block_shape=unit_shape,
            transfer_unit_tokens=transfer_unit_tokens,
        )

    def _get_tensor_transfer_layout(
        self,
        layer_name: str,
        cache: torch.Tensor,
        kv_caches: dict[str, torch.Tensor | list[torch.Tensor]],
    ) -> KVTransferTensorLayout:
        glm_indexer_layout = self._get_glm_indexer_tensor_transfer_layout(layer_name, cache, kv_caches)
        if glm_indexer_layout is not None:
            return glm_indexer_layout
        return self._get_default_tensor_transfer_layout(layer_name, cache)

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor | list[torch.Tensor]],
    ) -> None:
        """Register configured KV cache allocations and publish metadata."""
        self.num_blocks = self.kv_cache_config.num_blocks
        logger.info("num_blocks: %s", self.num_blocks)
        self.kv_caches = kv_caches
        self._build_kv_cache_spec_mappings()
        layer_names: list[str] = []
        layer_block_sizes: list[int] = []
        group_indices: list[int] = []
        cache_roles: list[str] = []
        transfer_unit_tokens_per_layer: list[int] = []
        kv_caches_base_addr: list[list[int]] = []
        block_strides_per_layer: list[list[int]] = []
        block_lens_per_layer: list[list[int]] = []
        block_shapes_per_layer: list[list[tuple[int, ...]]] = []
        block_size_scales_per_layer: list[list[int]] = []
        configured_layer_names: set[str] = set()
        layout_summaries: dict[tuple[object, ...], tuple[str, int]] = {}

        for tensor_config in self.kv_cache_config.kv_cache_tensors:
            for layer_name in tensor_config.shared_by:
                if layer_name in configured_layer_names:
                    raise ValueError(f"Layer {layer_name!r} is referenced by more than one configured KV cache tensor.")
                if layer_name not in self.layer_name_to_group_index:
                    raise ValueError(f"Configured KV cache layer {layer_name!r} does not belong to a KV cache group.")

                cache_or_caches = kv_caches.get(layer_name)
                if cache_or_caches is None:
                    raise ValueError(f"No KV cache was registered for configured layer {layer_name!r}.")

                base_addrs: list[int] = []
                block_strides: list[int] = []
                block_lens: list[int] = []
                block_shapes: list[tuple[int, ...]] = []
                block_size_scales: list[int] = []
                transfer_unit_tokens: int | None = None

                cache_tensors = as_kv_cache_tensors(cache_or_caches)
                cache_role = self._get_layer_cache_role(layer_name)
                if cache_role == "indexer_k" and len(cache_tensors) != 1:
                    raise ValueError(
                        f"GLM-Next indexer layer {layer_name!r} must expose one tensor, got {len(cache_tensors)}."
                    )

                for cache in cache_tensors:
                    layout = self._get_tensor_transfer_layout(layer_name, cache, kv_caches)
                    if transfer_unit_tokens is None:
                        transfer_unit_tokens = layout.transfer_unit_tokens
                    elif transfer_unit_tokens != layout.transfer_unit_tokens:
                        raise ValueError(
                            f"Mooncake layer {layer_name!r} exposes tensors with different "
                            f"transfer units: {transfer_unit_tokens} and "
                            f"{layout.transfer_unit_tokens}."
                        )
                    base_addrs.append(cache.data_ptr())
                    block_strides.append(layout.block_stride)
                    block_lens.append(layout.block_len)
                    block_shapes.append(layout.block_shape)
                    block_size_scales.append(layout.block_size_scale)

                if transfer_unit_tokens is None:
                    raise ValueError(f"No transfer-unit metadata was derived for layer {layer_name!r}.")

                configured_layer_names.add(layer_name)
                layer_names.append(layer_name)
                group_index = self.layer_name_to_group_index[layer_name]
                spec_index = self.layer_name_to_spec_index[layer_name]
                layer_block_sizes.append(self.kv_cache_specs[spec_index].block_size)
                group_indices.append(group_index)
                cache_roles.append(cache_role)
                transfer_unit_tokens_per_layer.append(transfer_unit_tokens)
                kv_caches_base_addr.append(base_addrs)
                block_strides_per_layer.append(block_strides)
                block_lens_per_layer.append(block_lens)
                block_shapes_per_layer.append(block_shapes)
                block_size_scales_per_layer.append(block_size_scales)

                summary_key = (
                    group_index,
                    spec_index,
                    cache_role,
                    type(self._get_layer_spec(layer_name)).__name__,
                    self._get_layer_spec(layer_name).block_size,
                    tuple(block_size_scales),
                    transfer_unit_tokens,
                    tuple(block_lens),
                    tuple(block_shapes),
                )
                example_layer, layer_count = layout_summaries.get(summary_key, (layer_name, 0))
                layout_summaries[summary_key] = (example_layer, layer_count + 1)

        unexpected_layers = kv_caches.keys() - configured_layer_names
        if unexpected_layers:
            raise ValueError(f"KV caches contain layers absent from kv_cache_tensors: {sorted(unexpected_layers)}.")

        register_regions = collect_configured_register_regions(self.kv_cache_config, kv_caches)
        validate_register_region_count(register_regions)
        global_te.register_buffer(register_regions.ptrs, register_regions.lengths)

        transfer_metadata = MooncakeTransferMetadata(
            engine_id=self.engine_id,
            te_rpc_port=self.te_rpc_port,
            block_size=self.block_size,
            num_blocks=self.num_blocks,
            layer_names=layer_names,
            layer_block_sizes=layer_block_sizes,
            group_indices=group_indices,
            cache_roles=cache_roles,
            transfer_unit_tokens=transfer_unit_tokens_per_layer,
            kv_caches_base_addr=kv_caches_base_addr,
            block_strides=block_strides_per_layer,
            block_lens=block_lens_per_layer,
            block_shapes=block_shapes_per_layer,
            block_size_scales=block_size_scales_per_layer,
            local_ip=self.side_channel_host,
            handshake_port=self.handshake_port,
        )
        self.transfer_metadata = transfer_metadata
        self.xfer_handshake_metadata = transfer_metadata

        for summary, (example_layer, layer_count) in layout_summaries.items():
            (
                group_index,
                spec_index,
                cache_role,
                spec_type,
                logical_block_size,
                block_size_scales,
                transfer_unit_tokens,
                block_lens,
                block_shapes,
            ) = summary
            logger.info(
                "Mooncake transfer layout: group=%s, spec=%s, role=%s, "
                "spec_type=%s, logical_block_size=%s, scales=%s, "
                "transfer_unit_tokens=%s, block_lens=%s, block_shapes=%s, "
                "layers=%s, example_layer=%s",
                group_index,
                spec_index,
                cache_role,
                spec_type,
                logical_block_size,
                block_size_scales,
                transfer_unit_tokens,
                block_lens,
                block_shapes,
                layer_count,
                example_layer,
            )

        logger.debug(
            "Mooncake KV cache transfer metadata: metadata=%s, register_ptrs=%s, register_lengths=%s",
            transfer_metadata,
            register_regions.ptrs,
            register_regions.lengths,
        )

    def get_finished(self) -> tuple[set[str], set[str]]:
        """Return requests with completed receive and send operations."""
        raise NotImplementedError

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Return local block IDs whose KV load failed."""
        raise NotImplementedError

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        """Return and reset transfer statistics for the current interval."""
        if self.xfer_stats.is_empty():
            return None
        return self.xfer_stats.clone_and_reset()

    def start_load_kv(self, metadata: MooncakeConnectorMetadata) -> None:
        """Start D2D KV loading described by scheduler metadata."""
        raise NotImplementedError


__all__ = ["KVTransferTensorLayout", "MooncakeBaseConnectorWorker"]
