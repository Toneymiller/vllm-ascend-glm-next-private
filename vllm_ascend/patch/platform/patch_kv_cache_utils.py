# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
import math
from collections import defaultdict
from dataclasses import dataclass

import vllm.v1.core.kv_cache_utils
from vllm.config import VllmConfig
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils.math_utils import cdiv, round_up
from vllm.v1.core.kv_cache_utils import (
    _approximate_gcd,
    may_override_num_blocks,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

_orig_resolve_kv_cache_block_sizes = vllm.v1.core.kv_cache_utils.resolve_kv_cache_block_sizes
_orig_get_kv_cache_groups = vllm.v1.core.kv_cache_utils.get_kv_cache_groups
_orig_max_memory_usage_bytes_from_groups = getattr(
    vllm.v1.core.kv_cache_utils,
    "_max_memory_usage_bytes_from_groups",
    None,
)


@dataclass(frozen=True)
class _Glm5CacheLayout:
    full_group: KVCacheGroupSpec
    state_group: KVCacheGroupSpec
    mamba_groups: tuple[KVCacheGroupSpec, ...]
    mla_names: tuple[str, ...]
    indexer_names: tuple[str, ...]
    state_names: tuple[str, ...]
    main_page_size: int
    small_page_size: int
    main_slot_count: int
    small_slot_count: int


def _is_glm5_spec(spec: KVCacheSpec) -> bool:
    return getattr(spec, "model_version", None) == "glm5_next"


def _unpadded_page_size(spec: KVCacheSpec) -> int:
    if hasattr(spec, "unpadded_page_size_bytes"):
        return spec.unpadded_page_size_bytes
    if hasattr(spec, "real_page_size_bytes"):
        return spec.real_page_size_bytes
    return spec.page_size_bytes


def _sorted_layer_names(layer_names: list[str]) -> tuple[str, ...]:
    try:
        return tuple(sorted(layer_names, key=extract_layer_index))
    except ValueError:
        # Synthetic/unit-test names need not contain a numeric layer index.
        return tuple(layer_names)


def _layer_indices(layer_names: tuple[str, ...]) -> tuple[int, ...] | None:
    try:
        return tuple(extract_layer_index(name) for name in layer_names)
    except ValueError:
        return None


def _is_glm5_main_spec(spec: KVCacheSpec) -> bool:
    return (
        isinstance(spec, MLAAttentionSpec)
        and _is_glm5_spec(spec)
        and spec.compress_ratio == 1
    )


def _is_glm5_indexer_spec(spec: KVCacheSpec) -> bool:
    return (
        isinstance(spec, MLAAttentionSpec)
        and _is_glm5_spec(spec)
        and spec.compress_ratio > 1
    )


def _is_glm5_state_spec(spec: KVCacheSpec) -> bool:
    return (
        isinstance(spec, SlidingWindowMLASpec)
        and _is_glm5_spec(spec)
        and getattr(spec, "cache_role", None) == "indexer_state"
    )


def _align_glm5_cache_specs(kv_cache_spec: dict[str, KVCacheSpec]) -> None:
    """Align GLM-5 specs into two physical page-size classes in-place."""

    main_specs = [spec for spec in kv_cache_spec.values() if _is_glm5_main_spec(spec)]
    indexer_specs = [
        spec for spec in kv_cache_spec.values() if _is_glm5_indexer_spec(spec)
    ]
    state_specs = [spec for spec in kv_cache_spec.values() if _is_glm5_state_spec(spec)]
    mamba_specs = [spec for spec in kv_cache_spec.values() if isinstance(spec, MambaSpec)]

    if not main_specs and not indexer_specs and not state_specs:
        return
    if not main_specs or not indexer_specs or not state_specs:
        raise ValueError("GLM-5 cache layout requires main MLA, compressed indexer, and compressor-state specs.")

    main_page_size = max(spec.page_size_bytes for spec in (*main_specs, *mamba_specs))
    main_page_size = max(
        main_page_size,
        *(_unpadded_page_size(spec) for spec in (*main_specs, *mamba_specs)),
    )
    small_page_size = max(spec.page_size_bytes for spec in (*indexer_specs, *state_specs))
    small_page_size = max(
        small_page_size,
        *(_unpadded_page_size(spec) for spec in (*indexer_specs, *state_specs)),
    )

    for spec in (*main_specs, *mamba_specs):
        object.__setattr__(spec, "page_size_padded", main_page_size)
    for spec in (*indexer_specs, *state_specs):
        object.__setattr__(spec, "page_size_padded", small_page_size)


def _create_glm5_attention_groups(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """Create one full-history group and one indexer-state group for GLM-5."""

    main_names = _sorted_layer_names(
        [name for name, spec in kv_cache_spec.items() if _is_glm5_main_spec(spec)]
    )
    indexer_names = _sorted_layer_names(
        [name for name, spec in kv_cache_spec.items() if _is_glm5_indexer_spec(spec)]
    )
    state_names = _sorted_layer_names(
        [name for name, spec in kv_cache_spec.items() if _is_glm5_state_spec(spec)]
    )

    classified_names = {*main_names, *indexer_names, *state_names}
    if classified_names != set(kv_cache_spec):
        raise ValueError("GLM-5 KV cache specs contain an unsupported cache role.")
    if not (len(main_names) == len(indexer_names) == len(state_names) > 0):
        raise ValueError(
            "Every GLM-5 MLA layer must own one main MLA, compressed indexer, "
            "and compressor-state cache."
        )

    main_indices = _layer_indices(main_names)
    if main_indices is not None and (
        main_indices != _layer_indices(indexer_names)
        or main_indices != _layer_indices(state_names)
    ):
        raise ValueError("GLM-5 MLA, indexer, and state cache layer indices do not match.")

    full_block_sizes = {
        kv_cache_spec[name].block_size for name in (*main_names, *indexer_names)
    }
    if len(full_block_sizes) != 1:
        raise ValueError(
            "GLM-5 main MLA and compressed indexer caches must use one logical block size."
        )

    for main_name, indexer_name, state_name in zip(
        main_names, indexer_names, state_names
    ):
        main_spec = kv_cache_spec[main_name]
        indexer_spec = kv_cache_spec[indexer_name]
        state_spec = kv_cache_spec[state_name]
        assert isinstance(main_spec, MLAAttentionSpec)
        assert isinstance(indexer_spec, MLAAttentionSpec)
        assert isinstance(state_spec, SlidingWindowMLASpec)

        if main_spec.block_size % indexer_spec.compress_ratio:
            raise ValueError(
                "GLM-5 logical block size must be divisible by the indexer "
                f"compression ratio: block_size={main_spec.block_size}, "
                f"compress_ratio={indexer_spec.compress_ratio}."
            )
        if (
            state_spec.block_size != indexer_spec.compress_ratio
            or state_spec.sliding_window != indexer_spec.compress_ratio
        ):
            raise ValueError(
                "GLM-5 indexer state block/window size must equal the paired "
                f"compression ratio {indexer_spec.compress_ratio}."
            )

    # Interleave each main/indexer pair while keeping a main spec first. The
    # scheduler unwraps the first nested spec as the representative, so this
    # makes it a 128-token FullAttentionManager group. Both physical caches
    # nevertheless receive the same block IDs from that manager.
    full_names = [
        name
        for main_name, indexer_name in zip(main_names, indexer_names)
        for name in (main_name, indexer_name)
    ]
    full_specs = {name: kv_cache_spec[name] for name in full_names}
    full_uniform_spec = UniformTypeKVCacheSpecs.from_specs(full_specs)
    if full_uniform_spec is None:
        raise ValueError(
            "GLM-5 main MLA and compressed indexer caches must have uniform "
            "full-attention block-table semantics."
        )

    state_specs = {name: kv_cache_spec[name] for name in state_names}
    state_uniform_spec = UniformTypeKVCacheSpecs.from_specs(state_specs)
    if state_uniform_spec is None:
        raise ValueError(
            "GLM-5 compressor-state caches must have uniform sliding-window "
            "block-table semantics."
        )

    return [
        KVCacheGroupSpec(full_names, full_uniform_spec),
        KVCacheGroupSpec(list(state_names), state_uniform_spec),
    ]


def _get_glm5_cache_layout(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> _Glm5CacheLayout | None:
    if not kv_cache_groups or not all(
        isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs) for group in kv_cache_groups
    ):
        return None

    full_groups: list[KVCacheGroupSpec] = []
    state_groups: list[KVCacheGroupSpec] = []
    mamba_groups: list[KVCacheGroupSpec] = []
    for group in kv_cache_groups:
        specs = group.kv_cache_spec.kv_cache_specs
        values = list(specs.values())
        if all(isinstance(spec, MambaSpec) for spec in values):
            mamba_groups.append(group)
        elif (
            all(
                isinstance(spec, MLAAttentionSpec) and _is_glm5_spec(spec)
                for spec in values
            )
            and any(_is_glm5_main_spec(spec) for spec in values)
            and any(_is_glm5_indexer_spec(spec) for spec in values)
        ):
            full_groups.append(group)
        elif all(_is_glm5_state_spec(spec) for spec in values):
            state_groups.append(group)

    has_glm5_group = bool(full_groups or state_groups)
    if not has_glm5_group:
        return None
    if len(full_groups) != 1 or len(state_groups) != 1:
        raise ValueError(
            "GLM-5 requires exactly one combined main/indexer group and one "
            "compressor-state KV cache group."
        )
    classified = len(full_groups) + len(state_groups) + len(mamba_groups)
    if classified != len(kv_cache_groups):
        raise ValueError("GLM-5 KV cache groups contain an unsupported cache spec.")

    full_group = full_groups[0]
    state_group = state_groups[0]
    full_specs = full_group.kv_cache_spec.kv_cache_specs
    mla_names = _sorted_layer_names(
        [name for name in full_group.layer_names if _is_glm5_main_spec(full_specs[name])]
    )
    indexer_names = _sorted_layer_names(
        [
            name
            for name in full_group.layer_names
            if _is_glm5_indexer_spec(full_specs[name])
        ]
    )
    state_names = _sorted_layer_names(state_group.layer_names)
    if not (len(mla_names) == len(indexer_names) == len(state_names)):
        raise ValueError("Every GLM-5 MLA layer must own one compressed indexer and one compressor-state cache.")
    mla_indices = _layer_indices(mla_names)
    if mla_indices is not None and (
        mla_indices != _layer_indices(indexer_names) or mla_indices != _layer_indices(state_names)
    ):
        raise ValueError("GLM-5 MLA, indexer, and state cache layer indices do not match.")

    main_specs = {name: full_specs[name] for name in mla_names}
    indexer_specs = {name: full_specs[name] for name in indexer_names}
    state_specs = state_group.kv_cache_spec.kv_cache_specs
    main_page_sizes = {spec.page_size_bytes for spec in main_specs.values()} | {
        spec.page_size_bytes for group in mamba_groups for spec in group.kv_cache_spec.kv_cache_specs.values()
    }
    small_page_sizes = {spec.page_size_bytes for spec in (*indexer_specs.values(), *state_specs.values())}
    if len(main_page_sizes) != 1 or len(small_page_sizes) != 1:
        raise ValueError("GLM-5 cache specs were not aligned to two physical page sizes.")

    sorted_mamba_groups = tuple(mamba_groups)
    main_slot_count = max(
        [
            len(mla_names),
            *(len(group.layer_names) for group in sorted_mamba_groups),
        ]
    )
    return _Glm5CacheLayout(
        full_group=full_group,
        state_group=state_group,
        mamba_groups=sorted_mamba_groups,
        mla_names=mla_names,
        indexer_names=indexer_names,
        state_names=state_names,
        main_page_size=main_page_sizes.pop(),
        small_page_size=small_page_sizes.pop(),
        main_slot_count=main_slot_count,
        small_slot_count=len(indexer_names),
    )


def _ascend_resolve_kv_cache_block_sizes(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
) -> tuple[int, int]:
    """Ascend-compatible resolve_kv_cache_block_sizes.

    vLLM PR #40860 added a restriction that hybrid KV cache groups with
    multiple block sizes do not support context parallelism (dcp/pcp > 1).
    This restriction is correct for CUDA but not for Ascend, which implements
    context parallelism for MLA and SWA-MLA layers independently.

    For multiple KV cache groups with CP, compute scheduler_block_size as
    lcm(group_block_sizes) * dcp * pcp to maintain alignment, consistent
    with the pre-PR-#40860 behavior of block_size * dcp * pcp.
    """
    cache_config = vllm_config.cache_config
    dcp = vllm_config.parallel_config.decode_context_parallel_size
    pcp = vllm_config.parallel_config.prefill_context_parallel_size
    groups = kv_cache_config.kv_cache_groups

    if len(groups) <= 1:
        bs = cache_config.block_size * dcp * pcp
        return bs, bs

    if dcp != 1 or pcp != 1:
        # Ascend supports CP with multiple KV cache groups; compute
        # scheduler_block_size using the LCM of all group block sizes
        # multiplied by the CP factors for proper alignment.
        group_block_sizes = [g.kv_cache_spec.block_size for g in groups]
        scheduler_block_size = math.lcm(*group_block_sizes) * dcp * pcp
        if not cache_config.enable_prefix_caching:
            return scheduler_block_size, scheduler_block_size
        hash_block_size = math.gcd(*group_block_sizes)
        return scheduler_block_size, hash_block_size

    return _orig_resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)


def group_and_unify_kv_cache_specs(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[UniformTypeKVCacheSpecs] | None:
    """
    Group the KV cache specs and unify each group into one UniformTypeKVCacheSpecs.
    Currently, this is only used for DeepseekV4.
    """
    if not any(isinstance(spec, SlidingWindowMLASpec) for spec in kv_cache_spec.values()):
        return None

    ratio_specs: dict[int, dict[str, KVCacheSpec]] = defaultdict(dict)
    grouped_swa_mla_specs: dict[int, dict[str, KVCacheSpec]] = defaultdict(dict)
    for name, spec in kv_cache_spec.items():
        if isinstance(spec, SlidingWindowMLASpec):
            grouped_swa_mla_specs[spec.block_size][name] = spec
        elif isinstance(spec, MLAAttentionSpec):
            ratio_specs[spec.compress_ratio][name] = spec

    mla_uniform_specs = []
    for ratio in sorted(ratio_specs, key=lambda ratio: (ratio != 4, ratio)):
        spec_dict = ratio_specs[ratio]
        assert len(spec_dict) > 0
        mla_uniform_specs.append(UniformTypeKVCacheSpecs.from_specs(spec_dict))
    assert mla_uniform_specs is not None

    swa_uniform_specs: list[UniformTypeKVCacheSpecs] = []
    for spec_dict in grouped_swa_mla_specs.values():
        uniform_spec = UniformTypeKVCacheSpecs.from_specs(spec_dict)
        assert uniform_spec is not None
        swa_uniform_specs.append(uniform_spec)

    return [*mla_uniform_specs, *swa_uniform_specs]


def _create_uniform_mamba_groups(
    mamba_specs: dict[str, MambaSpec],
    grouped_layer_names: list[list[str]],
) -> list[KVCacheGroupSpec]:
    groups = []
    for layer_names in grouped_layer_names:
        layer_names = list(_sorted_layer_names(layer_names))
        layer_specs = {name: mamba_specs[name] for name in layer_names}
        uniform_spec = UniformTypeKVCacheSpecs.from_specs(layer_specs)
        if uniform_spec is None:
            raise ValueError("GLM-5 Mamba layers in one KV cache group must have uniform block-table semantics.")
        groups.append(KVCacheGroupSpec(layer_names, uniform_spec))
    return groups


def _group_glm5_mamba_layer_names(
    kv_cache_spec: dict[str, KVCacheSpec],
    mamba_specs: dict[str, MambaSpec],
) -> list[list[str]]:
    """Recover GLM-5's recurrent pattern independent of dict insertion order.

    The model runner intentionally collects attention specs before Mamba specs,
    so iterating ``kv_cache_spec`` directly makes all Mamba layers look like one
    long run. Instead, reconstruct model-layer order from the layer indices,
    find the maximum number of consecutive Mamba/KDA layers, and use the same
    strided grouping strategy as vLLM's generic hybrid-cache grouping.
    """

    layer_is_mamba: dict[int, bool] = {}
    mamba_name_by_index: dict[int, str] = {}
    for name in kv_cache_spec:
        try:
            layer_idx = extract_layer_index(name)
        except ValueError as exc:
            raise ValueError(
                "GLM-5 Mamba cache grouping requires layer names with numeric "
                f"indices, got {name!r}."
            ) from exc

        is_mamba = name in mamba_specs
        previous_kind = layer_is_mamba.setdefault(layer_idx, is_mamba)
        if previous_kind != is_mamba:
            raise ValueError(
                "GLM-5 model layer cannot contain both Mamba and MLA cache "
                f"specs: layer index {layer_idx}."
            )
        if is_mamba:
            if layer_idx in mamba_name_by_index:
                raise ValueError(
                    "GLM-5 model layer must own exactly one Mamba cache spec: "
                    f"layer index {layer_idx}."
                )
            mamba_name_by_index[layer_idx] = name

    max_run_length = 0
    run_length = 0
    previous_layer_idx: int | None = None
    for layer_idx in sorted(layer_is_mamba):
        is_consecutive = previous_layer_idx is not None and layer_idx == previous_layer_idx + 1
        if layer_is_mamba[layer_idx]:
            run_length = run_length + 1 if is_consecutive else 1
            max_run_length = max(max_run_length, run_length)
        else:
            run_length = 0
        previous_layer_idx = layer_idx

    if max_run_length == 0:
        raise ValueError("GLM-5 Mamba specs were provided but no Mamba layers were found.")

    sorted_mamba_names = [mamba_name_by_index[idx] for idx in sorted(mamba_name_by_index)]
    return [sorted_mamba_names[i::max_run_length] for i in range(max_run_length)]


def get_kv_cache_groups(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """Build GLM-5's paired full group and packed Mamba subgroups."""

    is_glm5 = any(getattr(spec, "model_version", None) == "glm5_next" for spec in kv_cache_spec.values())
    if not is_glm5:
        return _orig_get_kv_cache_groups(vllm_config, kv_cache_spec)

    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    if getattr(scheduler_config, "disable_hybrid_kv_cache_manager", False):
        raise ValueError(
            "GLM-5's paired MLA/indexer and sliding state layout requires the "
            "hybrid KV cache manager."
        )

    _align_glm5_cache_specs(kv_cache_spec)
    mamba_specs = {name: spec for name, spec in kv_cache_spec.items() if isinstance(spec, MambaSpec)}
    attention_specs = {name: spec for name, spec in kv_cache_spec.items() if not isinstance(spec, MambaSpec)}
    groups = _create_glm5_attention_groups(attention_specs)
    if not mamba_specs:
        # The standalone MTP runner has the same paired full group and state
        # group, but no KDA/Mamba subgroups.
        return groups

    mamba_grouped_names = _group_glm5_mamba_layer_names(
        kv_cache_spec,
        mamba_specs,
    )
    # Keep all groups represented as UniformTypeKVCacheSpecs. Otherwise vLLM's
    # allocator falls back to the legacy uniform-page-size path when MambaSpec
    # and UniformTypeKVCacheSpecs coexist, which cannot represent GLM-5's
    # independently sized KDA/MLA/Indexer pages.
    groups.extend(_create_uniform_mamba_groups(mamba_specs, mamba_grouped_names))
    return groups


def _get_kv_cache_groups_uniform_groups(
    grouped_specs: list[UniformTypeKVCacheSpecs],
) -> list[KVCacheGroupSpec]:
    """
    Generate the KV cache groups from the grouped specs.
    """
    assert len(grouped_specs) > 0 and all(isinstance(spec, UniformTypeKVCacheSpecs) for spec in grouped_specs)
    # For now, we restrict the first grouped_spec to be UniformTypeKVCacheSpecs
    # containing only MLAAttentionSpec.
    full_mla_spec = grouped_specs[0]
    full_mla_c128_spec = grouped_specs[1]

    assert all(isinstance(spec, MLAAttentionSpec) for spec in full_mla_spec.kv_cache_specs.values())
    full_mla_group = KVCacheGroupSpec(
        layer_names=list(full_mla_spec.kv_cache_specs.keys()),
        kv_cache_spec=full_mla_spec,
    )
    full_mla_c128_group = KVCacheGroupSpec(
        layer_names=list(full_mla_c128_spec.kv_cache_specs.keys()),
        kv_cache_spec=full_mla_c128_spec,
    )

    # We define a layer tuple as a group of layers with different page sizes, and
    # one UniformTypeKVCacheSpecs contains a list of layer tuples.
    # For example, if we have 11 C4 layers and 10 C128 layers, we can define a layer
    # tuple as [C4I, C4A, C128], and the full_mla_group will contain "11" layer tuples.
    # The other uniform KV cache specs will be similarly partitioned into layer tuples.
    # Say we have 21 SWA layers, all with the same page size, then we will have "21"
    # layer tuples.
    num_layer_tuples_per_group: list[int] = [g_spec.get_num_layer_tuples() for g_spec in grouped_specs]
    # Choose `num_layer_tuples` to minimize total padding across groups.
    num_layer_tuples = _approximate_gcd(num_layer_tuples_per_group, lower_bound=num_layer_tuples_per_group[0])
    # Round up to the nearest multiple of `num_layer_tuples` (i.e., padding)
    num_layer_tuples_per_group = [round_up(x, num_layer_tuples) for x in num_layer_tuples_per_group]

    # TODO(cmq): this is not general enough
    swa_mla_specs = grouped_specs[2:]

    assert all(
        isinstance(spec, SlidingWindowMLASpec) for group in swa_mla_specs for spec in group.kv_cache_specs.values()
    )

    # Split each SWA UniformKV group into smaller groups to align their #(layer tuples)
    # Possibly padding layer tuples for this.
    # Additionally, we also pad KV blocks in each SWA layer, to align the page size
    # with the corresponding layer in the full-MLA group.
    all_page_sizes = full_mla_spec.get_page_sizes()
    swa_mla_groups = []
    for sm_spec in swa_mla_specs:
        sm_page_sizes = sm_spec.get_page_sizes()
        layers_per_size: dict[int, list[str]] = defaultdict(list)
        is_glm5_state_group = all(
            getattr(spec, "model_version", None) == "glm5_next" for spec in sm_spec.kv_cache_specs.values()
        )
        if not is_glm5_state_group:
            assert max(sm_page_sizes) <= max(all_page_sizes)

        # Unify page size by padding layers' page_size to the nearest larger page_size.
        # Compute candidate (nearest larger page_size) for each unique page size.
        size_to_candidate: dict[int, int] = {}
        for ps in sm_page_sizes:
            size_to_candidate[ps] = ps if is_glm5_state_group else min(x for x in all_page_sizes if x >= ps)
        # Pad and collect layer names per page size.
        for layer_name, layer_spec in sm_spec.kv_cache_specs.items():
            current_size = layer_spec.page_size_bytes
            candidate = size_to_candidate[current_size]
            if current_size < candidate:
                object.__setattr__(layer_spec, "page_size_padded", candidate)
            layers_per_size[candidate].append(layer_name)
        # NOTE(yifan): for now, inside a UniformKV group, each page_size should
        # have the same number of layers. This also means we don't need to pad layers
        # inside a partial-full layer tuple.
        assert len(set(len(layers) for layers in layers_per_size.values())) == 1
        num_layers_per_size = len(next(iter(layers_per_size.values())))

        # Split layers inside each UniformKV group for aligned #(layers).
        # See `_get_kv_cache_groups_uniform_page_size` for more details.
        num_tuple_groups = cdiv(num_layers_per_size, num_layer_tuples)
        layer_tuples = list(zip(*layers_per_size.values()))
        for i in range(num_tuple_groups):
            group_layer_tuples = layer_tuples[i::num_tuple_groups]
            # Flatten tuples and build dict for from_specs
            group_layer_names = [name for layer_tuple in group_layer_tuples for name in layer_tuple]
            group_layer_specs = {name: sm_spec.kv_cache_specs[name] for name in group_layer_names}
            sub_sm_spec = UniformTypeKVCacheSpecs.from_specs(group_layer_specs)
            assert sub_sm_spec is not None
            swa_mla_groups.append(
                KVCacheGroupSpec(
                    layer_names=group_layer_names,
                    kv_cache_spec=sub_sm_spec,
                )
            )

    return [full_mla_group, full_mla_c128_group, *swa_mla_groups]


def _get_kv_cache_config_deepseek_v4(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> tuple[int, list[KVCacheTensor]]:
    """DeepseekV4 KV cache tensor layout planning.

    Precondition: kv_cache_groups[0] is the full-MLA group; its page sizes
    define the canonical bucket set. Non-full-MLA groups must have been
    page_size-padded upstream (see _get_kv_cache_groups_uniform_groups) so
    every layer's page_size matches one of the full-MLA bucket sizes.

    For each group, bucket its layers by page_size_bytes and place each
    layer at tuple_idx = position-within-bucket. Emit one KVCacheTensor
    per (tuple_idx, bucket) whose shared_by is the union of per-group
    layers at that slot.
    """
    glm5_layout = _get_glm5_cache_layout(kv_cache_groups)
    if glm5_layout is not None:
        bytes_per_block = (
            glm5_layout.main_slot_count * glm5_layout.main_page_size
            + glm5_layout.small_slot_count * glm5_layout.small_page_size
        )
        num_blocks = may_override_num_blocks(
            vllm_config,
            available_memory // bytes_per_block,
        )
        kv_cache_tensors: list[KVCacheTensor] = []
        for slot_idx in range(glm5_layout.main_slot_count):
            shared_by = []
            if slot_idx < len(glm5_layout.mla_names):
                shared_by.append(glm5_layout.mla_names[slot_idx])
            for group in glm5_layout.mamba_groups:
                if slot_idx < len(group.layer_names):
                    shared_by.append(group.layer_names[slot_idx])
            kv_cache_tensors.append(
                KVCacheTensor(
                    size=glm5_layout.main_page_size * num_blocks,
                    shared_by=shared_by,
                )
            )
        for slot_idx in range(glm5_layout.small_slot_count):
            kv_cache_tensors.append(
                KVCacheTensor(
                    size=glm5_layout.small_page_size * num_blocks,
                    shared_by=[
                        glm5_layout.indexer_names[slot_idx],
                        glm5_layout.state_names[slot_idx],
                    ],
                )
            )
        return num_blocks, kv_cache_tensors

    full_mla_spec = kv_cache_groups[0].kv_cache_spec
    assert isinstance(full_mla_spec, UniformTypeKVCacheSpecs)
    page_sizes = sorted(full_mla_spec.get_page_sizes())
    layer_tuple_page_bytes = sum(page_sizes)

    # Pre-bucket each group's layers by page_size (registration order within
    # bucket). bucketed[g_idx][page_size] = [layer_name, ...].
    mtp_layer_names = []
    mtp_page_size = 0
    bucketed: list[dict[int, list[str]]] = []
    for group in kv_cache_groups:
        assert isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        specs = group.kv_cache_spec.kv_cache_specs
        b: dict[int, list[str]] = defaultdict(list)
        for name in group.layer_names:
            if "mtp" not in name:
                b[specs[name].page_size_bytes].append(name)
            else:
                mtp_layer_names.append(name)
                mtp_page_size = specs[name].page_size_bytes
        bucketed.append(b)

    # num_layer_tuples = longest bucket list across all groups. For the
    # full-MLA group this equals the count of layers in the largest
    # per-page-size bucket (= get_num_layer_tuples()); for SWA sub-groups
    # this equals the sub-group size (each has a single page_size).
    num_layer_tuples = max(len(layers) for b in bucketed for layers in b.values()) + len(mtp_layer_names)

    num_blocks = available_memory // (layer_tuple_page_bytes * num_layer_tuples)
    num_blocks = may_override_num_blocks(vllm_config, num_blocks)

    kv_cache_tensors: list[KVCacheTensor] = []
    for tuple_idx in range(num_layer_tuples - len(mtp_layer_names)):
        for ps in page_sizes:
            shared_by: list[str] = []
            for b in bucketed:
                bucket = b.get(ps)
                if bucket is not None and tuple_idx < len(bucket):
                    shared_by.append(bucket[tuple_idx])
            kv_cache_tensors.append(KVCacheTensor(size=ps * num_blocks, shared_by=shared_by))
    for i in range(len(mtp_layer_names)):
        kv_cache_tensors.append(KVCacheTensor(size=mtp_page_size * num_blocks, shared_by=[mtp_layer_names[i]]))

    return num_blocks, kv_cache_tensors


def _max_memory_usage_bytes_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    """Account for GLM-5's shared block-id pool and two page classes."""

    glm5_layout = _get_glm5_cache_layout(kv_cache_groups)
    if glm5_layout is None:
        if _orig_max_memory_usage_bytes_from_groups is None:
            raise RuntimeError("The installed vLLM does not expose group memory accounting.")
        return _orig_max_memory_usage_bytes_from_groups(
            vllm_config,
            kv_cache_groups,
        )

    bytes_per_block = (
        glm5_layout.main_slot_count * glm5_layout.main_page_size
        + glm5_layout.small_slot_count * glm5_layout.small_page_size
    )
    blocks_needed = sum(group.kv_cache_spec.max_memory_usage_pages(vllm_config) for group in kv_cache_groups)
    return bytes_per_block * blocks_needed


vllm.v1.core.kv_cache_utils.resolve_kv_cache_block_sizes = _ascend_resolve_kv_cache_block_sizes
vllm.v1.core.kv_cache_utils.get_kv_cache_groups = get_kv_cache_groups
vllm.v1.core.kv_cache_utils.group_and_unify_kv_cache_specs = group_and_unify_kv_cache_specs
vllm.v1.core.kv_cache_utils._get_kv_cache_groups_uniform_groups = _get_kv_cache_groups_uniform_groups
# Patch the allocator entry point that the installed upstream actually calls.
# vLLM releases have used both names; the runtime is an unmodified upstream
# package, so do not infer the symbol from local reference-tree changes.
if hasattr(vllm.v1.core.kv_cache_utils, "_get_kv_cache_config_deepseek_v4"):
    vllm.v1.core.kv_cache_utils._get_kv_cache_config_deepseek_v4 = _get_kv_cache_config_deepseek_v4
if hasattr(vllm.v1.core.kv_cache_utils, "_get_kv_cache_config_packed"):
    vllm.v1.core.kv_cache_utils._get_kv_cache_config_packed = _get_kv_cache_config_deepseek_v4
if _orig_max_memory_usage_bytes_from_groups is not None:
    vllm.v1.core.kv_cache_utils._max_memory_usage_bytes_from_groups = _max_memory_usage_bytes_from_groups

# Also patch the reference used by engine/core.py which imports the function directly.
import vllm.v1.engine.core  # noqa: E402

vllm.v1.engine.core.resolve_kv_cache_block_sizes = _ascend_resolve_kv_cache_block_sizes
