from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import MLAAttentionSpec, UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import AscendIndexerKPoolStateSpec
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_scheduler import (
    MooncakeBaseConnectorScheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_worker import (
    MooncakeBaseConnectorWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker import (
    MooncakePullRecvingThread,
)


def _build_glm_next_worker(block_size: int, num_blocks: int = 2):
    main_name = "model.layers.3.self_attn.attn"
    index_name = "model.layers.3.self_attn.indexer.k_cache"
    state_name = "model.layers.3.self_attn.indexer.compressor.state_cache"
    main_spec = MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        compress_ratio=1,
        model_version="glm5_next",
    )
    index_spec = MLAAttentionSpec(
        block_size=block_size,
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
        compress_ratio=1,
        model_version="glm5_next",
        cache_role="indexer_state",
    )
    attention_specs = {main_name: main_spec, index_name: index_spec}
    cache_config = SimpleNamespace(
        num_blocks=num_blocks,
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=list(attention_specs),
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=block_size,
                    kv_cache_specs=attention_specs,
                ),
            ),
            SimpleNamespace(
                layer_names=[state_name],
                kv_cache_spec=state_spec,
            ),
        ],
    )
    worker = MooncakeBaseConnectorWorker.__new__(MooncakeBaseConnectorWorker)
    worker.kv_cache_config = cache_config
    worker.num_blocks = num_blocks
    worker._build_kv_cache_spec_mappings()
    return worker, main_name, index_name, state_name, index_spec, state_spec


@pytest.mark.parametrize(
    ("block_size", "scale"),
    [
        (640, 5),  # A3 P
        (1152, 9),  # A3 D / A5 P
        (2176, 17),  # A5 D
    ],
)
def test_glm_next_indexer_inner_axis_layout(block_size: int, scale: int):
    worker, main_name, index_name, _, _, _ = _build_glm_next_worker(block_size)
    main_cache = torch.empty((2 * scale, 128, 1, 512), dtype=torch.bfloat16)
    index_cache = torch.empty((2, scale * 32, 1, 128), dtype=torch.bfloat16)

    layout = worker._get_tensor_transfer_layout(
        index_name,
        index_cache,
        {main_name: main_cache, index_name: index_cache},
    )

    assert worker.layer_name_to_group_index[main_name] == 0
    assert worker.layer_name_to_group_index[index_name] == 0
    assert worker.layer_name_to_spec_index[main_name] != worker.layer_name_to_spec_index[index_name]
    assert worker._get_layer_cache_role(index_name) == "indexer_k"
    assert layout.block_size_scale == scale
    assert layout.block_stride == 8192
    assert layout.block_len == 8192
    assert layout.block_shape == (32, 1, 128)
    assert layout.transfer_unit_tokens == 128


def test_glm_next_indexer_uses_raw_token_transfer_units_across_a5_pd():
    _, _, _, _, index_spec, _ = _build_glm_next_worker(2176)
    thread = MooncakePullRecvingThread.__new__(MooncakePullRecvingThread)
    thread.dcp_size = 1
    thread.num_speculative_tokens = 0

    transfers = thread._compute_group_block_ids(
        request_id="glm-next",
        remote_tp_rank_groups=[[0]],
        remote_dcp_size=1,
        spec_index=1,
        local_block_size=2176,
        remote_block_size=1152,
        local_group_block_ids=[2],
        local_full_group_block_ids=[2],
        remote_group_block_ids=[1],
        local_num_prompt_tokens=1152,
        remote_num_prompt_tokens=1152,
        num_computed_tokens=128,
        local_block_size_scale=17,
        remote_block_size_scale=9,
        local_transfer_unit_tokens=128,
        remote_transfer_unit_tokens=128,
        cache_role="indexer_k",
        spec=index_spec,
        selection_index=0,
    )

    assert transfers == [(0, list(range(34, 42)), list(range(10, 18)))]


def test_glm_next_indexer_state_keeps_original_group_block_ids():
    worker, _, _, state_name, index_spec, state_spec = _build_glm_next_worker(2176)
    assert worker.layer_name_to_group_index[state_name] == 1
    assert worker._get_layer_cache_role(state_name) == "indexer_state"

    scheduler = MooncakeBaseConnectorScheduler.__new__(MooncakeBaseConnectorScheduler)
    scheduler.group_unique_specs = [[index_spec], [state_spec]]
    scheduler.group_block_size = [2176, 4]
    scheduler.pcp_size = 1
    scheduler.dcp_size = 1
    scheduler.num_speculative_tokens = 0

    assert scheduler._needs_prefill_token_truncation()
    assert scheduler._get_transfer_block_ids(([2], [6, 7]), prompt_len=1024) == ([2], [6, 7])

    thread = MooncakePullRecvingThread.__new__(MooncakePullRecvingThread)
    thread.dcp_size = 1
    thread.num_speculative_tokens = 0
    transfers = thread._compute_group_block_ids(
        request_id="glm-next-state",
        remote_tp_rank_groups=[[0]],
        remote_dcp_size=1,
        spec_index=2,
        local_block_size=4,
        remote_block_size=4,
        local_group_block_ids=[6, 7],
        local_full_group_block_ids=[6, 7],
        remote_group_block_ids=[4, 5],
        local_num_prompt_tokens=1024,
        remote_num_prompt_tokens=1024,
        num_computed_tokens=0,
        local_block_size_scale=1,
        remote_block_size_scale=1,
        local_transfer_unit_tokens=4,
        remote_transfer_unit_tokens=4,
        cache_role="indexer_state",
        spec=state_spec,
        selection_index=0,
    )

    assert transfers == [(0, [6, 7], [4, 5])]


def test_mtp_indexer_state_drops_only_remote_speculative_tail():
    _, _, _, _, _, state_spec = _build_glm_next_worker(2176)
    thread = MooncakePullRecvingThread.__new__(MooncakePullRecvingThread)
    thread.dcp_size = 1
    thread.num_speculative_tokens = 2

    transfers = thread._compute_group_block_ids(
        request_id="glm-next-mtp-state",
        remote_tp_rank_groups=[[0]],
        remote_dcp_size=1,
        spec_index=3,
        local_block_size=4,
        remote_block_size=4,
        local_group_block_ids=[60, 61],
        local_full_group_block_ids=[50, 60, 61],
        remote_group_block_ids=[40, 41, 42, 43],
        local_num_prompt_tokens=1024,
        remote_num_prompt_tokens=1024,
        num_computed_tokens=0,
        local_block_size_scale=1,
        remote_block_size_scale=1,
        spec=state_spec,
        selection_index=0,
        local_transfer_unit_tokens=4,
        remote_transfer_unit_tokens=4,
        cache_role="indexer_state",
    )

    assert transfers == [(0, [60, 61], [41, 42])]


def test_mtp_indexer_state_preserves_local_and_remote_group_ids():
    _, _, _, _, _, state_spec = _build_glm_next_worker(2176)
    thread = MooncakePullRecvingThread.__new__(MooncakePullRecvingThread)
    thread.spec_indices = [0]
    thread.kv_cache_specs = [state_spec]
    thread.layer_names = ["model.layers.61.self_attn.indexer.compressor.state_cache"]
    thread.layer_block_sizes = [4]
    thread.group_indices = [1]
    thread.cache_roles = ["indexer_state"]
    thread.transfer_unit_tokens = [4]
    thread.block_size_scales = [[1]]
    thread.dcp_size = 1
    thread.num_speculative_tokens = 2

    remote_metadata = SimpleNamespace(
        layer_block_sizes=[4],
        group_indices=[3],
        cache_roles=["indexer_state"],
        transfer_unit_tokens=[4],
        block_size_scales=[[1]],
    )
    request = SimpleNamespace(
        local_block_ids=([], [61]),
        local_full_block_ids=([], [61]),
        remote_block_ids=([100], [101], [102], [91, 92]),
        local_num_prompt_tokens=1024,
        remote_num_prompt_tokens=1024,
        num_computed_tokens=0,
    )

    buckets, _ = thread._build_transfer_block_buckets(
        remote_metadata=remote_metadata,
        layer_pairs=[(0, 0)],
        tp_rank_groups_by_layer={(0, 0): [[0]]},
        remote_dcp_size=1,
        requests={"request-mtp": request},
        transfer_block_ids_by_spec={},
    )

    assert buckets[0][0][(0, 0)] == [("request-mtp", [61], [91])]


def test_target_and_mtp_indexer_states_keep_group_scoped_specs():
    target_name = "model.layers.3.self_attn.indexer.compressor.state_cache"
    mtp_name = "model.layers.61.self_attn.indexer.compressor.state_cache"
    state_spec = AscendIndexerKPoolStateSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=256,
        dtype=torch.bfloat16,
        sliding_window=4,
        compress_ratio=1,
        model_version="glm5_next",
        cache_role="indexer_state",
    )
    worker = MooncakeBaseConnectorWorker.__new__(MooncakeBaseConnectorWorker)
    worker.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(layer_names=[target_name], kv_cache_spec=state_spec),
            SimpleNamespace(layer_names=[mtp_name], kv_cache_spec=state_spec),
        ]
    )

    worker._build_kv_cache_spec_mappings()

    assert worker.layer_name_to_group_index[target_name] == 0
    assert worker.layer_name_to_group_index[mtp_name] == 1
    assert worker.layer_name_to_spec_index[target_name] != worker.layer_name_to_spec_index[mtp_name]
