# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import torch

from vllm_ascend.models.glm5_next_multimodal import (
    AscendGlm5NextProcessingInfo,
)
from vllm_ascend.transformers_utils.processors.glm5_next import (
    Glm5NextProcessor,
)


def _fake_video_processor():
    return SimpleNamespace(
        patch_size=14,
        temporal_patch_size=2,
        merge_size=2,
        patch_expand_factor=1,
        min_image_tokens=1,
        max_image_tokens=1000,
        max_frames=16,
    )


def test_glm5_next_registers_video_modality():
    info = AscendGlm5NextProcessingInfo.__new__(
        AscendGlm5NextProcessingInfo
    )

    assert info.get_supported_mm_limits() == {"image": None, "video": 1}


def test_glm5_next_video_token_budget_uses_temporal_and_spatial_merge():
    info = AscendGlm5NextProcessingInfo.__new__(
        AscendGlm5NextProcessingInfo
    )
    info.get_video_processor = lambda **kwargs: _fake_video_processor()

    num_tokens = info.get_num_video_tokens(
        image_width=56,
        image_height=56,
        num_frames=4,
    )

    assert num_tokens == 8


def test_glm5_next_video_cache_budget_is_independent_from_image_budget():
    info = AscendGlm5NextProcessingInfo.__new__(
        AscendGlm5NextProcessingInfo
    )
    video_processor = _fake_video_processor()
    video_processor.max_image_tokens = 240000
    info.get_max_image_tokens = lambda: 7930
    info.get_video_processor = lambda **kwargs: video_processor

    result = info.get_mm_max_tokens_per_item(
        seq_len=16000,
        mm_counts={"image": 1, "video": 1},
    )

    assert result == {"image": 7930, "video": 16000}


def test_glm5_next_video_cache_budget_respects_processor_limit():
    info = AscendGlm5NextProcessingInfo.__new__(
        AscendGlm5NextProcessingInfo
    )
    video_processor = _fake_video_processor()
    video_processor.max_image_tokens = 12000
    info.get_video_processor = lambda **kwargs: video_processor

    result = info.get_mm_max_tokens_per_item(
        seq_len=16000,
        mm_counts={"video": 1},
    )

    assert result == {"video": 12000}


def test_glm5_next_video_placeholder_matches_transformers_semantics():
    processor = Glm5NextProcessor.__new__(Glm5NextProcessor)
    processor.image_token = "<|image|>"
    processor.video_processor = _fake_video_processor()
    video_inputs = {
        "video_grid_thw": torch.tensor([[2, 4, 4]]),
        "video_metadata": [
            SimpleNamespace(
                timestamps=np.asarray([0.0, 0.5, 1.0, 1.5]),
            )
        ],
    }

    replacement = processor.replace_video_token(video_inputs, 0)

    frame = (
        "<|begin_of_image|>"
        + "<|image|>" * 4
        + "<|end_of_image|>"
    )
    assert replacement == f"{frame}0.0 seconds{frame}1.0 seconds"


def test_glm5_next_video_image_tokens_use_video_modality_type():
    processor = Glm5NextProcessor.__new__(Glm5NextProcessor)
    processor.image_token_id = 10
    processor.video_start_token_id = 20
    processor.video_end_token_id = 21

    token_types = processor.create_mm_token_type_ids(
        [[20, 10, 11, 10, 21, 10]]
    )

    assert token_types == [[0, 2, 0, 2, 0, 1]]
