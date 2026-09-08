# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.models import glm5_next_multimodal as glm5_mm
from vllm_ascend.models.glm5_next_multimodal import (
    AscendGlm5NextProcessingInfo,
)
from vllm_ascend.transformers_utils.processors.glm5_next import (
    Glm5NextImageProcessor,
    Glm5NextProcessor,
    smart_resize,
)


def _transformers_reference_smart_resize(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = 2,
    factor: int = 28,
    min_pixels: int = 16,
    max_pixels: int = 8000,
) -> tuple[int, int]:
    pixels_per_token = temporal_factor * factor**2
    min_pixels *= pixels_per_token
    max_pixels *= pixels_per_token

    def align(value, factor):
        return math.ceil(value / factor) * factor

    def fit_within_budget(aligned_frames):
        minimum_pixels = aligned_frames * factor**2
        if max_pixels < minimum_pixels:
            raise ValueError

        low, high = 1, height
        best_height, best_width = factor, factor
        while low <= high:
            content_height = (low + high) // 2
            content_width = max(1, math.floor(width * content_height / height))
            candidate_height = align(content_height, factor)
            candidate_width = align(content_width, factor)
            pixel_budget = aligned_frames * candidate_height * candidate_width
            if pixel_budget <= max_pixels:
                best_height, best_width = candidate_height, candidate_width
                low = content_height + 1
            else:
                high = content_height - 1
        return best_height, best_width

    aligned_frames = max(
        temporal_factor,
        round(num_frames / temporal_factor) * temporal_factor,
    )
    aligned_height = align(height, factor)
    aligned_width = align(width, factor)
    aligned_pixel_budget = aligned_frames * aligned_height * aligned_width

    if aligned_pixel_budget < min_pixels:
        scale = math.sqrt(min_pixels / (num_frames * height * width))
        aligned_height = align(max(1, math.ceil(height * scale)), factor)
        aligned_width = align(max(1, math.ceil(width * scale)), factor)
        aligned_pixel_budget = aligned_frames * aligned_height * aligned_width

    if aligned_pixel_budget > max_pixels:
        aligned_height, aligned_width = fit_within_budget(aligned_frames)

    return aligned_height, aligned_width


@pytest.mark.parametrize(
    ("height", "width"),
    [
        (224, 224),
        (224, 896),
        (896, 224),
        (32, 4096),
    ],
)
def test_glm5_next_smart_resize_matches_transformers_reference(height, width):
    expected = _transformers_reference_smart_resize(2, height, width)

    actual = smart_resize(2, height, width)

    assert actual == expected


def test_glm5_next_smart_resize_respects_alignment_and_token_budget():
    height, width = smart_resize(
        2,
        4096,
        4096,
        factor=28,
        min_pixels=16,
        max_pixels=128,
    )

    assert height % 28 == 0
    assert width % 28 == 0
    assert 2 * height * width <= 128 * 2 * 28**2


def test_glm5_next_patchify_matches_transformers_operation_order():
    image = torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(1, 3, 4, 4)

    actual, grid_h, grid_w = Glm5NextImageProcessor.patchify(
        image,
        patch_size=2,
        merge_size=2,
        temporal_patch_size=2,
    )

    expected = image.reshape(1, 3, 1, 2, 2, 1, 2, 2)
    expected = expected.permute(0, 2, 5, 3, 6, 1, 4, 7)
    expected = (
        expected.unsqueeze(6)
        .expand(-1, -1, -1, -1, -1, -1, 2, -1, -1)
        .reshape(1, 4, 24)
    )

    assert (grid_h, grid_w) == (2, 2)
    torch.testing.assert_close(actual, expected)


def test_glm5_next_patch_count_matches_resized_grid():
    processor = Glm5NextImageProcessor(
        patch_size=14,
        temporal_patch_size=2,
        merge_size=2,
        min_image_tokens=16,
        max_image_tokens=8000,
    )
    resized_height, resized_width = smart_resize(
        2,
        333,
        777,
        factor=28,
        min_pixels=16,
        max_pixels=8000,
    )

    num_patches = processor.get_number_of_image_patches(333, 777)

    assert num_patches == (resized_height // 14) * (resized_width // 14)


def test_glm5_next_image_processor_emits_pixel_values_and_grid():
    processor = Glm5NextImageProcessor(
        patch_size=2,
        temporal_patch_size=2,
        merge_size=2,
        min_image_tokens=1,
        max_image_tokens=16,
    )
    image = torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(3, 4, 4)

    outputs = processor(
        images=image,
        do_resize=False,
        do_rescale=False,
        do_normalize=False,
        return_tensors="pt",
    )

    assert outputs["pixel_values"].shape == (4, 24)
    torch.testing.assert_close(
        outputs["image_grid_thw"],
        torch.tensor([[1, 2, 2]]),
    )


def test_glm5_next_placeholder_count_matches_merged_vit_tokens():
    processor = Glm5NextProcessor.__new__(Glm5NextProcessor)
    processor.image_processor = SimpleNamespace(merge_size=2)
    processor.image_token = "<|image|>"
    image_inputs = {"image_grid_thw": torch.tensor([[1, 8, 12]])}

    replacement = processor.replace_image_token(image_inputs, 0)

    assert replacement.count("<|image|>") == 8 * 12 // 2**2


def test_glm5_next_processor_marks_image_and_video_tokens():
    processor = Glm5NextProcessor.__new__(Glm5NextProcessor)
    processor.image_token_id = 99
    processor.video_start_token_id = 100
    processor.video_end_token_id = 101

    token_types = processor.create_mm_token_type_ids(
        [[1, 99, 100, 99, 101, 99, 2]]
    )

    assert token_types == [[0, 1, 0, 2, 0, 1, 0]]


def test_glm5_next_processor_requires_multimodal_or_text_input():
    processor = Glm5NextProcessor.__new__(Glm5NextProcessor)

    with pytest.raises(
        ValueError,
        match="At least one of images, videos, or text",
    ):
        processor()


def test_glm5_next_processing_info_advertises_image_and_video():
    info = AscendGlm5NextProcessingInfo.__new__(AscendGlm5NextProcessingInfo)

    assert info.get_supported_mm_limits() == {"image": None, "video": 1}


def test_glm5_next_processor_config_uses_hf_resolution_and_offline_flag():
    info = AscendGlm5NextProcessingInfo.__new__(AscendGlm5NextProcessingInfo)
    info.ctx = SimpleNamespace(
        model_config=SimpleNamespace(
            model="org/glm5-next",
            download_dir="/cache/models",
            revision="test-revision",
        )
    )
    tokenizer = object()
    info.get_tokenizer = lambda: tokenizer

    with (
        patch(
            "transformers.models.auto.image_processing_auto."
            "get_image_processor_config",
            return_value={
                "image_processor_type": "Glm5NextImageProcessor",
                "patch_size": 14,
            },
        ) as get_config,
        patch(
            "transformers.models.auto.video_processing_auto."
            "get_video_processor_config",
            return_value={
                "video_processor_type": "Glm5NextVideoProcessor",
                "patch_size": 14,
            },
        ) as get_video_config,
        patch(
            "transformers.models.glm5_next.video_processing_glm5_next."
            "Glm5NextVideoProcessor"
        ) as video_cls,
        patch.object(glm5_mm, "Glm5NextImageProcessor") as image_cls,
        patch.object(glm5_mm, "Glm5NextProcessor") as processor_cls,
    ):
        image_processor = image_cls.return_value
        video_processor = video_cls.return_value
        result = info.get_hf_processor(local_files_only=True)

    get_config.assert_called_once_with(
        "org/glm5-next",
        cache_dir="/cache/models",
        revision="test-revision",
        local_files_only=True,
    )
    get_video_config.assert_called_once_with(
        "org/glm5-next",
        cache_dir="/cache/models",
        revision="test-revision",
        local_files_only=True,
    )
    image_cls.assert_called_once_with(patch_size=14)
    video_cls.assert_called_once_with(patch_size=14)
    processor_cls.assert_called_once_with(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
    )
    assert result is processor_cls.return_value
