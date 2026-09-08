# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Image and video processor integration for GLM-5.3-Flash.

The implementation follows the GLM5Next processor merged into Transformers
after the transformers 5.5.4 version used by this branch. Image preprocessing
stays local to preserve the validated image path. Video pixel preprocessing is
provided by the patched Transformers package shipped in the runtime image.
"""

import math

import numpy as np
import torch
from torchvision.transforms.v2 import functional as tvF
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_backends import TorchvisionBackend
from transformers.image_transforms import group_images_by_shape, reorder_images
from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ImageInput,
    PILImageResampling,
    SizeDict,
)
from transformers.processing_utils import (
    ImagesKwargs,
    MultiModalData,
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
)
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import TensorType
from transformers.video_utils import VideoInput


class Glm5NextImageProcessorKwargs(ImagesKwargs, total=False):
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    patch_expand_factor: int
    min_image_tokens: int
    max_image_tokens: int


def smart_resize(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = 2,
    factor: int = 28,
    min_pixels: int = 16,
    max_pixels: int = 8000,
) -> tuple[int, int]:
    """Compute an aligned canvas within the GLM5Next token budget."""
    if num_frames <= 0 or height <= 0 or width <= 0:
        raise ValueError(
            "num_frames, height and width must all be positive, got "
            f"{num_frames}, {height}, {width}"
        )
    if temporal_factor <= 0 or factor <= 0:
        raise ValueError("temporal_factor and factor must be positive")

    pixels_per_token = temporal_factor * factor**2
    min_pixels *= pixels_per_token
    max_pixels *= pixels_per_token

    def align(value: int | float) -> int:
        return math.ceil(value / factor) * factor

    aligned_frames = max(
        temporal_factor,
        round(num_frames / temporal_factor) * temporal_factor,
    )

    def fit_within_budget() -> tuple[int, int]:
        minimum_pixels = aligned_frames * factor**2
        if max_pixels < minimum_pixels:
            raise ValueError(
                f"max_pixels={max_pixels} is too small. At least "
                f"{minimum_pixels} pixels are required for one aligned patch."
            )

        low, high = 1, height
        best_height, best_width = factor, factor
        while low <= high:
            content_height = (low + high) // 2
            content_width = max(1, math.floor(width * content_height / height))
            candidate_height = align(content_height)
            candidate_width = align(content_width)
            pixel_budget = aligned_frames * candidate_height * candidate_width
            if pixel_budget <= max_pixels:
                best_height, best_width = candidate_height, candidate_width
                low = content_height + 1
            else:
                high = content_height - 1
        return best_height, best_width

    aligned_height = align(height)
    aligned_width = align(width)
    aligned_pixel_budget = aligned_frames * aligned_height * aligned_width

    if aligned_pixel_budget < min_pixels:
        scale = math.sqrt(min_pixels / (num_frames * height * width))
        aligned_height = align(max(1, math.ceil(height * scale)))
        aligned_width = align(max(1, math.ceil(width * scale)))
        aligned_pixel_budget = aligned_frames * aligned_height * aligned_width

    if aligned_pixel_budget > max_pixels:
        aligned_height, aligned_width = fit_within_budget()

    return aligned_height, aligned_width


class Glm5NextImageProcessor(TorchvisionBackend):
    do_resize = True
    resample = PILImageResampling.BICUBIC
    size = {"longest_edge": 1}
    default_to_square = False
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    do_convert_rgb = True
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2
    valid_kwargs = Glm5NextImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_thw"]
    patch_expand_factor = 1
    min_image_tokens = 16
    max_image_tokens = 8000

    def preprocess(
        self,
        images: ImageInput,
        **kwargs: Unpack[Glm5NextImageProcessorKwargs],
    ) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def resize(
        self,
        images: torch.Tensor,
        resample: PILImageResampling | tvF.InterpolationMode | int | None,
        factor: int,
        temporal_factor: int,
        min_image_tokens: int,
        max_image_tokens: int,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        height, width = images.shape[-2:]
        target_height, target_width = smart_resize(
            height=height,
            width=width,
            num_frames=temporal_factor,
            factor=factor,
            temporal_factor=temporal_factor,
            min_pixels=min_image_tokens,
            max_pixels=max_image_tokens,
        )

        pixels_per_token = temporal_factor * factor**2
        scale = min(target_height / height, target_width / width)
        if temporal_factor * height * width >= (
            pixels_per_token * min_image_tokens
        ):
            scale = min(1.0, scale)
        content_height = max(1, min(target_height, math.floor(height * scale)))
        content_width = max(1, min(target_width, math.floor(width * scale)))

        if (content_height, content_width) != (height, width):
            images = super().resize(
                images,
                SizeDict(height=content_height, width=content_width),
                resample=resample,
            )

        return tvF.pad(
            images,
            [
                0,
                0,
                target_width - content_width,
                target_height - content_height,
            ],
            fill=0,
        )

    @staticmethod
    def patchify(
        images: torch.Tensor,
        patch_size: int,
        merge_size: int,
        temporal_patch_size: int,
    ) -> tuple[torch.Tensor, int, int]:
        batch_size, channel, resized_height, resized_width = images.shape
        grid_h = resized_height // patch_size
        grid_w = resized_width // patch_size
        patches = images.reshape(
            batch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)
        flatten_patches = (
            patches.unsqueeze(6)
            .expand(
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                temporal_patch_size,
                -1,
                -1,
            )
            .reshape(
                batch_size,
                grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            )
        )
        return flatten_patches, grid_h, grid_w

    def _preprocess(
        self,
        images: list[torch.Tensor],
        do_resize: bool,
        size: SizeDict,
        resample: PILImageResampling | tvF.InterpolationMode | int | None,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        patch_expand_factor: int,
        min_image_tokens: int,
        max_image_tokens: int,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        del size, kwargs
        grouped_images, grouped_images_index = group_images_by_shape(
            images,
            disable_grouping=disable_grouping,
        )
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(
                    images=stacked_images,
                    resample=resample,
                    factor=patch_size * merge_size * patch_expand_factor,
                    temporal_factor=temporal_patch_size,
                    min_image_tokens=min_image_tokens,
                    max_image_tokens=max_image_tokens,
                )
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(
            resized_images_grouped,
            grouped_images_index,
        )

        grouped_images, grouped_images_index = group_images_by_shape(
            resized_images,
            disable_grouping=disable_grouping,
        )
        processed_images_grouped = {}
        processed_grids = {}
        for shape, stacked_images in grouped_images.items():
            stacked_images = self.rescale_and_normalize(
                stacked_images,
                do_rescale,
                rescale_factor,
                do_normalize,
                image_mean,
                image_std,
            )
            patches, grid_h, grid_w = self.patchify(
                stacked_images,
                patch_size=patch_size,
                merge_size=merge_size,
                temporal_patch_size=temporal_patch_size,
            )
            processed_images_grouped[shape] = patches
            processed_grids[shape] = [[1, grid_h, grid_w]] * len(stacked_images)

        processed_images = reorder_images(
            processed_images_grouped,
            grouped_images_index,
        )
        processed_grids = reorder_images(
            processed_grids,
            grouped_images_index,
        )
        pixel_values = (
            processed_images[0]
            if len(processed_images) == 1
            else torch.cat(processed_images, dim=0)
        )
        image_grid_thw = torch.tensor(processed_grids)
        return BatchFeature(
            data={
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
            },
            tensor_type=return_tensors,
        )

    def get_number_of_image_patches(
        self,
        height: int,
        width: int,
        images_kwargs: dict | None = None,
    ) -> int:
        images_kwargs = images_kwargs or {}
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)
        min_image_tokens = images_kwargs.get(
            "min_image_tokens",
            self.min_image_tokens,
        )
        max_image_tokens = images_kwargs.get(
            "max_image_tokens",
            self.max_image_tokens,
        )
        factor = patch_size * merge_size
        resized_height, resized_width = smart_resize(
            num_frames=self.temporal_patch_size,
            height=height,
            width=width,
            factor=factor,
            min_pixels=min_image_tokens,
            max_pixels=max_image_tokens,
            temporal_factor=self.temporal_patch_size,
        )
        grid_h = resized_height // patch_size
        grid_w = resized_width // patch_size
        return grid_h * grid_w


class Glm5NextProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_token_type_ids": False,
            "return_mm_token_type_ids": True,
        },
        "videos_kwargs": {
            "return_metadata": True,
        },
    }


class Glm5NextProcessor(ProcessorMixin):
    """Transformers-5.5-compatible GLM5Next multimodal processor."""

    valid_processor_kwargs = Glm5NextProcessorKwargs

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template=None,
        **kwargs,
    ) -> None:
        del kwargs
        self.image_token = (
            "<|image|>"
            if not hasattr(tokenizer, "image_token")
            else tokenizer.image_token
        )
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None) is not None
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token = "<|video|>"
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.video_token)
        self.video_start_token_id = tokenizer.convert_tokens_to_ids(
            "<|begin_of_video|>"
        )
        self.video_end_token_id = tokenizer.convert_tokens_to_ids(
            "<|end_of_video|>"
        )
        super().__init__(
            image_processor,
            tokenizer,
            video_processor,
            chat_template=chat_template,
        )

    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput]
        | None = None,
        videos: VideoInput | None = None,
        **kwargs: Unpack[Glm5NextProcessorKwargs],
    ) -> BatchFeature:
        if images is None and videos is None and text is None:
            raise ValueError(
                "At least one of images, videos, or text must be provided"
            )

        output_kwargs = self._merge_kwargs(
            Glm5NextProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None:
            image_inputs = self.image_processor(
                images=images,
                **output_kwargs["images_kwargs"],
            )
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None
        if videos is not None:
            if self.video_processor is None:
                raise RuntimeError(
                    "GLM5Next video input requires the patched Transformers "
                    "package that provides Glm5NextVideoProcessor"
                )
            video_inputs = self.video_processor(
                videos=videos,
                **output_kwargs["videos_kwargs"],
            )
            video_grid_thw = video_inputs["video_grid_thw"]
        else:
            video_inputs = {}
            video_grid_thw = None

        if not isinstance(text, list):
            text = [text]
        processed_text = list(text)
        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            image_index = 0
            for text_index, value in enumerate(processed_text):
                if value is None:
                    raise ValueError("text must be provided when images are used")
                while self.image_token in value:
                    if image_index >= len(image_grid_thw):
                        raise ValueError(
                            "Prompt contains more image placeholders than images"
                        )
                    num_image_tokens = (
                        int(image_grid_thw[image_index].prod()) // merge_length
                    )
                    value = value.replace(
                        self.image_token,
                        "<|placeholder|>" * num_image_tokens,
                        1,
                    )
                    image_index += 1
                processed_text[text_index] = value.replace(
                    "<|placeholder|>",
                    self.image_token,
                )
            if image_index != len(image_grid_thw):
                raise ValueError(
                    "Received more images than image placeholders in the prompt"
                )
        if video_grid_thw is not None:
            video_index = 0
            for text_index, value in enumerate(processed_text):
                if value is None:
                    raise ValueError("text must be provided when videos are used")
                while self.video_token in value:
                    if video_index >= len(video_grid_thw):
                        raise ValueError(
                            "Prompt contains more video placeholders than videos"
                        )
                    value = value.replace(
                        self.video_token,
                        "<|video_placeholder|>",
                        1,
                    )
                    value = value.replace(
                        "<|video_placeholder|>",
                        self.replace_video_token(video_inputs, video_index),
                        1,
                    )
                    video_index += 1
                processed_text[text_index] = value
            if video_index != len(video_grid_thw):
                raise ValueError(
                    "Received more videos than video placeholders in the prompt"
                )

        return_tensors = output_kwargs["text_kwargs"].pop(
            "return_tensors",
            None,
        )
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop(
            "return_mm_token_type_ids",
            False,
        )
        text_inputs = self.tokenizer(
            processed_text,
            **output_kwargs["text_kwargs"],
        )
        if videos is None:
            self._check_special_mm_tokens(
                processed_text,
                text_inputs,
                modalities=["image"],
            )
        if return_mm_token_type_ids:
            text_inputs["mm_token_type_ids"] = self.create_mm_token_type_ids(
                text_inputs["input_ids"]
            )
        if not kwargs.get("return_metadata"):
            video_inputs.pop("video_metadata", None)
        return BatchFeature(
            data={**text_inputs, **image_inputs, **video_inputs},
            tensor_type=return_tensors,
        )

    def replace_image_token(
        self,
        image_inputs: dict,
        image_idx: int,
        **kwargs,
    ) -> str:
        del kwargs
        merge_length = self.image_processor.merge_size**2
        num_image_tokens = (
            int(image_inputs["image_grid_thw"][image_idx].prod()) // merge_length
        )
        return self.image_token * num_image_tokens

    def replace_video_token(
        self,
        video_inputs: dict,
        video_idx: int,
        **kwargs,
    ) -> str:
        """Build timestamped frame placeholders for one video."""
        del kwargs
        merge_length = self.video_processor.merge_size**2
        grid = video_inputs["video_grid_thw"][video_idx]
        num_frames = int(grid[0])
        num_image_tokens = int(grid.prod()) // merge_length // num_frames
        metadata = video_inputs["video_metadata"][video_idx]
        timestamps = metadata.timestamps[
            :: self.video_processor.temporal_patch_size
        ]
        selected_timestamps = list(timestamps[:num_frames])
        while len(selected_timestamps) < num_frames:
            selected_timestamps.append(
                selected_timestamps[-1] if selected_timestamps else 0
            )
        return "".join(
            self.replace_frame_token_id(
                float(timestamp),
                num_image_tokens=num_image_tokens,
            )
            for timestamp in selected_timestamps
        )

    def replace_frame_token_id(
        self,
        timestamp_sec: float,
        num_image_tokens: int = 1,
    ) -> str:
        return (
            f"<|begin_of_image|>{self.image_token * num_image_tokens}"
            f"<|end_of_image|>{timestamp_sec:.1f} seconds"
        )

    def _get_num_multimodal_tokens(
        self,
        image_sizes=None,
        video_sizes=None,
        **kwargs,
    ) -> MultiModalData:
        vision_data = {}
        if image_sizes is not None:
            images_kwargs = dict(
                Glm5NextProcessorKwargs._defaults.get("images_kwargs", {})
            )
            images_kwargs.update(kwargs)
            merge_size = images_kwargs.get(
                "merge_size",
                self.image_processor.merge_size,
            )
            num_image_patches = [
                self.image_processor.get_number_of_image_patches(
                    *image_size,
                    images_kwargs,
                )
                for image_size in image_sizes
            ]
            vision_data.update(
                num_image_tokens=[
                    num_patches // merge_size**2
                    for num_patches in num_image_patches
                ],
                num_image_patches=num_image_patches,
            )
        if video_sizes is not None:
            if self.video_processor is None:
                raise RuntimeError(
                    "GLM5Next video input requires Glm5NextVideoProcessor"
                )
            processor = self.video_processor
            factor = (
                processor.patch_size
                * processor.merge_size
                * getattr(processor, "patch_expand_factor", 1)
            )
            num_video_tokens = []
            for num_frames, height, width in video_sizes:
                num_frames = min(int(num_frames), int(processor.max_frames))
                num_frames += (-num_frames) % processor.temporal_patch_size
                resized_height, resized_width = smart_resize(
                    num_frames=num_frames,
                    height=int(height),
                    width=int(width),
                    factor=factor,
                    min_pixels=processor.min_image_tokens,
                    max_pixels=processor.max_image_tokens,
                    temporal_factor=processor.temporal_patch_size,
                )
                num_video_tokens.append(
                    num_frames
                    // processor.temporal_patch_size
                    * resized_height
                    * resized_width
                    // processor.patch_size**2
                    // processor.merge_size**2
                )
            vision_data["num_video_tokens"] = num_video_tokens
        return MultiModalData(**vision_data)

    @property
    def model_input_names(self):
        return super().model_input_names + ["mm_token_type_ids"]

    def create_mm_token_type_ids(self, input_ids: list) -> list[list[int]]:
        mm_token_type_ids = []
        for item in input_ids:
            array_ids = np.asarray(item)
            mm_token_types = np.zeros_like(array_ids)
            starts = np.cumsum(
                array_ids == self.video_start_token_id,
                axis=0,
            )
            ends = np.cumsum(
                array_ids == self.video_end_token_id,
                axis=0,
            )
            is_video_modality = starts > ends
            is_image_token = array_ids == self.image_token_id
            mm_token_types[is_image_token & is_video_modality] = 2
            mm_token_types[is_image_token & ~is_video_modality] = 1
            mm_token_type_ids.append(mm_token_types.tolist())
        return mm_token_type_ids


__all__ = ["Glm5NextImageProcessor", "Glm5NextProcessor", "smart_resize"]
