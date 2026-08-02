# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project.

"""GLM5 Next Multimodal wrapper for Ascend.

This module provides the multimodal wrapper class for GLM5 Next model.
It combines the Vision Tower with the Language Model.

Key components:
- AscendGlm5NextForConditionalGeneration: Main multimodal wrapper
- Uses AscendGlm5NextVisionTransformer for vision encoding
- Uses existing AscendGlm5NextForCausalLM for language modeling
"""

from typing import Any, Mapping

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import (
    SupportsMultiModal,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.inputs import MultiModalDataDict

from vllm_ascend.models.glm5_next_vision import AscendGlm5NextVisionTransformer


ASCEND_GLM5_NEXT_WEIGHTS_MAPPER = WeightsMapper(
    orig_to_new_prefix={
        "lm_head.": "language_model.lm_head.",
        "model.language_model.": "language_model.model.",
        "model.visual.": "visual.",
    }
)


class AscendGlm5NextForConditionalGeneration(nn.Module, SupportsMultiModal):
    """GLM5 Next Multimodal model for Ascend.

    This class combines the Vision Tower (Glm5NextVisionTransformer) with
    the Language Model (AscendGlm5NextForCausalLM) to support multimodal
    inference (image + text).
    """

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_up_proj"],
    }

    hf_to_vllm_mapper = ASCEND_GLM5_NEXT_WEIGHTS_MAPPER

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = AscendGlm5NextVisionTransformer(
                config.text_config,
                config.vision_config,
                norm_eps=config.vision_config.rms_norm_eps,
                quant_config=None,
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["Glm5NextForCausalLM"],
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None:
            return None

        return pixel_values, image_grid_thw

    def _parse_and_validate_video_input(
        self, **kwargs: object
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)

        if pixel_values_videos is None:
            return None

        return pixel_values_videos, video_grid_thw

    def _process_image_input(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        image_embeds = self.visual(pixel_values, grid_thw=grid_thw)
        merge_size = self.visual.spatial_merge_size
        sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
        return image_embeds.split(sizes)

    def _process_video_input(
        self, pixel_values_videos: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw)
        merge_size = self.visual.spatial_merge_size
        sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
        return video_embeds.split(sizes)

    def get_input_modality(
        self,
        mm_kwargs: dict[str, Any],
    ) -> str:
        if "image_grid_thw" in mm_kwargs:
            return "image"
        elif "video_grid_thw" in mm_kwargs:
            return "video"
        raise AssertionError("This line should be unreachable.")

    def embed_multimodal(
        self,
        mm_kwargs: dict[str, Any],
    ) -> torch.Tensor | None:
        """Embed multimodal inputs and combine with language model hidden states."""
        image_input = self._parse_and_validate_image_input(**mm_kwargs)
        video_input = self._parse_and_validate_video_input(**mm_kwargs)

        if image_input is None and video_input is None:
            return None

        if image_input is not None:
            pixel_values, grid_thw = image_input
            return self._process_image_input(pixel_values, grid_thw)

        if video_input is not None:
            pixel_values_videos, grid_thw = video_input
            return self._process_video_input(pixel_values_videos, grid_thw)

        return None

    def load_weights(self, weights: list[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
