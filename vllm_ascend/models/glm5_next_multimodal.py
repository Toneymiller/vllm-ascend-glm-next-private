# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Mapping
from typing import ClassVar, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.model_executor.layers.conv import Conv2dLayer
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.glm4_1v import (
    Glm4vDummyInputsBuilder,
    Glm4vForConditionalGeneration,
    Glm4vMultiModalProcessor,
    Glm4vProcessingInfo,
)
from vllm.model_executor.models.glm_ocr import (
    GlmOcrVisionAttention,
    GlmOcrVisionPatchEmbed,
    GlmOcrVisionTransformer,
)
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.model_executor.models.vision import (
    get_vit_attn_backend,
    is_vit_use_data_parallel,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_ascend.transformers_utils.processors.glm5_next import (
    Glm5NextImageProcessor,
    Glm5NextProcessor,
    smart_resize as glm5_next_smart_resize,
)


class Glm5NextSiluAndMul(nn.Module):
    """GLM5Next SwiGLU with the checkpoint-defined activation clamp."""

    def __init__(self, swiglu_limit: float) -> None:
        super().__init__()
        self.swiglu_limit = swiglu_limit

    def forward(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return F.silu(gate) * up


class Glm5NextVisionRMSNorm(nn.Module):
    """Weight-only RMSNorm matching the Transformers GLM5Next ViT.

    Keep this vision-specific implementation independent from vLLM's RMSNorm.
    On Ascend, the latter can acquire a quantization bias from the global text
    quantization config even though the vision tower itself is unquantized.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(
            variance + self.variance_epsilon
        )
        return self.weight * hidden_states.to(input_dtype)


class AscendGlm5NextVisionMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        swiglu_limit: float,
        bias: bool = True,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        use_data_parallel = is_vit_use_data_parallel()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=in_features,
            output_sizes=[hidden_features] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
            disable_tp=use_data_parallel,
        )
        self.down_proj = RowParallelLinear(
            hidden_features,
            in_features,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
            disable_tp=use_data_parallel,
        )
        self.act_fn = Glm5NextSiluAndMul(swiglu_limit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class AscendGlm5NextVisionAttention(GlmOcrVisionAttention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        norm_eps: float,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            projection_size=projection_size,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.q_norm = Glm5NextVisionRMSNorm(self.head_dim, eps=norm_eps)
        self.k_norm = Glm5NextVisionRMSNorm(self.head_dim, eps=norm_eps)


class AscendGlm5NextVisionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        norm_eps: float,
        swiglu_limit: float,
        bias: bool = True,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.norm1 = Glm5NextVisionRMSNorm(dim, eps=norm_eps)
        self.norm2 = Glm5NextVisionRMSNorm(dim, eps=norm_eps)
        self.attn = AscendGlm5NextVisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            norm_eps=norm_eps,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.mlp = AscendGlm5NextVisionMLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            swiglu_limit=swiglu_limit,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            max_seqlen=max_seqlen,
        )
        x = x + self.mlp(self.norm2(x))
        return x


class AscendGlm5NextVisionPatchEmbed(GlmOcrVisionPatchEmbed):
    pass


class AscendGlm5NextVisionPatchMerger(nn.Module):
    def __init__(
        self,
        d_model: int,
        context_dim: int,
        swiglu_limit: float,
        quant_config=None,
        bias: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        use_data_parallel = is_vit_use_data_parallel()
        self.hidden_size = d_model
        self.proj = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=bias,
            gather_output=True,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
            disable_tp=use_data_parallel,
        )
        self.post_projection_norm = nn.LayerNorm(self.hidden_size)
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=[context_dim] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
            disable_tp=use_data_parallel,
        )
        self.down_proj = RowParallelLinear(
            context_dim,
            self.hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
            disable_tp=use_data_parallel,
        )
        self.act_fn = Glm5NextSiluAndMul(swiglu_limit)
        self.extra_activation_func = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.proj(x)
        x = self.extra_activation_func(self.post_projection_norm(x))
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class AscendGlm5NextVisionTransformer(GlmOcrVisionTransformer):
    stacked_params_mapping = (
        ("gate_up_proj.", "gate_proj.", 0),
        ("gate_up_proj.", "up_proj.", 1),
    )

    def __init__(
        self,
        text_config,
        vision_config,
        norm_eps: float = 1e-5,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        # Initialize the dedicated GLM5Next tower directly. Calling the
        # GLM-OCR initializer and replacing its blocks would briefly allocate
        # two complete vision towers for the production 24-layer config.
        nn.Module.__init__(self)
        # text_config remains in the signature for compatibility with the
        # existing multimodal model construction path.

        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads
        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.out_hidden_size = vision_config.out_hidden_size

        self.patch_embed = AscendGlm5NextVisionPatchEmbed(
            patch_size=vision_config.patch_size,
            temporal_patch_size=vision_config.temporal_patch_size,
            in_channels=vision_config.in_channels,
            hidden_size=self.hidden_size,
        )
        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = get_rope(
            head_size=head_dim,
            max_position=8192,
            is_neox_style=True,
            rope_parameters={"partial_rotary_factor": 0.5},
        )
        swiglu_limit = vision_config.swiglu_limit
        attention_bias = vision_config.attention_bias
        self.blocks = nn.ModuleList(
            [
                AscendGlm5NextVisionBlock(
                    dim=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_hidden_dim=vision_config.intermediate_size,
                    norm_eps=norm_eps,
                    swiglu_limit=swiglu_limit,
                    bias=attention_bias,
                    quant_config=quant_config,
                    prefix=f"{prefix}.blocks.{layer_idx}",
                )
                for layer_idx in range(vision_config.depth)
            ]
        )
        self.merger = AscendGlm5NextVisionPatchMerger(
            d_model=vision_config.out_hidden_size,
            context_dim=vision_config.projection_intermediate_size,
            swiglu_limit=swiglu_limit,
            quant_config=quant_config,
            bias=False,
            prefix=f"{prefix}.merger",
        )
        self.downsample = Conv2dLayer(
            in_channels=vision_config.hidden_size,
            out_channels=vision_config.out_hidden_size,
            kernel_size=vision_config.spatial_merge_size,
            stride=vision_config.spatial_merge_size,
        )
        self.post_layernorm = Glm5NextVisionRMSNorm(
            vision_config.hidden_size,
            eps=norm_eps,
        )
        self.attn_backend = get_vit_attn_backend(
            head_size=head_dim,
            dtype=torch.get_default_dtype(),
        )

    @classmethod
    def _map_weight_name(
        cls,
        name: str,
    ) -> tuple[str, int | None]:
        for param_name, weight_name, shard_id in cls.stacked_params_mapping:
            if weight_name in name:
                return name.replace(weight_name, param_name), shard_id
        return name, None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_shards: set[tuple[str, int | None]] = set()
        loaded_sources: set[str] = set()

        for source_name, loaded_weight in weights:
            if source_name in loaded_sources:
                raise ValueError(
                    f"Duplicate GLM5Next vision weight {source_name!r}"
                )
            loaded_sources.add(source_name)

            target_name, shard_id = self._map_weight_name(source_name)
            if target_name not in params_dict:
                raise ValueError(
                    "Unexpected GLM5Next vision weight "
                    f"{source_name!r} mapped to {target_name!r}"
                )

            target_shard = (target_name, shard_id)
            if target_shard in loaded_shards:
                raise ValueError(
                    "Duplicate GLM5Next vision target shard "
                    f"{target_name!r}, shard={shard_id!r}"
                )

            param = params_dict[target_name]
            weight_loader = getattr(
                param,
                "weight_loader",
                default_weight_loader,
            )
            try:
                if shard_id is None:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
            except (AssertionError, RuntimeError, ValueError) as exc:
                raise ValueError(
                    f"Failed to load GLM5Next vision weight {source_name!r} "
                    f"into {target_name!r}: checkpoint shape "
                    f"{tuple(loaded_weight.shape)}, target shape "
                    f"{tuple(param.shape)}"
                ) from exc
            loaded_shards.add(target_shard)

        # AutoWeightsLoader may call a child loader once per checkpoint shard.
        # Global completeness must therefore be checked against the checkpoint
        # index, not against the weights received by this individual call.
        return {name for name, _ in loaded_shards}


class AscendGlm5NextProcessingInfo(Glm4vProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "video": 1}

    def _is_glmga_model(self, processor: object) -> bool:
        # GLM5Next uses the same fixed-fps timestamp construction as GLMGA,
        # rather than GLM4.6V's duration-dependent sampling policy.
        return isinstance(processor, Glm5NextProcessor) or super()._is_glmga_model(
            processor
        )

    def get_hf_processor(self, **kwargs: object):
        proc = getattr(self, "_glm5_hf_processor", None)
        if proc is None:
            from huggingface_hub.constants import HF_HUB_OFFLINE
            from transformers.models.auto.image_processing_auto import (
                get_image_processor_config,
            )
            from transformers.models.auto.video_processing_auto import (
                get_video_processor_config,
            )
            try:
                from transformers.models.glm5_next.video_processing_glm5_next import (
                    Glm5NextVideoProcessor,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "GLM5Next video support requires the Transformers patch "
                    "shipped with the runtime image"
                ) from exc

            model_path = self.ctx.model_config.model
            model_config = self.ctx.model_config
            tokenizer = self.get_tokenizer()
            ip_cfg = get_image_processor_config(
                model_path,
                cache_dir=getattr(model_config, "download_dir", None),
                revision=getattr(model_config, "revision", None),
                local_files_only=bool(
                    kwargs.get("local_files_only", HF_HUB_OFFLINE)
                ),
            )
            ip_cfg = {
                key: value
                for key, value in ip_cfg.items()
                if key
                not in {
                    "auto_map",
                    "image_processor_type",
                    "processor_class",
                }
            }
            image_processor = Glm5NextImageProcessor(**ip_cfg)
            vp_cfg = get_video_processor_config(
                model_path,
                cache_dir=getattr(model_config, "download_dir", None),
                revision=getattr(model_config, "revision", None),
                local_files_only=bool(
                    kwargs.get("local_files_only", HF_HUB_OFFLINE)
                ),
            )
            vp_cfg = {
                key: value
                for key, value in vp_cfg.items()
                if key
                not in {
                    "auto_map",
                    "image_processor_type",
                    "video_processor_type",
                    "processor_class",
                }
            }
            video_processor = Glm5NextVideoProcessor(**vp_cfg)
            proc = Glm5NextProcessor(
                image_processor=image_processor,
                tokenizer=tokenizer,
                video_processor=video_processor,
            )
            self._glm5_hf_processor = proc
        return proc

    def get_image_size_with_most_features(self):
        image_processor = self.get_image_processor()
        factor = image_processor.patch_size * image_processor.merge_size
        height, width = glm5_next_smart_resize(
            num_frames=image_processor.temporal_patch_size,
            height=9999999,
            width=9999999,
            factor=factor,
            min_pixels=image_processor.min_image_tokens,
            max_pixels=image_processor.max_image_tokens,
            temporal_factor=image_processor.temporal_patch_size,
        )
        from vllm.multimodal.parse import ImageSize

        return ImageSize(width=width, height=height)

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        image_processor = self.get_image_processor()
        num_patches = image_processor.get_number_of_image_patches(
            image_height,
            image_width,
        )
        return num_patches // image_processor.merge_size**2

    def get_max_image_tokens(self) -> int:
        image_size = self.get_image_size_with_most_features()
        return self.get_num_image_tokens(
            image_width=image_size.width,
            image_height=image_size.height,
        )

    def get_num_video_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int,
    ) -> int:
        video_processor = self.get_video_processor()
        temporal_patch_size = video_processor.temporal_patch_size
        num_frames = min(num_frames, video_processor.max_frames)
        num_frames += (-num_frames) % temporal_patch_size
        factor = (
            video_processor.patch_size
            * video_processor.merge_size
            * getattr(video_processor, "patch_expand_factor", 1)
        )
        resized_height, resized_width = glm5_next_smart_resize(
            num_frames=num_frames,
            height=image_height,
            width=image_width,
            factor=factor,
            min_pixels=video_processor.min_image_tokens,
            max_pixels=video_processor.max_image_tokens,
            temporal_factor=temporal_patch_size,
        )
        return (
            num_frames
            // temporal_patch_size
            * resized_height
            * resized_width
            // video_processor.patch_size**2
            // video_processor.merge_size**2
        )

    def _get_max_timestamp_tokens(self) -> int:
        cached = getattr(self, "_glm5_max_timestamp_tokens", None)
        if cached is not None:
            return cached

        video_processor = self.get_video_processor()
        max_grid_t = (
            video_processor.max_frames
            + video_processor.temporal_patch_size
            - 1
        ) // video_processor.temporal_patch_size
        timestamps = list(range(min(max_grid_t, 300)))
        if max_grid_t > 300:
            timestamps.append(max_grid_t - 1)
        tokenizer = self.get_tokenizer()
        cached = max(
            len(
                tokenizer.encode(
                    f"{timestamp:.1f} seconds",
                    add_special_tokens=False,
                )
            )
            for timestamp in timestamps
        )
        self._glm5_max_timestamp_tokens = cached
        return cached

    def _get_video_total_tokens(self, num_frames: int) -> int:
        target_size = self.get_image_size_with_most_features()
        vision_tokens = self.get_num_video_tokens(
            image_width=target_size.width,
            image_height=target_size.height,
            num_frames=num_frames,
        )
        video_processor = self.get_video_processor()
        grid_t = (
            num_frames + video_processor.temporal_patch_size - 1
        ) // video_processor.temporal_patch_size
        return (
            vision_tokens
            + grid_t * (2 + self._get_max_timestamp_tokens())
            + 2
        )

    def _get_max_video_frames(self, max_tokens: int) -> int:
        video_processor = self.get_video_processor()
        max_frames = min(video_processor.max_frames, max(max_tokens, 0))
        num_frames = 0
        for candidate in range(1, max_frames + 1):
            if self._get_video_total_tokens(candidate) > max_tokens:
                break
            num_frames = candidate
        return num_frames

    def get_num_frames_with_most_features(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> int:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)
        image_tokens = self.get_max_image_tokens() * num_images
        tokens_per_video = max(
            (seq_len - image_tokens) // max(num_videos, 1),
            0,
        )
        return max(self._get_max_video_frames(tokens_per_video), 1)

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        result: dict[str, int] = {}
        if mm_counts.get("image", 0):
            result["image"] = min(self.get_max_image_tokens(), seq_len)
        if mm_counts.get("video", 0):
            video_processor = self.get_video_processor()
            patch_expand_factor = getattr(
                video_processor,
                "patch_expand_factor",
                1,
            )
            max_video_tokens = (
                int(video_processor.max_image_tokens)
                * patch_expand_factor**2
            )
            result["video"] = min(max_video_tokens, seq_len)
        return result


class AscendGlm5NextDummyInputsBuilder(Glm4vDummyInputsBuilder):
    pass


@MULTIMODAL_REGISTRY.register_processor(
    Glm4vMultiModalProcessor,
    info=AscendGlm5NextProcessingInfo,
    dummy_inputs=AscendGlm5NextDummyInputsBuilder,
)
class AscendGlm5NextForConditionalGeneration(
    Glm4vForConditionalGeneration, HasInnerState, IsHybrid
):
    has_inner_state: ClassVar[Literal[True]] = True
    is_hybrid: ClassVar[Literal[True]] = True

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        from vllm_ascend.models.glm5_next import AscendGlm5NextForCausalLM
        return AscendGlm5NextForCausalLM.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        from vllm_ascend.models.glm5_next import AscendGlm5NextForCausalLM
        return AscendGlm5NextForCausalLM.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls):
        from vllm_ascend.models.glm5_next import AscendGlm5NextForCausalLM
        return AscendGlm5NextForCausalLM.get_mamba_state_copy_func()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Glm4vForConditionalGeneration, self).__init__()
        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        assert multimodal_config is not None

        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.is_multimodal_pruning_enabled = (
            multimodal_config.is_multimodal_pruning_enabled()
        )

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

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        skip_prefixes = ["rot.weight"]

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=skip_prefixes,
        )

        return loader.load_weights(
            weights,
            mapper=self.hf_to_vllm_mapper,
        )
