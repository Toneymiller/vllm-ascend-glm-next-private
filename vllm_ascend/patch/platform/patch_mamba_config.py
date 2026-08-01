# mypy: ignore-errors
import math

import vllm.model_executor.models.config
from vllm.logger import logger
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.config import MambaModelConfig
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, get_dtype_size

GLM5_KERNEL_BLOCK_SIZE = 128


def _is_glm5_next_model(model_config) -> bool:
    model_types = {
        getattr(getattr(model_config, "hf_config", None), "model_type", None),
        getattr(
            getattr(model_config, "hf_text_config", None),
            "model_type",
            None,
        ),
    }
    return bool(model_types & {"glm5_next", "glm5_next_text"})


def _using_kv_store(vllm_config) -> bool:
    """
    Check whether AscendStoreConnector is used.
    In the scenario where only PD separation is used, mamba_cache_mode is not automatically set to align.
    """
    if not vllm_config.kv_transfer_config:
        return False
    if vllm_config.kv_transfer_config.kv_connector == "AscendStoreConnector":
        return True
    if vllm_config.kv_transfer_config.kv_connector == "MultiConnector":
        kv_connector_extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        if not kv_connector_extra_config:
            return False
        if connectors := kv_connector_extra_config.get("connectors"):
            return any(connector.get("kv_connector") == "AscendStoreConnector" for connector in connectors)
    return False


def _get_mamba_target_page_size(
    *,
    is_glm5_next: bool,
    attn_page_size: int,
    mamba_raw_size: int,
    conv_block_page_size: int,
) -> int:
    if is_glm5_next:
        # GLM5-Next packs main MLA and Mamba/KDA layers into the same large
        # physical tensor slots. One page must therefore cover both the full
        # attention payload and the complete SSM + conv state.
        return max(attn_page_size, mamba_raw_size)
    return attn_page_size + conv_block_page_size


@classmethod
def verify_and_update_config(cls, vllm_config) -> None:
    """
    Ensure that page size of attention layers is greater than or
    equal to the mamba layers. If not, automatically set the attention
    block size to ensure that it is. If the attention page size is
    strictly greater than the mamba page size, we pad the mamba page size
    to make them equal.

    Args:
        vllm_config: vLLM Config
    """
    using_kv_store_with_hybrid = not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager and _using_kv_store(
        vllm_config
    )
    logger.debug("Using kv store: %s", using_kv_store_with_hybrid)
    # Enable FULL_AND_PIECEWISE by default
    MambaModelConfig.verify_and_update_config(vllm_config)

    cache_config = vllm_config.cache_config
    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config
    is_glm5_next = _is_glm5_next_model(model_config)

    if cache_config.cache_dtype == "auto":
        kv_cache_dtype = model_config.dtype
    else:
        kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

    kernel_block_size = GLM5_KERNEL_BLOCK_SIZE
    model_cls, _ = ModelRegistry.resolve_model_cls(
        model_config.architecture,
        model_config=model_config,
    )

    # get mamba block size
    mamba_shapes = model_cls.get_mamba_state_shape_from_config(vllm_config)
    mamba_dtypes = model_cls.get_mamba_state_dtype_from_config(vllm_config)
    mamba_sizes = []
    for shape, dtype in zip(mamba_shapes, mamba_dtypes):
        mamba_sizes.append(math.prod(shape) * get_dtype_size(dtype))
    ssm_block_page_size, conv_block_page_size = max(mamba_sizes), min(mamba_sizes)
    mamba_raw_size = sum(mamba_sizes)

    # Pure linear attention models (e.g. bailing 2.5) have only SSM state,
    # no conv block. Detected by a single 3-D mamba shape (ssm only, no conv).
    # Example shape: MambaSpec(shapes=((8, 128, 128),), mamba_type='linear_attention')
    if len(mamba_shapes) == 1 and len(mamba_shapes[0]) == 3:
        conv_block_page_size = 0

    # NOTE(zxr): because of the limit of Ascend Hardware, we need to keep
    # all cache tensors contiguous, so we align the page size of ssm_block
    # and single attn_block
    if model_config.use_mla:
        attn_num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        kv_lora_rank = model_config.hf_text_config.kv_lora_rank
        qk_rope_head_dim = model_config.hf_text_config.qk_rope_head_dim
        attn_single_token_k_page_size = kv_lora_rank * attn_num_kv_heads * get_dtype_size(kv_cache_dtype)
        attn_rope_token_page_size = qk_rope_head_dim * attn_num_kv_heads * get_dtype_size(kv_cache_dtype)
        attn_token_page_size = attn_single_token_k_page_size + attn_rope_token_page_size
    else:
        attn_num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        attn_head_size = model_config.get_head_size()
        attn_single_token_k_page_size = attn_head_size * attn_num_kv_heads * get_dtype_size(kv_cache_dtype)
        attn_token_page_size = 2 * attn_head_size * attn_num_kv_heads * get_dtype_size(kv_cache_dtype)

    if is_glm5_next:
        # Main MLA is the history cache and therefore the hot read path. Make
        # its payload define the large physical page, then pad the one-state
        # KDA cache to that page. Keeping the logical block a multiple of the
        # C128 SFA kernel block lets the worker expose a contiguous sequence of
        # kernel pages instead of an as_strided MLA view with a hole per block.
        min_attn_block_size = kernel_block_size * cdiv(
            mamba_raw_size,
            kernel_block_size * attn_token_page_size,
        )
        requested_block_size = cache_config.block_size or kernel_block_size
        attn_block_size = kernel_block_size * cdiv(
            max(requested_block_size, min_attn_block_size),
            kernel_block_size,
        )
    else:
        attn_block_size = kernel_block_size * cdiv(
            ssm_block_page_size,
            kernel_block_size * attn_single_token_k_page_size,
        )
        if attn_single_token_k_page_size * attn_block_size != ssm_block_page_size:
            raise AssertionError(
                "Cannot align ssm_page_size and attn_page_size."
            )

    # override attention block size if either (a) the
    # user has not set it or (b) the user has set it
    # too small.
    if is_glm5_next and cache_config.block_size != attn_block_size:
        logger.info(
            "Setting GLM-5 logical attention block size to %d tokens so "
            "the contiguous MLA page covers the complete KDA state.",
            attn_block_size,
        )
        cache_config.block_size = attn_block_size
    elif not is_glm5_next and (
        cache_config.block_size is None
        or cache_config.block_size < attn_block_size
    ):
        cache_config.block_size = attn_block_size
        logger.info(
            "Setting attention block size to %d tokens to ensure that attention page size is >= mamba page size.",
            attn_block_size,
        )

    # compute new attention page size
    attn_page_size = cache_config.block_size * attn_token_page_size

    # GLM5-Next shares each large physical tensor slot between one MLA layer
    # and up to one KDA layer from each KDA group. Their page strides must
    # therefore match, while the smaller indexer/state tensors use a separate
    # page-size class. Other hybrid models retain the established extra conv
    # padding behavior.
    target_page_size = _get_mamba_target_page_size(
        is_glm5_next=is_glm5_next,
        attn_page_size=attn_page_size,
        mamba_raw_size=mamba_raw_size,
        conv_block_page_size=conv_block_page_size,
    )

    if target_page_size < mamba_raw_size:
        raise ValueError(
            "The padded hybrid cache page is smaller than the Mamba/KDA "
            f"state: target={target_page_size}, required={mamba_raw_size}."
        )

    if (
        cache_config.mamba_page_size_padded is None
        or cache_config.mamba_page_size_padded != target_page_size
    ):
        cache_config.mamba_page_size_padded = target_page_size
        if target_page_size > mamba_raw_size:
            mamba_padding_pct = 100 * (target_page_size - mamba_raw_size) / target_page_size
            logger.info(
                "Padding mamba page size by %.2f%% to ensure "
                "that mamba page size and attention page size are "
                "exactly equal.",
                mamba_padding_pct,
            )
    # The extract_hidden_states connector (ExampleHiddenStatesConnector) only
    # manages the dedicated hidden-state cache-only layer; it does not migrate
    # mamba KV blocks across instances, so it does not require the block-aligned
    # mamba cache mode. Forcing "align" for it would route hybrid models onto
    # vLLM's fused GPU postprocess Triton kernel (introduced in vLLM #40172),
    # which the Ascend Triton backend cannot compile. Leave the mode as vLLM
    # derived it (e.g. "none" when prefix caching is off) for this case.
    spec_config = vllm_config.speculative_config
    is_extract_hidden_states = (
        spec_config is not None and getattr(spec_config, "method", None) == "extract_hidden_states"
    )
    if using_kv_store_with_hybrid and not is_extract_hidden_states:
        if cache_config.mamba_cache_mode == "none":
            cache_config.mamba_cache_mode = "align"
        else:
            assert cache_config.mamba_cache_mode == "align", (
                "mamba_cache_mode only support 'align' when kv_transfer enabled now!"
            )
    if cache_config.enable_prefix_caching and cache_config.mamba_cache_mode == "align":
        cache_config.mamba_block_size = cache_config.block_size
    else:
        cache_config.mamba_block_size = model_config.max_model_len


vllm.model_executor.models.config.HybridAttentionMambaModelConfig.verify_and_update_config = verify_and_update_config
