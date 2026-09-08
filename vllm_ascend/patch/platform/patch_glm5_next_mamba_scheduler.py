from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

from vllm_ascend.patch.platform.patch_mamba_config import _is_glm5_next_model


_ORIGINAL_MAMBA_BLOCK_ALIGNED_SPLIT = Scheduler._mamba_block_aligned_split


def _mamba_block_aligned_split(
    self: Scheduler,
    request: Request,
    num_new_tokens: int,
    num_new_local_computed_tokens: int = 0,
    num_external_computed_tokens: int = 0,
) -> int:
    """Align GLM-Next prefill chunks to complete Mamba state checkpoints.

    In a hybrid cache, ``cache_config.block_size`` can be rewritten to the
    smallest physical block among cache groups. For GLM-Next that can be the
    4-token indexer-state block, while ``self.block_size`` is the resolved
    scheduler/common boundary at which a complete Mamba state is cacheable.
    Other models retain the exact upstream behavior.
    """
    if not _is_glm5_next_model(self.vllm_config.model_config):
        return _ORIGINAL_MAMBA_BLOCK_ALIGNED_SPLIT(
            self,
            request,
            num_new_tokens,
            num_new_local_computed_tokens,
            num_external_computed_tokens,
        )

    num_computed_tokens = (
        request.num_computed_tokens
        + num_new_local_computed_tokens
        + num_external_computed_tokens
    )
    if num_computed_tokens < max(
        request.num_prompt_tokens,
        request.num_tokens - 1,
    ):
        block_size = self.block_size
        last_cache_position = request.num_tokens - request.num_tokens % block_size
        if self.use_eagle:
            last_cache_position = max(last_cache_position - block_size, 0)
        num_computed_tokens_after_sched = num_computed_tokens + num_new_tokens
        if num_computed_tokens_after_sched < last_cache_position:
            num_new_tokens = num_new_tokens // block_size * block_size
        elif (
            num_computed_tokens
            < last_cache_position
            < num_computed_tokens_after_sched
        ):
            num_new_tokens = last_cache_position - num_computed_tokens

    return num_new_tokens


Scheduler._mamba_block_aligned_split = _mamba_block_aligned_split
