# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project.

"""Utility functions for Vision Transformer support."""

import torch

from vllm.v1.attention.backends.registry import AttentionBackendEnum


def get_vit_attn_backend(
    head_size: int,
    dtype: torch.dtype,
) -> AttentionBackendEnum | None:
    """Stub for Vision Transformer attention backend selection.
    
    Returns None to use default attention backend.
    """
    return None


def is_vit_use_data_parallel(num_heads: int | None = None) -> bool:
    """Stub for Vision Transformer data parallel mode.
    
    Returns False to use tensor parallel mode.
    """
    return False
