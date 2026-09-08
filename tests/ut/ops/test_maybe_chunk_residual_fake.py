"""Regression test for the ``maybe_chunk_residual`` fake implementation.

The real op chunks ``residual`` along dim 0 to match ``x`` and keeps every
trailing dim. The fake must mirror that (e.g. MHC residual streams shaped
[N, n, d]); returning ``empty_like(x)`` collapsed the stream dims and broke
Dynamo shape inference for the GLM-5-Next MHC path.
"""

import torch

from vllm_ascend.ops.register_custom_ops import _maybe_chunk_residual_fake


def test_fake_preserves_residual_trailing_dims():
    attn_output = torch.zeros((4, 4096))
    residual_mhc = torch.zeros((32, 4, 4096))

    fake = _maybe_chunk_residual_fake(attn_output, residual_mhc)

    assert fake.shape == (4, 4, 4096)


def test_fake_matches_x_when_trailing_dims_agree():
    x = torch.zeros((4, 4096))
    residual = torch.zeros((32, 4096))

    fake = _maybe_chunk_residual_fake(x, residual)

    assert fake.shape == (4, 4096)


def test_fake_keeps_post_comb_stream_shapes():
    attn_output = torch.zeros((4, 4096))
    post = torch.zeros((32, 4))
    comb = torch.zeros((32, 4, 4))

    assert _maybe_chunk_residual_fake(attn_output, post).shape == (4, 4)
    assert _maybe_chunk_residual_fake(attn_output, comb).shape == (4, 4, 4)