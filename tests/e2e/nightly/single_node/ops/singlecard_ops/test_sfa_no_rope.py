"""Standalone precision check for npu_sparse_flash_attention with rope=None.

Compares the NPU kernel (query_rope=None / key_rope=None, i.e. ropeHeadDim=0)
against the PyTorch golden path used by Indexer KPool MLA.

Usage on A2:
    python test_sfa_no_rope.py
    python test_sfa_no_rope.py --tokens 10 --actual-kv 10 --topk 2051
"""

import argparse
import sys

import torch
import torch_npu  # noqa: F401  (required to register NPU backend)

from vllm_ascend.utils import enable_custom_op

torch_npu.npu.config.allow_internal_format = True
enable_custom_op()


BF16_ATOL = 6e-2
BF16_RTOL = 7.8125e-3

KV_LORA_RANK = 512
ROPE_HEAD_DIM = 0  # this script targets the rope=None path only


def make_inputs(
    tokens: int,
    heads: int,
    actual_kv: int,
    topk: int,
    block_size: int,
    seed: int,
) -> dict:
    """Build one single-request decode-shaped input set on NPU.

    actual_seq_lengths_query is cumulative (TND layout), matching what the
    golden path expects: query_ends = actual_seq_lengths_query.
    """
    torch.manual_seed(seed)

    cache_dim = KV_LORA_RANK + ROPE_HEAD_DIM
    num_blocks = max(1, (actual_kv + block_size - 1) // block_size)
    scale = float(cache_dim) ** -0.5

    ql_nope = (
        torch.empty((tokens, heads, KV_LORA_RANK), dtype=torch.float32)
        .uniform_(-1.0, 1.0)
        .to(torch.bfloat16)
    )
    q_pe = torch.empty((tokens, heads, ROPE_HEAD_DIM), dtype=torch.bfloat16)

    packed_kv_cache = (
        torch.empty((num_blocks, block_size, 1, cache_dim), dtype=torch.float32)
        .uniform_(-1.0, 1.0)
        .to(torch.bfloat16)
    )

    # sparse_mode=3 causal limit for a single request whose query and key
    # lengths are both `tokens`/`actual_kv`: kv_len - q_len + q_idx + 1.
    # Valid indices go first, -1 padding fills the tail, mirroring the
    # indexer's real output layout.
    topk_indices = torch.full((tokens, 1, topk), -1, dtype=torch.int32)
    for token_idx in range(tokens):
        limit = min(actual_kv - tokens + token_idx + 1, actual_kv)
        valid = min(max(limit, 0), topk)
        if valid > 0:
            topk_indices[token_idx, 0, :valid] = torch.randperm(limit, dtype=torch.int32)[:valid]

    block_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
    actual_seq_lengths_query = torch.tensor([tokens], dtype=torch.int32)
    actual_seq_lengths_key = torch.tensor([actual_kv], dtype=torch.int32)

    return {
        "ql_nope": ql_nope.npu(),
        "q_pe": q_pe.npu(),
        "packed_kv_cache": packed_kv_cache.npu(),
        "topk_indices": topk_indices.npu(),
        "block_table": block_table.npu(),
        "actual_seq_lengths_query": actual_seq_lengths_query.npu(),
        "actual_seq_lengths_key": actual_seq_lengths_key.npu(),
        "scale": scale,
        "num_actual_tokens": tokens,
    }


def tensor_stats(name: str, t: torch.Tensor) -> str:
    """One-line stats, safe on empty tensors."""
    head = f"{name}: shape={tuple(t.shape)}, dtype={t.dtype}"
    if t.numel() == 0:
        return f"{head}, <empty>"
    f = t.detach().float().cpu()
    nonzero_ratio = float((f != 0).sum()) / f.numel()
    return (
        f"{head}, min={f.min():.6f}, max={f.max():.6f}, "
        f"mean={f.mean():.6f}, absmax={f.abs().max():.6f}, "
        f"nonzero={nonzero_ratio:.4f}, "
        f"nan={bool(f.isnan().any())}, inf={bool(f.isinf().any())}"
    )


def run_golden(inputs: dict) -> torch.Tensor:
    """Inline PyTorch reference for sparse attention with rope=0.

    Equivalent to AscendIndexerKPoolMLAImpl._sparse_attention_pytorch
    when q_pe.shape[-1] == 0.
    """
    ql_nope = inputs["ql_nope"]
    packed_kv_cache = inputs["packed_kv_cache"]
    topk_indices = inputs["topk_indices"]
    block_table = inputs["block_table"]
    actual_seq_lengths_query = inputs["actual_seq_lengths_query"]
    actual_seq_lengths_key = inputs["actual_seq_lengths_key"]
    scale = inputs["scale"]
    num_actual_tokens = inputs["num_actual_tokens"]

    tokens, heads, head_dim = ql_nope.shape
    block_size = packed_kv_cache.shape[1]

    # topk_indices: (T, 1, K) -> squeeze to (T, K)
    if topk_indices.ndim == 3:
        topk_indices = topk_indices.squeeze(1)

    output = torch.zeros_like(ql_nope)
    if num_actual_tokens == 0:
        return output

    query_ends = actual_seq_lengths_query
    query_starts = torch.cat([torch.zeros_like(query_ends[:1]), query_ends[:-1]])
    token_ids = torch.arange(num_actual_tokens, device=ql_nope.device, dtype=query_ends.dtype)
    request_ids = torch.bucketize(token_ids, query_ends, right=True)
    num_block_columns = block_table.shape[1]
    score_mask_value = torch.finfo(torch.float32).min

    for t in range(num_actual_tokens):
        req = int(request_ids[t])
        q_offset = int(token_ids[t] - query_starts[req])
        q_len = int(query_ends[req] - query_starts[req])
        kv_len = int(actual_seq_lengths_key[req])
        causal_limit = kv_len - q_len + q_offset + 1

        sparse_idx = topk_indices[t].to(torch.int64)  # (K,)
        valid = (sparse_idx >= 0) & (sparse_idx < causal_limit)
        safe_idx = sparse_idx.clamp_min(0)
        logical_pages = torch.div(safe_idx, block_size, rounding_mode="floor")
        valid = valid & (logical_pages < num_block_columns)
        safe_pages = logical_pages.clamp(max=num_block_columns - 1)
        page_offsets = torch.remainder(safe_idx, block_size)
        phys_blocks = block_table[req, safe_pages].to(torch.int64)
        valid = valid & (phys_blocks >= 0) & (phys_blocks < packed_kv_cache.shape[0])
        safe_phys = phys_blocks.clamp(min=0, max=packed_kv_cache.shape[0] - 1)

        gathered_kv = packed_kv_cache[safe_phys, page_offsets, 0]  # (K, D)

        q = ql_nope[t]  # (N, D)
        scores = torch.matmul(q, gathered_kv.T).float() * scale  # (N, K)
        scores = scores.masked_fill(~valid[None, :], score_mask_value)
        probs = torch.softmax(scores, dim=-1)
        probs = torch.where(valid[None, :], probs, torch.zeros_like(probs))
        probs = probs.to(ql_nope.dtype)
        output[t] = torch.matmul(probs, gathered_kv)

    return output


def run_kernel(inputs: dict) -> torch.Tensor:
    """Call the op the same way execute_sparse_attention_indexer_kpool_mla does."""
    query = torch.cat([inputs["ql_nope"], inputs["q_pe"]], dim=-1).contiguous()
    result = torch.ops._C_ascend.npu_sparse_flash_attention(
        query=query,
        key=inputs["packed_kv_cache"],
        value=inputs["packed_kv_cache"],
        sparse_indices=inputs["topk_indices"],
        scale_value=inputs["scale"],
        sparse_block_size=1,
        block_table=inputs["block_table"],
        actual_seq_lengths_query=inputs["actual_seq_lengths_query"],
        actual_seq_lengths_kv=inputs["actual_seq_lengths_key"],
        query_rope=None,
        key_rope=None,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
        attention_mode=2,
        return_softmax_lse=False,
    )
    return result[0] if isinstance(result, tuple) else result


def compare(golden: torch.Tensor, actual: torch.Tensor) -> bool:
    g = golden.detach().float().cpu()
    a = actual.detach().float().cpu()

    if g.shape != a.shape:
        print(f"[FAIL] shape mismatch: golden={tuple(g.shape)} actual={tuple(a.shape)}")
        return False

    if not torch.isfinite(a).all():
        print("[FAIL] kernel output contains NaN or Inf")
        return False

    if bool((a == 0).all()) and not bool((g == 0).all()):
        print("[FAIL] kernel output is all zeros while golden is not -- kernel computed nothing")
        return False

    try:
        torch.testing.assert_close(a, g, atol=BF16_ATOL, rtol=BF16_RTOL)
        print("  torch.testing.assert_close PASSED")
        ok = True
    except AssertionError as e:
        print(f"  torch.testing.assert_close FAILED:\n  {e}")
        ok = False

    diff = (g - a).abs()
    max_abs = float(diff.max())
    max_expected = float(g.abs().max())
    relative_diff = max_abs / max_expected if max_expected > 0 else 0.0
    cos = float(
        torch.nn.functional.cosine_similarity(g.reshape(1, -1), a.reshape(1, -1)).item()
    )
    print(f"  max_abs_diff={max_abs:.6f}, relative_diff={relative_diff:.6f}, cosine={cos:.6f}")
    return ok


import random


def run_batch(n: int, heads: int, block_size: int, base_seed: int) -> int:
    """Run N randomized test cases with varying shapes."""
    rng = random.Random(base_seed)
    passed = 0
    failed = 0
    fail_cases = []

    print(f"Running {n} randomized test cases (heads={heads}, block_size={block_size})")
    print("-" * 72)

    for i in range(n):
        tokens = rng.choice([1, 2, 4, 8, 10, 16, 32, 64, 128])
        cur_heads = rng.randint(1, 16)
        actual_kv = rng.choice([10, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16400])
        topk = rng.choice([512, 1024, 2048, 2051])
        topk = min(topk, actual_kv)
        seed = base_seed + i

        inputs = make_inputs(
            tokens=tokens,
            heads=cur_heads,
            actual_kv=actual_kv,
            topk=topk,
            block_size=block_size,
            seed=seed,
        )

        golden = run_golden(inputs)
        actual = run_kernel(inputs)

        g = golden.detach().float().cpu()
        a = actual.detach().float().cpu()

        all_zero_fail = bool((a == 0).all()) and not bool((g == 0).all())
        finite_ok = bool(torch.isfinite(a).all())

        try:
            torch.testing.assert_close(a, g, atol=BF16_ATOL, rtol=BF16_RTOL)
            close_ok = True
        except AssertionError:
            close_ok = False

        ok = close_ok and finite_ok and not all_zero_fail

        diff = (g - a).abs()
        max_abs = float(diff.max()) if g.numel() > 0 else 0.0

        status = "PASS" if ok else "FAIL"
        print(
            f"  [{i+1:3d}/{n}] {status} | T={tokens:3d} N={cur_heads:2d} actual_kv={actual_kv:5d} "
            f"topk={topk:5d} seed={seed} | max_abs_diff={max_abs:.6f}"
        )

        if ok:
            passed += 1
        else:
            failed += 1
            fail_cases.append(
                f"T={tokens} N={cur_heads} actual_kv={actual_kv} topk={topk} seed={seed}"
            )

    print("-" * 72)
    print(f"Results: {passed} passed, {failed} failed out of {n}")
    if fail_cases:
        print("Failed cases:")
        for c in fail_cases:
            print(f"  {c}")

    import gc
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()

    return 0 if failed == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=None, help="T in TND layout (ignored in batch mode)")
    parser.add_argument("--heads", type=int, default=16, help="query head count")
    parser.add_argument("--actual-kv", type=int, default=None, help="actual KV length (ignored in batch mode)")
    parser.add_argument("--topk", type=int, default=None, help="sparse index count (ignored in batch mode)")
    parser.add_argument("--block-size", type=int, default=128, help="page size")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=0,
                        help="Run N randomized test cases. 0 means single run with explicit params.")
    args = parser.parse_args()

    if args.batch > 0:
        return run_batch(args.batch, args.heads, args.block_size, args.seed)

    tokens = args.tokens or 10
    actual_kv = args.actual_kv or 10
    topk = args.topk or 2051

    print("=" * 72)
    print(
        f"rope=None precision check | T={tokens} N={args.heads} "
        f"D={KV_LORA_RANK} actual_kv={actual_kv} topk={topk} "
        f"block_size={args.block_size}"
    )
    print("=" * 72)

    inputs = make_inputs(
        tokens=tokens,
        heads=args.heads,
        actual_kv=actual_kv,
        topk=topk,
        block_size=args.block_size,
        seed=args.seed,
    )

    print("[inputs]")
    print("  " + tensor_stats("ql_nope", inputs["ql_nope"]))
    print("  " + tensor_stats("q_pe", inputs["q_pe"]))
    print("  " + tensor_stats("packed_kv_cache", inputs["packed_kv_cache"]))
    valid_topk = int((inputs["topk_indices"] >= 0).sum())
    print(f"  topk_indices: shape={tuple(inputs['topk_indices'].shape)}, valid(>=0)={valid_topk}")

    return report(inputs)


def report(inputs: dict) -> int:
    print("\n[golden] running PyTorch reference ...")
    golden = run_golden(inputs)
    print("  " + tensor_stats("golden", golden))

    print("\n[kernel] running npu_sparse_flash_attention (rope=None) ...")
    actual = run_kernel(inputs)
    print("  " + tensor_stats("actual", actual))

    print("\n[compare]")
    ok = compare(golden, actual)
    print("\n" + ("PASS" if ok else "FAIL"))

    import gc
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()

    return 0 if ok else 1


def test_sfa_no_rope_batch():
    """pytest entry: run 50 randomized cases."""
    ret = run_batch(n=50, heads=16, block_size=128, base_seed=1024)
    assert ret == 0, "Some test cases failed"


if __name__ == "__main__":
    sys.exit(main())
