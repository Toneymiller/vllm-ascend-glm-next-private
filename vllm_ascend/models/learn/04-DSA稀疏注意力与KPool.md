# 第 04 ：DSA 稀疏注意力与 KPool（本仓库最硬核的部分）

> 对应源码：[`glm5_next.py:261-1402`](../glm5_next.py#L261-L1402)（三个类）+ `ops/glm5_next_lightning_indexer.py` + `ops/glm5_next_kpool_state_compress.py`
> 三个类分工：`AscendGlm5NextIndexer`（权重和缓存的 owner，[:1235](../glm5_next.py#L1235)）→ `AscendSparseAttnIndexerKpool`（算法，[:370](../glm5_next.py#L370)）→ 两个 cache 类（[:261](../glm5_next.py#L261), [:312](../glm5_next.py#L312)）

---

## 0. 直觉与术语（新手先读这节）

**一个类比读懂全程**：注意力 = 图书馆查资料。1M 上下文 = 馆里 100 万本书；普通注意力 = 写论文前把 100 万本全翻一遍（不可能）。DSA 加了一套**图书检索系统（Indexer）**：先按"相关性分"挑出 top-k 本，只精读这几本。

**KPool 的比喻**：检索柜空间有限，把相邻的 **16 本书（一个 pool）压成一张摘要卡**放进检索柜（压缩缓存）；还没凑满 16 本的新书先放**暂存架（state cache）**，保证最新内容永远可查。

术语卡（后文反复出现）：

| 术语 | 意思 |
|---|---|
| top-k | 每 query 只选分数最高的 k 个 key 参与注意力 |
| pool / index_kpool=16 | 16 个相邻 token 打包成一个"池子"，压缩后占 1 个 cache 槽 |
| 分页缓存 / block_table | KV cache 按固定大小的"页"分配（像 OS 内存页）；block_table 是逻辑页号→物理页号的页表（第 0 课 §0.7） |
| slot_mapping | 本步每个新 token 该写入 cache 的物理格号 |
| cu_seqlen_ks/ke | 变长批量时标记"每行合法区间"，实现因果掩码 |
| 尾巴（tail） | 还没凑满 16 个、没进压缩池的最新几个 token |

## 1. 问题：为什么需要稀疏化

第 3 课算过：1M 上下文 × 512 维 × 45 层 ≈ 45 GB KV cache，这还能忍；但**注意力计算**要 QK^T 全量打分，1M×1M 的打分矩阵不可计算。

DSA（DeepSeek Sparse Attention）思路：训练时模型就学会了用一个小网络（Indexer）给每个 (query, key) 对打"相关性分"，推理时每个 query 只对 **top-k 个 key** 做精确注意力。k << 1M，计算量骤降。

GLM5-Next 在 DSA 之上加了 **KPool（Ascend 独有）**：被选中与否不该非黑即白，用一个 16 token 的"压缩池"平滑过渡。

核心参数（第 1 课）：`index_topk`（每 query 选多少 key）、`index_n_heads` / `index_head_dim`（Indexer 自己的头数/维度）、`index_kpool=16`（池大小）。

## 2. 全景数据流（背下这张图）

```
                          ┌──────────────── 每个新 token 到达 ────────────────┐
                          │                                                   │
hidden [T,4096] ──┬── wq_b(qr) ──▶ q [T, H_idx, D_idx]        （Indexer 的 Q）
                  │
                  └── wk_weights_proj(hidden) [FP32]
                       ├─ 前 head_dim 列 ── k_norm(LayerNorm) ──▶ k [T, D_idx]（Indexer 的 K）
                       └─ 后 n_heads 列 ─────────────▶ weights [T, H_idx]（每头打分权重）
                  │
                  └── gate = hidden @ compress_gate^T [T, D_idx]（FP32，压缩门控分）

  ① kpool_state_compress_and_write_cache（融合算子）：
     每 16 个 token 的 (k, gate) 组成一个 pool：
       compressed_k = softmax(gate + ape) 加权求和 16 个 k   ← 注意力式池化！
       → 写进 BF16 分页 indexer K cache（压缩后 1 token 占 1 槽）
       同时把 (k, gate) 原始 FP32 写进 state cache（尾巴状态）

  ② glm5_next_lightning_indexer：
       q·compressed_k 打分 → 每 query 选 pool 级 top-k
       → 展开成 token 级 top-k → 追加因果尾巴（不足 16 个的最新 token）
       → topk_indices_buffer

  ③ 主注意力（AscendIndexerKPoolMLAAttention）只对 top-k token 算 MLA
```

## 3. Indexer 的权重（[:1263-1331](../glm5_next.py#L1263-L1331)）

| 参数 | 形状 | 作用 |
|---|---|---|
| `wq_b` | [H_idx×D_idx, q_lora_rank] | 从压缩后的 Q 低秩向量再投出 Indexer 的多头 Q |
| `wk_weights_proj` | [D_idx+H_idx, 4096]（Merged） | K 投影 + 每头权重，一矩阵两用 |
| `k_norm` | LayerNorm(D_idx) | K 投影后归一化 |
| `index_kpool_compress_gate` | [D_idx, 4096] BF16 | 压缩打分门控 |
| `index_kpool_compress_ape` | [16, D_idx] FP32，**零初始化** | 池内位置偏置（Absolute Positional Encoding of pool），可学习 |

注意 `wk_weights_proj` 的 Merged 设计：一个矩阵输出 `D_idx + H_idx` 列，前 `D_idx` 列是 K 的投影方向，后 `H_idx` 列是每个 Indexer 头的打分权重——对应 checkpoint 里的 `wk` 和 `weights_proj` 两个权重（第 09 课的 `mla_params_mapping`）。

## 4. 三块显存（cache 三兄弟）

| 类 | 存什么 | dtype | 粒度 |
|---|---|---|---|
| `AscendGlm5NextIndexerKPoolCache`（[:261](../glm5_next.py#L261)） | 压缩后的 Indexer K | BF16 | 每 pool 1 槽，分页 |
| `AscendGlm5NextCompressorStateCache`（[:312](../glm5_next.py#L312)） | 每个 token 的 (k, gate) 尾巴 | **FP32** | 每 token，滑动窗口 |
| 主 MLA 的 KV cache（第 3 课） | c_KV 隐向量 | 模型 dtype | 每 token，分页 |

`CompressorStateCache` 用 FP32 是精度设计（[:312-321](../glm5_next.py#L312-L321) 注释）：压缩是 softmax 加权和，用 BF16 累加误差会放大；只有压缩**结果**才降级成 BF16 进 indexer cache。它不是环形 buffer，而是"每请求一页的滑动尾巴"（`sliding_window = compress_ratio = 16`）。

三个 cache 都注册进 `compilation_config.static_forward_context`（[:293](../glm5_next.py#L293) 等）——这是 ACLGraph 静态捕获找到"这层的 cache 张量"的机制，重复注册同名会直接 raise。

## 5. KPool 压缩数学（[:1013-1092](../glm5_next.py#L1013-L1092)）

```python
scores = slot_score.float() + ape.float().unsqueeze(0)      # [pool, D] 16 个 token 各自的门控分+位置偏置
compressed_k = (softmax(scores, dim=1) * slot_k.float()).sum(dim=1)   # 加权平均 → [1, D]
```

直觉：**16 个相邻 token 的 Indexer-K 被软聚类成 1 个代表 K**。门控分由 hidden 学出，`ape` 让池内位置可分辨（比如池里最新的 token 可以被加权更高）。写回用 `_scatter_paged_cache`（[:453](../glm5_next.py#L453)）处理分页 + 无效 slot（ACLGraph 兼容：固定形状 + row-0 哨兵技巧，注释 [:448](../glm5_next.py#L448)）。

## 6. top-k 选择（[:490-1011](../glm5_next.py#L490-L1011)）

两条路径，forward_ascend（[:1140](../glm5_next.py#L1140)）实际用的是 **NPU 融合算子**：

- **解码/统一路径**：`torch.ops._C_ascend.npu_lightning_indexer(..., sparse_mode=3)`（[:490](../glm5_next.py#L490)）直接吐 pool 级 top-k
- **纯 torch 参考路径**（`indexer_kpool_topk_pytorch`，[:806](../glm5_next.py#L806)）：读代码理解用，逐 16-query × 2048-key 分块，`bf16_mqa_logits` + `top_k_per_row_prefill` 交替合并局部最优

三个必须懂的数学技巧：

1. **权重吸收**（`bf16_mqa_logits`，[:705-745](../glm5_next.py#L705-L745)）：
   ```python
   # sum_h(weight_h * dot(q_h, k)) == dot(sum_h(weight_h * q_h), k)
   weighted_q = (query * weights.unsqueeze(-1)).sum(dim=1)
   logits = weighted_q @ key^T
   ```
   多头打分先在 Q 侧加权合并成单头，避免物化 `[M, H, N]` 大张量。
2. **因果 mask**：`cu_seqlen_ks/ke` 每行的合法区间，区间外填 `finfo.min`，top-k 自动不会选到。
3. **pool→token 展开**（`expand_pools_to_tokens`，[:522](../glm5_next.py#L522)）：`token_id = pool_id * 16 + [0..15]`，一个 pool 换 16 个 token 索引。

**因果尾巴**（`append_tail_to_topk`，[:572](../glm5_next.py#L572)）：最新不足 16 个、还没组成完整 pool 的 token 不在压缩缓存里，必须手动追加到 top-k 末尾（最多 15 个）——这就是 `topk + index_kpool - 1` 宽度、以及 `index_kpool_always_select_tail=True` 的由来。

## 7. Indexer.forward 串联（[:1333-1400](../glm5_next.py#L1333-L1400)）

```python
q, _ = self.wq_b(qr)                       # Indexer Q
hidden_f32 = hidden_states.float()         # ★ K 链路全程 FP32
kw = hidden_f32 @ wk_weights_proj^T
weights = kw[:, head_dim:].to(BF16)        # 每头权重
k = self.k_norm(kw[:, :head_dim])          # LayerNorm 后的 K
# rope_dim=0 → 跳过 RoPE 分支
weights = weights * (softmax_scale * n_heads**-0.5)
gate_score = hidden_f32 @ compress_gate^T  # 压缩门控分
return self.indexer_op(hidden_states, q, weights,
                       k=k, gate_score=gate_score,
                       compress_ape=self.index_kpool_compress_ape,
                       index_kpool=self.index_kpool, positions=positions)
```

`forward_ascend`（[:1140](../glm5_next.py#L1140)）里两个融合算子的分工（[:1152-1164](../glm5_next.py#L1152-L1164) 注释值得精读）：一个算子包办"写状态缓存 + 聚池窗口 + 压缩 + 写 indexer 缓存"，另一个做 top-k——把 torch 组合会爆炸成的小算子流（aclnnIndex/SearchSorted 洪泛）收进单个 kernel，同时绕开 AscendC 原生 key_pool 算子的性能回退。**这是"为什么不用现成算子"的活教材。**

MTP 的处理（[:1192-1200](../glm5_next.py#L1192-L1200)）：eager 模式下用 `min(attn_metadata.num_actual_tokens, positions.shape[0])` 砍掉 padding 行；FULL 图模式保持固定形状。

## 8. 过关自测

1. 一条新 token（序列长 1000）到达，写出它经过的完整流水：哪些张量进了哪块 cache？top-k 候选有多少个？（答：1000/16≈62 个压缩 pool 候选 + 最多 15 个尾巴 token；pool 级 top-k 选 `topk//16` 个 pool）
2. 为什么压缩必须用 FP32 state cache，而 indexer K cache 用 BF16 就够？
3. `weights` 吸收技巧为什么在 Indexer 可行、在主注意力不行？（提示：主注意力 K/V 要从 c_KV 重建，weight 是逐 token 的）
4. `index_kpool_compress_ape` 为什么零初始化也能训练？

---

**下一课**：[05-KDA线性注意力.md](05-KDA线性注意力.md#L05) —— 换一条赛道：没有 KV cache 的注意力怎么工作。
