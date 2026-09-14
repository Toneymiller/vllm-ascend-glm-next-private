# 第 06 课：MLP 与 MoE

> 对应源码：[`glm5_next.py:1403-1539`](../glm5_next.py#L1403-L1539)
> 两个类：`AscendGlm5NextMLP`（dense 退路）、`AscendGlm5NextMoE`（主菜）

---

## 0. 新手包：MoE 一遍手算（第 0 课 §0.5 的展开）

4 个专家选 2 个（top-2），一个 token 的完整路由过程：

```
输入 x → router 打分 logits = [2.0, 0.5, 1.5, 0.1]     # 4 个专家各一个分
sigmoid → 分数 = [0.881, 0.622, 0.818, 0.525]
选 top-2 → 专家 0、专家 2 入选
归一化权重：0.881/(0.881+0.818)=0.518，0.818/(...) = 0.482
输出 = 0.518 × FFN_0(x) + 0.482 × FFN_2(x)
```

对比 dense FFN：`输出 = FFN(x)`，全员必算。MoE 的输出只动了**两个**专家 FFN + 一次加权——计算省了，但路由必须精确（这就是课文里 `force_fp32_compute` 的原因）。

GLM5-Next 的真实数字：288 选 8 + 1 共享。看完手算再读课文，每个参数都能对号入座。

## 1. AscendGlm5NextMLP：教科书 SwiGLU（[:1403](../glm5_next.py#L1403)）

```python
self.gate_up_proj = MergedColumnParallelLinear(hidden, [intermediate]*2)  # gate|up 拼一起
self.down_proj = RowParallelLinear(intermediate, hidden)
self.act_fn = SiluAndMulWithClamp(swiglu_limit) if swiglu_limit else SiluAndMul()
```

`SiluAndMulWithClamp` 是 vLLM 提供的变体：激活前先 clamp 到 ±limit（`swiglu_limit` 非 None 时）。文本 config 的 `swiglu_limit` 默认 None（不 clamp），**视觉塔才用 clamp**（第 7 课）。TP 切分：gate_up 按列切、down 按行切，前向各一次 all-reduce（由 RowParallel 的 reduce_results 完成）。

## 2. AscendGlm5NextMoE（[:1445](../glm5_next.py#L1445)）

### 2.1 成员三件套

```python
self.shared_experts = AscendGlm5NextMLP(hidden, 2048×1, reduce_results=False)
self.gate = GateLinear(hidden, 288, params_dtype=FP32, force_fp32_compute=True)
self.experts = FusedMoE(num_experts=288, top_k=8, intermediate_size=2048,
                        use_grouped_topk=True, scoring_func="sigmoid",
                        routed_scaling_factor=2.5, renormalize=..., ...)
```

| 组件 | 关键点 |
|---|---|
| **shared_experts** | 每 token 必走，`reduce_results=False`——它的输出不自己做 all-reduce，留给 FusedMoE 的输出合并后统一归约（省一次通信） |
| **gate** | 路由打分。FP32 强制计算：路由是"选 8 个"的 argmax 型操作，BF16 的分数抖动会改变选择，精度必须拉满 |
| **FusedMoE** | vLLM 统一 MoE 引擎：内部管专家权重排布、token 分发、EP/TP 通信 |

`e_score_correction_bias`（[:1488](../glm5_next.py#L1488)）：`topk_method == "noaux_tc"` 时启用（DeepSeek V3 同款：给每个专家分数加可学习 bias 后再排 top-k，bias 不进sigmoid 加权）。GLM5-Next 的 config 里没设 `topk_method`，默认关闭。

### 2.2 EPLB：专家负载均衡（[:1493-1500](../glm5_next.py#L1493-L1500)）

```python
n_physical_experts = 288 + n_redundant_experts   # 冗余专家
n_local = n_physical // ep_size                  # 每 EP rank 分到的物理专家
```

EP（Expert Parallel）切专家不切 token。热点专家可复制冗余副本，EPLB 模块动态调整映射（`vllm_ascend/eplb/`）。读懂这 8 行，就懂了 `FusedMoE` 的 `enable_eplb / num_redundant_experts` 参数从哪来。

### 2.3 forward（[:1525](../glm5_next.py#L1525)）

```python
def forward(self, hidden_states):
    if self.is_sequence_parallel:
        hidden_states = sequence_parallel_chunk(hidden_states)   # token 维切给 TP 组
    if self.experts.is_internal_router:
        out = self.experts(hidden_states, router_logits=hidden_states)
    else:
        router_logits, _ = self.gate(hidden_states)
        out = self.experts(hidden_states, router_logits)
    if self.is_sequence_parallel:
        out = tensor_model_parallel_all_gather(out, 0)[:num_tokens]
    return out.view(num_tokens, hidden_dim)
```

主分支就两步：gate 出 logits（[T, 288] FP32）→ `self.experts(...)` 出加权混合。`is_internal_router` 是 FusedMoE 的新接口：路由逻辑内化后直接传 hidden。序列并行模式下输出用 all-gather 拼回 token 维再裁剪 padding。

### 2.4 数字感受

每层 MoE 参数量：288 × 3 × 4096 × 2048 ≈ **7.2B**/层。45 层 ≈ 325B 总专家参数，但每 token 只激活 8/288 → 激活参数约 9B + 共享专家 + 注意力。这就是 MoE 的经济学。

## 3. 与 DecoderLayer 的接线（回忆第 2 课）

```python
if self.is_moe and num_experts and mlp_type == "sparse":
    self.mlp = AscendGlm5NextMoE(...)     # "sparse" = MoE
else:
    self.mlp = AscendGlm5NextMLP(...)     # "dense"
```

注意 MTP 层也会拿到一个完整 MoE（第 8 课 `_set_moe_parameters` 收集它）——草稿层和主模型共享 288 专家的路由分布，但权重独立。

## 4. 过关自测

1. 为什么 gate 要 `force_fp32_compute=True`，而专家内部用 BF16？

   <details><summary>参考答案</summary>

   因为**路由是离散决策，专家是连续计算**。gate 出 288 个分数后要选 top-8——两个分数若挨得近（0.5001 vs 0.4999），BF16 的舍入误差就足以翻转"选谁"，选错专家输出天差地别：数值小误差 → 结果离散跳变。专家 FFN 内部则是连续变换，数值小抖动只引起输出小抖动，BF16 省一半显存带宽、NPU 算得更快，完全划算。一句话：**argmax 型操作怕误差，连续计算容忍误差**（第 0 课 §0.5 的结论）。

   </details>

2. `shared_experts` 的 `reduce_results=False` 省了什么？

   <details><summary>参考答案</summary>

   省一次 all-reduce。TP 下 down_proj（RowParallel）的输出是各 rank 的部分和，默认要 all-reduce 才完整。shared_experts 每 token 必走，若 `reduce_results=True`：它自己归约一次 + FusedMoE 路由部分合并后再归约一次 = **2 次通信**。设成 False 后，shared 的部分和先不归约，与 FusedMoE 输出相加后**统一归约一次**——利用加法结合律把两次通信合成一次，整卡通信量直接减半（这层 MoE 的通信是按隐藏维大张量算的）。

   </details>

3. EP=8 时每个 rank 存几个专家的权重？token 的路由结果在几个 rank 上被计算？

   <details><summary>参考答案</summary>

   288 ÷ 8 = **36 个专家/rank**（无冗余时；开 EPLB 则按 `n_physical_experts = 288 + 冗余数` 再除，§2.2）。计算侧：每个 token 选 8 个专家，这 8 个专家分散在若干 rank 上——**每个被选中的专家都在它所属的 rank 上算**，所以一个 token 的专家 FFN 最多分散在 8 个 rank 上并行计算（8 个选择全落同一 rank 时只 1 个），之后跨 rank 合并。

   </details>

4. 一个 token 过一层 MoE，最多触发几次 all-reduce 类集合通信？（提示：router 不需要；专家输出合并 1 次；序列并行下另有 all-gather）

   <details><summary>参考答案</summary>

   **最多 2 次**。router（gate）是本地矩阵乘 + top-k，零通信；路由专家输出与 shared 部分和合并后统一 **all-reduce 1 次**（第 2 题的 `reduce_results=False` 保证只此一次）；若开序列并行（`is_sequence_parallel`，§2.3），输出还要 **all-gather 1 次**拼回 token 维 → 合计 2 次。

   </details>

---

**下一课**：[07-多模态结构.md](07-多模态结构.md#L07) —— 离开文本主干，看图怎么变成 token。
