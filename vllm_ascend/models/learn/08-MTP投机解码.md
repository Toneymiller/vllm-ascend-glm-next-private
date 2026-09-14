# 第 08 课：MTP 投机解码（glm5_next_mtp.py）

> 对应源码：`glm5_next_mtp.py`（363 行）
> 三个类由内到外：`MultiTokenPredictorLayer`（单层）→ `MultiTokenPredictor`（层容器）→ `AscendGlm5NextMTP`（引擎接口）

---

## 0. 新手包：投机解码从零

问题：decode 阶段每步只算 1 个 token，但一次前向的计算瓶颈是"读权重"而不是"算 token"——算 1 个 token 和算 10 个 token 的耗时几乎一样（都是把几百 GB 权重过一遍）。**单 token 是浪费。**

方案（草稿-验证）：

```
1. 草稿模型（便宜）连猜 4 个 token：   [甲, 乙, 丙, 丁]
2. 主模型（贵）一次前向，把这 4 个 token 当作 4 个位置同时"验算"
3. 逐位对答案：甲✓ 乙✓ 丙✓ 丁✗ → 接受前 3 个，扔掉丁
4. 一步前向白赚 3 个 token（原来一步只能出 1 个）
```

接受率取决于草稿质量，通常 50%~80%，吞吐直接翻倍级提升。**MTP 层就是那个"草稿模型"**——它不是一个独立小模型，而是主模型顶部多挂的一个标准 decoder 层（便宜、共享架构），这是 DeepSeek 首创、GLM5-Next 沿用的做法。带着这个画面读课文，`eh_proj` 拼接、`shared_head` 验证输出全都顺理成章。

## 1. MTP 是什么（30 秒版）

主模型每步只出 1 个 token，但 GPU 算力没吃满。MTP 训练时给主模型挂 1 个额外 decoder 层（`num_nextn_predict_layers=1`，权重名 `layers.45.*`），学会"看第 t 层输出 + 第 t+1 个 token 的 embedding，预测第 t+2 个 token"。推理时这层当**草稿模型**：一次猜 n 个 token，主模型一次前向并行验证，接受率 >60% 就白赚吞吐。

## 2. PredictorLayer：拼接层（[:39-124](../glm5_next_mtp.py#L39-L124)）

```python
self.enorm = RMSNorm(hidden)      # 归一化"下一个 token 的 embedding"
self.hnorm = RMSNorm(hidden)      # 归一化"主模型最后层的 hidden"
self.eh_proj = nn.Linear(hidden*2, hidden, bias=False)   # 拼接后压回
if self.is_rot_used:              # 量化描述 is_rot_used 时才有
    self.rot = nn.Linear(hidden, hidden, bias=False)     # rot.weight（主模型加载时跳过它！）
self.shared_head = SharedHead(config, prefix, quant_config)   # 输出头
self.mtp_block = AscendGlm5NextDecoderLayer(..., is_mtp_layer=True)
```

forward（[:94](../glm5_next_mtp.py#L94)）：

```python
inputs_embeds = inputs_embeds.masked_fill(positions.eq(0).unsqueeze(-1), 0)
# ↑ 序列首 token 没有"上一个" embedding，置零对齐上游 fused_eh_norm 语义
#   注释强调：用 mask 而不是先 .item() 判断，避免 decode 热路径 D2H 同步
embeddings = enorm(inputs_embeds)
if is_rot_used: previous_hidden = rot(previous_hidden)
previous_hidden = hnorm(previous_hidden)
hidden = eh_proj(cat([embeddings, previous_hidden], -1))
hidden, _ = self.mtp_block(positions, hidden, residual=None)
hidden = tensor_model_parallel_all_reduce(hidden)
hidden = self.shared_head(hidden)
return hidden, hidden
```

注意 `mtp_block` 收 `residual=None`：第 2 课讲过，`is_mtp_layer=True` 让 DecoderLayer 走**非 MHC 分支**且 forward 里 `residual = hidden_states` 自起（[:2195-2196](../glm5_next.py#L2195-L2196)）。

**topk_indices_buffer 自建**（[:65-77](../glm5_next_mtp.py#L65-L77)）：MTP 层不在 `AscendGlm5NextModel` 里，拿不到主模型的 buffer，所以要自己开一块；宽度计算值得看：
```python
topk_width = index_topk + index_kpool - 1
topk_width = round_up(topk_width, 128)        # sparse_block_size 对齐
```
主模型（第 2 课 [:2254](../glm5_next.py#L2254)）没做 128 对齐——MTP 路径更严格。

**`SharedHead`**（上游 deepseek_mtp 引入）：内含一个 `head`（lm_head 同规格）。`compute_logits`（[:175-183](../glm5_next_mtp.py#L175-L183)）用**主模型的 lm_head 权重还是 shared_head 自己的**取决于权重加载（checkpoint 里 MTP 层有独立 head 权重就加载到 shared_head；"shared" 之名指与主模型结构共享）。

## 3. MultiTokenPredictor：层容器（[:127-183](../glm5_next_mtp.py#L127-L183)）

```python
self.mtp_start_layer_idx = config.num_hidden_layers          # 45
self.layers = ModuleDict({str(45): PredictorLayer(...)})     # key 是字符串层号
self.embed_tokens = VocabParallelEmbedding(...)              # 独立 embedding
```

forward 用 `spec_step_idx % num_mtp_layers` 轮转选层（多 MTP 层时每步换一层，GLM5 只有 1 层恒选 45）。`compute_logits` 同样按 step 找层。

`embed_input_ids` 暴露给引擎：draft 的输入 id 先查这层自己的 embedding 表。

## 4. AscendGlm5NextMTP：引擎接口（[:186-243](../glm5_next_mtp.py#L186-L243)）

```python
class AscendGlm5NextMTP(nn.Module, DeepseekV2MixtureOfExperts):
    hf_to_vllm_mapper = GLM5_TRANSFORMERS_INTERNAL_WEIGHTS_MAPPER   # 复用主模型映射
    packed_modules_mapping = GLM5_PACKED_MODULES_MAPPING
```

继承 `DeepseekV2MixtureOfExperts` 是为了实现 `MixtureOfExperts` 协议（EPLB 等功能探测 MoE 层用）。`_set_moe_parameters`（[:200](../glm5_next_mtp.py#L200)）遍历 MTP 层收集 `mtp_block.mlp` 里的 `AscendGlm5NextMoE`——草稿层也有 288 专家。

## 5. 权重重写（[:244-254](../glm5_next_mtp.py#L244-L254)）——第 9 课的前菜

checkpoint 里 MTP 权重叫 `layers.45.xxx`，模型里实际挂在 `model.layers.45.mtp_block.xxx`：

```python
def _rewrite_spec_layer_name(self, spec_layer, name):
    if name == MTP_ROT_WEIGHT_NAME:      # "rot.weight"
        return f"model.layers.{spec_layer}.rot.weight"
    if ".embed_tokens." in name:  → 去掉层号，映射到共享 embedding
    if enorm/hnorm/eh_proj/shared_head: → 原名（挂在 layer 根上）
    return name.replace(prefix, f"{prefix}mtp_block.")
```

`load_weights`（[:256-362](../glm5_next_mtp.py#L256-L362)）流程：只接受 `get_spec_layer_idx_from_weight_name` 认出的 MTP 权重（主模型加载时第 9 课会跳过它们）→ 重写名字 → 三段映射（stacked → experts → 默认）→ **最后校验**：期望的 MTP 层号集合若缺层直接 raise（[:343-362](../glm5_next_mtp.py#L343-L362)）。

## 6. 全链路视角（把 1-9 课串起来）

```
主模型 forward（第 2 课）出 hidden_t
  → MTP 层：cat(enorm(embed(t+1)), hnorm(rot(hidden_t))) → eh_proj → DecoderLayer → 猜 token
  → 主模型用真 KV cache 一次验证 n 个猜的 token
  → 接受/拒绝，循环
```

`spec_decode` 配置驱动，`glm5_next_mtp.py` 只管"层长什么样"，调度策略在 vLLM v1 spec framework 里（超纲，跳过）。

## 7. 过关自测

1. `eh_proj` 输入的两个来源分别是什么？各自先过哪个 norm？

   <details><summary>参考答案</summary>

   输入 A：**草稿 token 的 embedding**（`inputs_embeds`）→ 先过 **`enorm`**（[:112](../glm5_next_mtp.py#L112)）。输入 B：**主模型最后层输出的 hidden**（`previous_hidden_states`）→ 若量化开启先过 `rot`，再过 **`hnorm`**（[:113-115](../glm5_next_mtp.py#L113-L115)）。两者 `cat` 成 2×hidden，由 `eh_proj`（2h→h）压回一个 hidden 喂给 `mtp_block`。

   </details>

2. 为什么主模型的 `load_weights` 要跳过 `rot.weight` 和 `layers.45+`？（答：它们属于 MTP 模型，AutoWeightsLoader 的 skip_prefixes，第 9 课 [:2520](../glm5_next.py#L2520)）

   <details><summary>参考答案</summary>

   一个 checkpoint 里同时装着主模型权重和 MTP 草稿层权重（`layers.45.*`）。主模型里根本没有 45 层和 rot 参数，`AutoWeightsLoader` 遇到匹配不上的权重会报错，所以用 `skip_prefixes=[MTP_ROT_WEIGHT_NAME]`（"rot.weight"，[:2520](../glm5_next.py#L2520)）跳过；45 层权重则由 MTP 模型自己的 `load_weights` 单独加载——它只认 `get_spec_layer_idx_from_weight_name` 认出的权重（[:288-290](../glm5_next_mtp.py#L288-L290)）。各拿各的，互不干扰。

   </details>

3. `positions.eq(0)` 的 mask trick 避免了什么？如果用 `if positions[0].item() == 0` 会怎样？

   <details><summary>参考答案</summary>

   每条序列的第一个 token 没有"上一个 token 的 embedding"，按上游 `fused_eh_norm` 语义要把它置 0。若写 `if positions[0].item() == 0`：`.item()` 是从 device tensor 取值到 CPU，会触发 **D2H 同步**，decode 热路径每次都被 CPU 卡住等 NPU，AsyncScheduler 直接堵死。`masked_fill` 全程在设备上完成，零同步（这也是 AGENTS.md 的 `item()` 红线）。

   </details>

4. MTP 层为什么禁用 MHC？说出两个理由。

   <details><summary>参考答案</summary>

   ① **结构对齐**：checkpoint 里 MTP 层（layers.45）是按标准 decoder 层训练的，没有 MHC 超连接的权重/流，走 MHC 分支根本无权重可加载。② **上下文不成立**：MHC 需要主模型从第 0 层开始一路维护的多条 residual 流（第 2 课 `_expand_mhc_residual_streams` 等）；MTP 层独立挂在 `AscendGlm5NextModel` 之外，拿到的只是主模型最后层的普通单流 hidden，没有前序 MHC 流可言。所以 `is_mtp_layer=True` 强制走标准分支且 `residual = hidden_states` 自起（[:2195-2196](../glm5_next.py#L2195-L2196)）。

   </details>

---

**最后一课**：[09-权重加载与模型注册.md](09-权重加载与模型注册.md#L09) —— 收尾：checkpoint 的名字怎么变成显存里的参数。
