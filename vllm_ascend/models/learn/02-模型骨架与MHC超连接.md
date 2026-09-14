# 第 02 课：模型骨架与 MHC 超连接

> 对应源码：[`vllm_ascend/models/glm5_next.py:2053-2532`](../glm5_next.py#L2053-L2532)
> 三层嵌套：`AscendGlm5NextForCausalLM` → `AscendGlm5NextModel` → `45 × AscendGlm5NextDecoderLayer`

---

## 0. 读前须知：四个会反复出现的词

1. **pre-norm 骨架**：`x = x + SubLayer(Norm(x))`。本课"非 MHC 分支"就是它的原样实现。
2. **PP / PPMissingLayer / IntermediateTensors**：45 层可能被切到多张卡（流水线并行）。不归本卡的层用 `PPMissingLayer` 占位；跨卡交接数据用 `IntermediateTensors` 这个"行李箱"。
3. **@support_torch_compile / ACLGraph**：把 Python 前向固化成静态执行图反复回放。代价是张量形状必须固定——所以代码里全是"预分配 buffer、往里写"的写法。
4. **写入式接口**：很多子层的 forward 不返回结果，而是往调用方预分配的 `output` 张量里写。第一次见到别当成 bug。

## 1. 顶层：AscendGlm5NextForCausalLM（[:2437](../glm5_next.py#L2437)）

vLLM 模型类的标准五件套，注意它继承的 4 个"能力接口"：

```python
class AscendGlm5NextForCausalLM(nn.Module, HasInnerState, SupportsPP,
                                MixtureOfExperts, IsHybrid):
    has_inner_state: ClassVar[bool] = True   # 有循环状态（KDA）
    is_hybrid: ClassVar[bool] = True         # 混合注意力（KV + 循环状态共存）
```

| 成员 | 作用 |
|---|---|
| `self.model = AscendGlm5NextModel(...)` | 主干 |
| `self.lm_head = ParallelLMHead(vocab, hidden)` | 输出投影（仅 PP 最后 rank） |
| `self.logits_processor` | 取最后一个 token 算 logits |
| `embed_input_ids()` | 供多模态模型复用 embedding |
| `get_mamba_state_*` 三件套 | 告诉引擎 KDA 状态的 shape/dtype/拷贝函数（[:2489-2513](../glm5_next.py#L2489-L2513)），hybrid KV cache 管理靠它 |

`packed_modules_mapping` 和 `hf_to_vllm_mapper` 是给权重加载用的，第 09 课讲。

## 2. 中层：AscendGlm5NextModel（[:2244](../glm5_next.py#L2244)）

```python
@support_torch_compile          # torch.compile 支持（动态 batch 维）
class AscendGlm5NextModel(nn.Module):
```

`__init__` 里的三个关键动作：

1. **预分配 topk_indices_buffer**（[:2252-2261](../glm5_next.py#L2252-L2261)）：
   ```python
   topk_indices_buffer = torch.empty(
       max_num_batched_tokens,
       config.index_topk + config.index_kpool - 1,  # topk + 尾巴
       dtype=torch.int32, device=...)
   ```
   这是 DSA Indexer 输出 top-k 索引的**全局共享工作区**，45 层复用同一块显存。宽度多出来的 `kpool-1` 给"因果尾巴"（第 04 课）。

2. **embed_tokens**：只有 PP 首 rank 分配（[:2263](../glm5_next.py#L2263)），否则 `PPMissingLayer`。

3. **make_layers**（[:2282](../glm5_next.py#L2282)）：vLLM 标准层工厂，处理 PP 切分——每个 rank 只建 `start_layer:end_layer`。

`forward`（[:2293](../glm5_next.py#L2293)）：教科书式流水线
```python
if PP首rank:  hidden = inputs_embeds or embed_tokens(input_ids)
else:         hidden = intermediate_tensors["hidden_states"]   # PP 中继
for layer in layers:  hidden, residual = layer(positions, hidden, residual)
if PP末rank:  hidden = self.norm(hidden)
else:         return IntermediateTensors({"hidden_states", "residual"})
```

注意 `residual` 是"流水线化的残差"：层与层之间只传 `(hidden, residual)` 两个张量，这是 vLLM 所有 LLaMA 系模型的通用写法。为什么？

**HF 原版的痛点**：pre-norm 每层有两个加法点（`x = x + Attn(...)`、`x = x + FFN(...)`），而下一层拿到主路后的第一件事必然是"先还账再 Norm"——**加法和 RMSNorm 总是挨在一起出现**，分开算就要读两遍写两遍显存。

**vLLM 的写法**：把加法拆成两半，本层只算"还没加进主路的子层输出"（`hidden`），把欠着的账（`residual`）一起传给下一层；下一层的第一个 RMSNorm 是融合版 `add_rms_norm(hidden, residual)`，顺手还账再归一化，一个 kernel 干两件事：

```
层 N 出口：  hidden = FFN输出（没加）    residual = 主路（欠一次加法）
                │ 两个张量一起传下去
层 N+1 入口：input_layernorm(hidden, residual)
             → residual += hidden      ← 账在这里还
             → 返回 Norm(residual)      ← 注意力的输入
```

**不变式**：层间传递时，`residual` = 截至上一半场的主路累积和（差一次加法没还），`hidden` = 刚算出来、等待入账的那一份。

三个收益：

1. **省显存带宽**（主要原因）：加法和 RMSNorm 都是显存受限操作，融合成 `add_rms_norm` 一个算子基本白送
2. **PP 无缝交接**：流水线并行时"行李箱"（`IntermediateTensors`）里装的正是这两个张量，下一段的第一个 norm 天然接着还账，段内段外语义一致
3. **ACLGraph 友好**：两个张量从头传到尾，静态图捕获时缓冲区分配模式固定

对照本模型：非 MHC 分支（[:2195-2210](../glm5_next.py#L2195-L2210)）出口 `return hidden_states, residual`；Model 层最后一层出来后 `self.norm` 还掉最后一笔账再进 lm_head。**MHC 分支反而返回 `None`**（[:2241](../glm5_next.py#L2241)）——4 条流状态本身就是完整主路，没有"欠着的加法"可传。

一句话：`(hidden, residual)` = "刚出锅的菜" + "还没收盘子的账"，传给下一层让它边收盘边做下一个菜。

## 3. 底层：AscendGlm5NextDecoderLayer（[:2053](../glm5_next.py#L2053)）

### 3.1 构造期分叉（回忆第 1 课）

```python
self.layer_kind = "kda" if _is_kda_layer(config, layer_idx) else "mla"
if _is_kda_layer(config, layer_idx):
    self.self_attn = KimiGatedDeltaNetAttention(...)   # 上游 KDN 层（第 05 课外围）
else:
    self.self_attn = AscendGlm5NextMLAAttention(...)   # NPU 自研 MLA（第 03 课）

if mlp_type == "sparse":
    self.mlp = AscendGlm5NextMoE(...)
else:
    self.mlp = AscendGlm5NextMLP(...)
```

### 3.2 MHC：把 1 条残差流变成 4 条（人话版）

**一句话直觉**：标准残差是一条国道——子层（注意力/FFN）从主路取货、加工完原路还回，`x = x + Attn(Norm(x))`。MHC 把这条国道扩成 **4 条并行高速**：每层先"取材料"（4 条流加权混合出一份给子层），子层干完活再"还回去"（结果按可学习权重撒回 4 条道）。**怎么取、怎么还，全是训练学出来的**。

| | 标准 pre-norm | MHC（`mhc=True`） |
|---|---|---|
| 残差流 | 1 条 [T, 4096] | 4 条，内存里摊平成 [T, 16384] |
| 子层输入 | `Norm(x)`——就是主路本身 | 4 条流的**可学习加权混合** |
| 子层输出合并 | `x = x + out`，固定加法 | 按可学习权重**撒回 4 条流** |
| 谁决定怎么接 | 写死 | 每层 2 组混合参数（注意力前 / FFN 前各一套） |

**三个关键认知**（读代码前先记住）：

1. **子层本身没变**。注意力、FFN 照旧只吃 [T, 4096] 的单流输入。MHC 全部发生在"子层怎么接进主路"上——所以 Python 代码里 MHC 的存在感只有 `hc_pre` / `hc_post` 两行，别在注意力里找它。
2. **主路状态首尾才变道**。第 0 层入口把 [T, 4096] 复制 4 份（`_expand_mhc_residual_streams`，[:148](../glm5_next.py#L148)），此后 45 层中间状态一直是 [T, 16384]，直到最后一层出口 `mean` 合并回 [T, 4096]（[:2237-2239](../glm5_next.py#L2237-L2239)）。
3. **混合是"带账本"的**。`hc_pre` 除了返回混合结果 `layer_input`，还返回 `residual_mhc`（4 条流快照）和 `post`、`comb` 两份中间记录——它们记着"这次混合用了什么系数"，`hc_post` 拿同一套账本把子层输出撒回去。取和还必须配对，账本在两个算子间传递。（post/comb 的字段级语义在 C++ 算子 `npu_hc_pre_v2` 内部，读 Python 层时当"账本"对待即可。）

**每层的 6 个可学习参数**（[:2140-2145](../glm5_next.py#L2140-L2145)，注意力/FFN 各一套，全部 FP32）：

| 参数 | 形状 | 角色 |
|---|---|---|
| `hc_*_fn` | (2+n)·n = 24 × 16384 | 混合权重：24 行 = 16（4 流×4 流的混合矩阵）+ 2×4（每流两组系数），n=4，`d_model = n×4096 = 16384`（[:2137-2138](../glm5_next.py#L2137-L2138)） |
| `hc_*_base` | (24,) | 混合的静态偏置 |
| `hc_*_scale` | (3,) | 三个全局缩放率，控制"新信息写入多猛、旧信息保留多少"的节奏 |

混合权重不是随便加权：`npu_hc_pre_v2` 内部会做 **Sinkhorn 归一化**（迭代 `mhc_sinkhorn_iterations=20` 轮，[:2157](../glm5_next.py#L2157)），保证 4 条流的混合系数幅度稳定、不随层数加深而爆炸。

混合本体不在 Python 里算，而是两个 NPU 融合算子（取和还各一个，融合成单 kernel 省调度开销）：

```python
def hc_pre(self, x, hc_fn, hc_scale, hc_base):        # [:2147](../glm5_next.py#L2147) ①取材料
    y, post, comb = torch.ops._C_ascend.npu_hc_pre_v2(
        x_mhc, hc_fn, hc_scale, hc_base, n,
        self.hc_sinkhorn_iters,    # Sinkhorn 归一化迭代（20 次）
        self.rms_norm_eps, self.hc_eps)
    return y, x_mhc, post, comb
    # y = layer_input（混合出的子层输入），x_mhc = 4 流快照，post/comb = 账本

def hc_post(self, x, residual_mhc, post, comb):        # [:2163](../glm5_next.py#L2163) ③还回去
    return torch.ops._C_ascend.npu_hc_post(...)
```

**为什么值得这么干**：

1. **梯度高速路 ×4**：反向传播时梯度有 4 条直达通道到浅层，45 层深也不容易"梯度堵车"
2. **多速率信息流**：4 条流经 45 层反复混合后，天然携带不同"新旧比例"的信息；子层每层取的混合比例可自适应——相当于给网络加了"快慢两个思考速度"
3. **代价极小**：每层 2 组参数共约 2×(24×16384+24+3) ≈ 79 万个 FP32（约 3MB），45 层合计 ~140MB，换来残差结构的整体升级

### 3.3 MHC 版 forward 全流程（[:2212-2241](../glm5_next.py#L2212-L2241)）

对照 §3.2 的"取材料 → 干活 → 还回去"三步走读，注意力半场和 FFN 半场各来一遍：

```
x = hidden  [tokens, 16384]（第 0 层先 expand；此后一直是 4 条流）
│
│ ── 注意力半场 ────────────────────────────────────────
├─ hc_pre(attn)   ①取材料：layer_input + 账本(residual_mhc/post/comb)
├─ input_layernorm(layer_input)
├─ self_attn(..., output=attn_output)   ②子层干活（照旧 [tokens, 4096]）
├─ maybe_chunk_residual(attn_output, ...) ×3   （FlashComm/序列并行的账本分片对齐）
├─ hc_post        ③还回去：attn_output 撒回 4 条流 → 新的 [tokens, 16384]
│
│ ── FFN 半场（换一套 hc_ffn_* 参数，一模一样的三步）──
├─ hc_pre(ffn) → post_attention_layernorm → mlp(MoE|dense) → hc_post
│
└─ 最后一层：mean(4 流) → [tokens, 4096] 出口
```

两个值得注意的细节：

1. **`output=attn_output` 写入式接口**（[:2171](../glm5_next.py#L2171), [:2198](../glm5_next.py#L2198)）：注意力结果不通过返回值传递，而是写进预分配的 `torch.empty_like` 缓冲——配合 ACLGraph 静态图捕获减少分配。
2. **`_make_attention_output` 的特例**（[:2171-2187](../glm5_next.py#L2171-L2187)）：多模态的第 0 层输入是完整视觉 embedding 序列，FlashComm1（TP 内 reduce-scatter）开启时输出 buffer 要按 `N/tp` 尺寸分配。

### 3.4 非 MHC 退路（[:2195-2210](../glm5_next.py#L2195-L2210)）

`mhc=False` 或 **MTP 层**（`is_mtp_layer=True`）走普通 pre-norm。MTP 层禁用 MHC 是明确的设计选择：草稿层要轻量。

## 4. 全局数据流一图流

```
input_ids [T]
   │ VocabParallelEmbedding
   ▼
hidden [T, 4096] ──第0层expand──▶ [T, 16384]   ← MHC 4 流开始
   │
   ▼ ×45 层（每层）
   ┌────────────────────────────────────────┐
   │ hc_pre(attn)  → norm → attn(KDA|MLA)   │
   │               → hc_post                │
   │ hc_pre(ffn)   → mlp(MoE|dense)         │
   │               → hc_post                │
   └────────────────────────────────────────┘
   │
   ▼ 最后一层 mean(4流)
hidden [T, 4096] → norm → lm_head → logits [T, 154880]
```

## 5. 过关自测

1. `topk_indices_buffer` 为什么放在 Model 层而不是每层一个？（答：45 层串行执行，工作区可复用，省 44 份显存）
2. MHC 参数存哪几个张量？`hc_attn_fn` 的形状是多少？（答：`hc_*_fn/base/scale` 注意力、FFN 各一套共 6 个；`hc_attn_fn` 是 `(2+4)*4=24` 行 × `4*4096=16384` 列，FP32）
3. `hc_pre` 返回的 4 个值各是什么角色？为什么 `hc_post` 需要 post/comb？（答：layer_input=混合出的子层输入；residual_mhc=4 流快照；post/comb=记录本次混合系数的"账本"——还回去要按同一套系数撒，取和还才配对）
4. 为什么 MTP 层没有 MHC？（答：`is_mtp_layer=True` 明确走非 MHC 分支，草稿层要轻量）

---

**下一课**：[03-MLA注意力.md](03-MLA注意力.md) —— 打开 `AscendGlm5NextMLAAttention`，看 512 维隐向量怎么装下 64 个头的 KV。
