# 第 03 课：MLA 注意力（AscendGlm5NextMLAAttention）

> 对应源码：[`vllm_ascend/models/glm5_next.py:1881-2052`](../glm5_next.py#L1881-L2052) + `vllm_ascend/ops/indexer_kpool_mla.py`
> 本课回答一个核心问题：**KV cache 里到底存了什么，省了多少？**

---

## 0. 新手包：三个先修概念

**① 什么是"低秩压缩"（LoRA 里的 Lo 就是它）**。想做一个 4096→32768 的大投影，有两条路：

```
直接做：  W [32768, 4096]           参数 = 1.34 亿
绕一下：  4096 → 512 → 32768        4096×512 + 512×32768 = 0.19 亿
          （先压到 512 维"摘要"，再从摘要展开）
```

绕路的矩阵连乘在数学上表达能力弱一些，但参数少 7 倍。**赌的是"关键信息其实集中在低维子空间"**——训练数据证明这个赌注赢了。

**② MLA 省的是两样东西，别混为一谈**：

| | 标准 MHA（64 头×256 维 K+V） | MLA |
|---|---|---|
| 每层每 token 存的 cache | 64×(256+256) = **32768 个数** = 64 KB | 512 个数 = **1 KB** |
| 45 层 × 1M token 总显存 | ≈ **2.9 TB**（放不下） | ≈ **45 GB**（可行） |
| 投影参数量 | 4096×32768×2（K、V 各一） | 4096×512 + 512×32768（先压后展） |

前两行是**省显存**（cache 只存 512 维隐向量），第三行是省参数。第 0 课 §0.3 公式里的"KV 头数×头维数"这一项被 MLA 从 32768 砍到 512。

**③ "权重吸收"预告**：MLA 的数学魔法是"从 c_KV 重建 K/V 的矩阵"可以**挪到 Q 那一侧合并掉**，于是推理时根本不需要显式重建。数学上就是矩阵乘法结合律（单头视角，W_k/W_v 是 kv_b_proj 里重建 K、V 的两半）：

```
score = q·k = q·(W_k·c_KV) = (W_k^T·q)·c_KV
      ├ 朴素：每个历史 token 都要重建 k（对照上表 2.9 TB 的来源）
      └ 吸收：只对新 token 的 q 算一次 q' = W_k^T·q（512 维），直接和 cache 里的 c_KV 内积
```

V 侧同理：`out = Σ a_h·(W_v·c_KV) = (Σ a_h·W_v)·c_KV`，W_v 吸收到输出侧。**历史 K/V 从不持久化**：cache 里永远只有 512 维 latent，decode 也从不重建历史。V 的"物化"只发生在**当前 token 的输出侧**——每步 64 头各一次 `W_UV·u`（64×256 个数），O(当前) 而非 O(历史)，这才是省显存的点。层间流动、最终喂给 LM head 的也始终是 4096 维 hidden state，latent 只是注意力"记忆"的形态。注意 prefill 是例外：为了复用高效 MHA kernel，会**临时**把 latent 重建成 K/V 跑标准注意力（[mla_v1.py:1214](../../attention/mla_v1.py#L1214)），用完即弃；写入 cache 的仍是 latent（[mla_v1.py:1199](../../attention/mla_v1.py#L1199)）。仓库里真这么写（MLA 后端加载后把 kv_b_proj 替换成吸收版权重）：

- [mla_v1.py:943-955](../../attention/mla_v1.py#L943-L955)：kv_b_proj 权重 → `W_UK_T`（[64,256,512]，Q 侧用）/ `W_UV`（[64,512,256]，输出侧用）
- [mla_v1.py:918](../../attention/mla_v1.py#L918)：`ql_nope = torch.bmm(q_nope, W_UK_T)` ← Q 从 256 维投进 512 维 latent
- [mla_v1.py:898-905](../../attention/mla_v1.py#L898-L905)：`_v_up_proj` 用 W_UV 从 latent 变回 V

**训练时也是吸收吗？** 不是。训练走**不吸收的显式形式**（kv_b_proj 正常展开物化 K/V）——训练算力充裕，显式形式易实现易并行；结合律保证两种形式的前向、梯度完全等价。所以 checkpoint 存的是"原始"矩阵，推理引擎加载后才现场变换出 `W_UK_T`/`W_UV`（`process_weights_after_loading`，[mla_v1.py:922](../../attention/mla_v1.py#L922)）——**训练形态 ≠ 推理形态，权重是同一份**。MLA 权重和 MHA 训出来的不一样，那是**低秩结构**的功劳（64 头被迫共享一张 512 维 c_KV"公共摘要卡"、从零训练），与吸收无关——吸收只是换了乘法顺序，不产生任何知识差异。

**为什么 GLM5-Next 能"全吸收"**：吸收要求矩阵固定（与位置无关）。RoPE 的旋转矩阵逐 token 不同，没法并进去——DeepSeek 得把 64 维 rope 部分单独缓存、绕开吸收；GLM5-Next `qk_rope_head_dim=0` 纯 nope，整条链路都可吸收。细节第 4 课打分时用到，本课先记住有这回事。

## 1. 先懂标准 MLA（DeepSeek 式）

普通 MHA 的 KV cache：每 token 存 `2 × n_heads × head_dim` 个数。64 头 × 128 维 × 2 = 16K 个数/token/层，1M 上下文下天文数字。

MLA（Multi-head Latent Attention）的思路：**把 KV 压缩成一个低秩隐向量 c_KV**，每个 token 只存 `c_KV`（kv_lora_rank=512 维）。需要时用两个上投影矩阵从 c_KV 重建 K 和 V。K/V 矩阵可以被吸收进 Q 侧投影（权重吸收），数学等价但显存降一个数量级。

GLM5-Next 的 MLA 参数（第 1 课看过）：

```
q_lora_rank=1536, kv_lora_rank=512
qk_nope_head_dim=256, qk_rope_head_dim=0, v_head_dim=256
mla_nope=True   →   完全没有 RoPE！
```

**与 DeepSeek MLA 的关键差异**：DeepSeek 的 Q/K 各带 64 维 RoPE 部分（RoPE 无法被权重吸收，所以单独缓存），而这里 `qk_rope_head_dim=0`——**纯 nope 变体**，整个 QK 都是可吸收的低秩形式。位置信息去哪了？第 04 课的 Indexer 和 DSA 层会给出答案。

## 2. 类结构：薄壳 + 引擎

```python
class AscendGlm5NextMLAAttention(nn.Module):        # :1881 薄壳
    def __init__(...):
        ... 建 5 个投影 + 2 个 Layernorm + indexer ...
        self.indexer = AscendGlm5NextIndexer(...)          # 第 04 课
        self.indexer_kpool_mla_attention = AscendIndexerKPoolMLAAttention(
            ..., modules=IndexerKPoolMLAModules(...))      # 真正的注意力引擎

    def forward(self, hidden_states, positions, output):   # :2043
        result = self.indexer_kpool_mla_attention(positions, hidden_states)
        output[:] = result                                 # 写入式接口
```

薄壳只负责建模块；计算全部委托给 `AscendIndexerKPoolMLAAttention`（`vllm_ascend/ops/indexer_kpool_mla.py`），后者是"注意力后端 + 组件集合"：通过 `IndexerKPoolMLAModules` 把 8 个子模块打包传入，避免重复构建。

## 3. 五个投影逐个看（[:1916-1963](../glm5_next.py#L1916-L1963)）

```
hidden [T, 4096]
   │
   ├─ fused_qkv_a_proj（MergedColumnParallel，TP 禁用=复制权重）
   │    输入 4096 → 输出 [1536 | 512] 拼在一起
   │    前半 = q_a（Q 的降维），后半 = kv_a（KV 压缩到 c_KV）
   │    ★ checkpoint 里是两个独立矩阵 q_a_proj + kv_a_proj_with_mqa，加载时拼装
   │
   ├─ q_a_layernorm(1536) → q_b_proj(1536 → 64×256)   # Q 升维：每头 256 维 nope
   │
   ├─ kv_a_layernorm(512) → kv_b_proj(512 → 64×(256+256))
   │    # c_KV 升维：每头 k_nope 256 维 + v 256 维
   │
   └─ o_proj(64×256 → 4096)                            # RowParallel，跨 TP 归约
```

形状速记（TP=8 时每 rank 8 个头）：

| 权重 | 形状 | 说明 |
|---|---|---|
| `fused_qkv_a_proj.weight` | [2048, 4096] | 1536+512 行 |
| `q_b_proj.weight` | [64×256, 1536] | TP 切行 |
| `kv_b_proj.weight` | [64×512, 512] | TP 切行 |
| `o_proj.weight` | [4096, 64×256] | TP 切列 |

**为什么 fused_qkv_a_proj 要 `disable_tp=True`**：A 投影是所有头共享的"入口"，按 token 计算量小，复制一份比通信省。

## 4. KV cache 存什么

`AscendGlm5NextIndexerKPoolCache.get_kv_cache_spec`（[:295](../glm5_next.py#L295)）告诉引擎：

```python
MLAAttentionSpec(block_size, num_kv_heads=1, head_size=kv_lora_rank=512, ...)
```

即每个 DSA 层每 token 缓存 **512 个数（c_KV 隐向量）**。对比 MHA：64 头 × (256+256) = 32K 个数。**压缩比 ~64×**。这就是"1M 上下文"能落地的第一支柱。

第二支柱是 top-k 稀疏化（第 04 课）：算注意力时也不是全量 c_KV 都参与，只挑 top-k 个 token。

## 5. RoPE 的两处"缺席"

[:1965](../glm5_next.py#L1965) `self.rotary_emb = None`（因为 `qk_rope_head_dim=0`）：主注意力不转。
[:1998](../glm5_next.py#L1998) `self.indexer_rotary_emb = None`：Indexer 也不转。
[:1986](../glm5_next.py#L1986) 处留有完整代码路径（`qk_rope_head_dim>0` 时会建 `get_rope(..., is_neox_style=...)`）——说明这套代码**同时服务带 RoPE 和不带 RoPE 的 checkpoint**，GLM5-Next 只是恰好全关了。

> 思考：没有 RoPE，注意力对位置不敏感，位置信息只能靠两个来源——①第 0 层输入 embedding 本身带位置感（训练学到的），②KDA 层的短卷积（局部顺序敏感）。这是架构层面的赌注。

## 6. `_pad_nope_kv_a_weight`：权重加载的小戏法（[:236](../glm5_next.py#L236)）

```python
def _pad_nope_kv_a_weight(config, name, loaded_weight):
    """Pad the checkpoint's latent-only KV projection for NoPE models."""
    if mla_nope 且 rope_dim<=0 且 name.endswith(".kv_a_proj_with_mqa.weight"):
        在权重底部补 rope_dim 行全零 → [kv_lora_rank + rope_dim, hidden]
```

checkpoint 存的是 `[512, 4096]`（纯 nope），而引擎的统一 kernel 按 `[512+64, 4096]` 分配。加载时补零行对齐——零权重乘任何输入都是零，数学无损。第 09 课会再遇到它。

## 7. 过关自测

1. 画出 `fused_qkv_a_proj` 输出列的切分图（哪 1536 列是 Q 路、哪 512 列是 KV 路）。（答：输出 `[T, 2048] = [ q_a 前 1536 列 | kv_a 后 512 列 ]`——权重按行拼装 `[q_a_proj(1536 行); kv_a_proj(512 行)]`，行序即输出列序。前 1536 列走 `q_a_layernorm → q_b_proj` 升到 64×256 的 Q（1536 只是 Q 的中间站）；后 512 列经 `kv_a_layernorm` 就是 c_KV，直接写 cache（512 是 KV 的终点）。记忆点：Q 降维再升维，KV 一步压到位）
2. 一个 1M token 请求，**单个 DSA 层**的 KV cache 显存 ≈？（答：1M × 512 × dtype_size ≈ 1 GB @ BF16；45 层就是 45 GB——所以还需要第 04 课的稀疏化）
3. `kv_b_proj` 的输出为什么是 `64×(256+256)` 而不是 `64×256`？（答：注意力下游要从 c_KV 重建**两样**东西——K（每头 256，打分用）和 V（每头 256，出值用），只给 64×256 就只够一半，所以一个矩阵一次产齐 64×(256+256)=64×512 维。这 32768 个输出列加载后被劈成两半：K 半 → `W_UK_T` 吸收进 Q 侧，V 半 → `W_UV` 留在输出侧（[mla_v1.py:943](../../attention/mla_v1.py#L943) 的 `split`）。这也是 §3 表格"省参数"的来源——K、V 共享同一个 512 维低秩 latent，不像 MHA 各自独立投影）

---

**下一课**：[04-DSA稀疏注意力与KPool.md](04-DSA稀疏注意力与KPool.md#L04) —— 本仓库最硬核的 900 行：Indexer 怎么给 1M 个 key 打分、KPool 怎么把 16 个 token 压成 1 个。
