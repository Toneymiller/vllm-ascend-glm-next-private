# 第 05 课：KDA 线性注意力（AscendGlm5NextLinearAttention）

> 对应源码：[`glm5_next.py:1541-1880`](../glm5_next.py#L1541-L1880) + `ops/triton/kda/kda.py` + `ops/kimi_kda_state.py`
> 血统：Kimi K2 的 Gated DeltaNet（GDN）。`layer_types` 中标 `"linear_attention"` 的层走这里。

---

## 0. 先玩一个玩具递推（2×2 状态，手算）

线性注意力的核心是一个"滚动记账"的状态矩阵 S，只有三个动作：**写入 → 打折 → 读出**，全是逐格运算。（k、v、q 都是编出来演示的数；真模型里三者都由当前 token 的 hidden state 经各自投影矩阵得到，角色见 §1。）

### 动作一：写入 k⊗v——按"地址"存内容

外积规则一句话：**结果的第 i 行第 j 列 = k 的第 i 个数 × v 的第 j 个数**，逐格填：

```
k=[1,0], v=[1,2]

逐行算更直观——第 i 行 = k 的第 i 个数 × 整个 v：

k⊗v 的第 1 行 = k[0] × v = 1 × [1, 2] = [1, 2]
k⊗v 的第 2 行 = k[1] × v = 0 × [1, 2] = [0, 0]

拼成 2×2 矩阵：

k⊗v = [[1, 2],
       [0, 0]]
```

**k 是行开关，v 是被存放的内容**：k=[1,0] → 第 1 行"开"，v 原样落进第 1 行；k=[0,1] → 落进第 2 行；若 k=[0.6,0.4] → 按 6:4 拆进两行（真模型的 k 不是 0/1，相当于"模糊地址"）。外积就是"按标签存放"这个动作的数学化。

### 动作二：递推 S ← 0.5×S + k⊗v——旧账打折、新账入册

标量乘矩阵、矩阵加矩阵，都逐格算：

```
token 1（k=[1,0], v=[1,2]）：旧 S 全 0，0.5×0 还是 0，等价于直接写入

  S = k1⊗v1 = [[1, 2],
               [0, 0]]

token 2（k=[0,1], v=[1,0]），分两步：

  先算新账 k2⊗v2（逐行：第 i 行 = k2[i] × v2）：
    第 1 行 = 0 × [1, 0] = [0, 0]
    第 2 行 = 1 × [1, 0] = [1, 0]

    k2⊗v2 = [[0, 0],
             [1, 0]]

  再递推——旧账逐格减半：

    0.5×S = [[0.5×1, 0.5×2],
             [0.5×0, 0.5×0]] = [[0.5, 1],
                                [0,   0]]

  与新账逐格相加：

    S = [[0.5+0, 1+0],
         [0+1,   0+0]] = [[0.5, 1],
                          [1,   0]]
```

0.5×S 每格减半 = **旧记忆淡忘**（没有这步，所有历史永远等权累加；把 0.5 换成按内容算的可学习数，就是 KDA 的"遗忘门"）。

### 动作三：读出 q·S——按匹配度挑行取回

向量乘矩阵 = 对 S 的**行**按 q 加权求和（q 的分量就是每行的系数）：

```
q=[1,1]（token 1 后读）：1×第1行 + 1×第2行 = [1,2]+[0,0] = [1,2]
                          ✓ 原样取回 v1（q·k1 = [1,1]·[1,0] = 1，写入没丢东西）

q=[0,1]（token 2 后读）：0×第1行 + 1×第2行 = [1,0]
                          ✓ 只命中 token 2（q·k1 = 0 匹配不上，q·k2 = 1 完全命中）
```

### 这套动作有什么用

把 S 拆回写入项，读出的含义自动浮现：

```
S   = 0.5·(k1⊗v1) + k2⊗v2
q·S = 0.5·(q·k1)·v1 + (q·k2)·v2
      └── 历史 v 按"q 认不认它的标签 k"加权求和 ──┘
```

**q·S 一步就等于"对所有历史 v 按匹配度加权求和"——这正是注意力的输出定义，只是没有 softmax**。softmax 注意力每步要遍历全部历史 K/V 重新打分；这里的答案是边走边累进 S 的，读时一步取回。S 永远 2×2，处理 100 万 token 也是 2×2（真模型是 128×128）——"线性注意力 O(1) 显存记历史"的全部含义。

KDA = 这个玩具递推 + 三件工程装备：**遗忘门**（衰减不再固定 0.5，按内容学习）、**短卷积**（邻居 token 先局部混合）、**输出门控**（读出时再过一道 sigmoid 闸门）。课文后面逐个看。

## 1. 线性注意力 vs softmax 注意力

softmax 注意力每步要看全部历史（KV cache 随长度线性增长）。线性注意力把"打分"换成"状态递推"：

```
S_t = S_{t-1} · (衰减) + k_t ⊗ v_t      # S 是固定大小的状态矩阵
o_t = q_t · S_t
```

历史被**压缩进一个固定大小的矩阵状态 S**——显存 O(1)，计算 O(1)/token。代价：丢失精确检索能力。所以 GLM5-Next 混合两种层：局部/模糊记忆交给 KDA 层，精确检索交给 DSA 层。

KDA（Kimi Delta Attention）= Gated DeltaNet：在标准线性注意力上加两个门——**遗忘门 g**（控制旧记忆衰减）和 **β**（控制新记忆写入强度），状态更新带"增量修正"语义。

## 2. 模块清单（[:1577-1678](../glm5_next.py#L1577-L1678)）

```
hidden [T, 4096]
   ├─ q_proj / k_proj / v_proj    4096 → 64头×128 = 8192（TP 切头）
   ├─ q_conv1d / k_conv1d / v_conv1d   kernel=4 的短卷积（本地顺序混合）
   ├─ b_proj                      4096 → 64（β 写入门，sigmoid）
   ├─ f_a_proj → f_b_proj         4096→128→8192（低秩瓶颈算遗忘门 g1）
   ├─ g_a_proj → g_b_proj         4096→128→8192（输出门 gate）
   ├─ A_log [1,1,64,1] FP32       遗忘衰减参数（log 域保证正性）
   ├─ dt_bias [8192] FP32         遗忘门偏置
   └─ o_norm（weight + 零 bias）+ o_proj  8192 → 4096
```

结构套路：**QKV 短卷积 + 双低秩门控 + 输出门控 RMSNorm**——与 Mamba2/GDN 同族。

## 3. 两个状态（hybrid 引擎的接口）

```python
class AscendGlm5NextLinearAttention(nn.Module, MambaBase):
    def get_state_shape(self):   # → MambaStateShapeCalculator.kda_state_shape
        ...                      # (conv_state, recurrent_state) 两个形状
    def get_state_dtype(self):   # → kda_state_dtype
    mamba_type = GDN_ATTN        # 用 GDN 注意力后端管理元数据
```

| 状态 | 内容 | 形状（每请求） |
|---|---|---|
| `conv_state` | 最近 `kernel-1 = 3` 个 token 的 QKV 原始值 | [3×3 组, 8192] 级别 |
| `recurrent_state` | 递推矩阵 S | [64 头, 128, 128] |

`self.kv_cache = (conv_state, recurrent_state)`——注意这里 `kv_cache` 名字没变，但存的是循环状态。引擎侧由 `AscendGDNAttentionBackend` + `GDNAttentionMetadata` 管理（`has_initial_state`：chunked prefill 续传时非零初始状态）。

## 4. forward 全流程（[:1707-1878](../glm5_next.py#L1707-L1878)）

### 4.1 门与 β 的计算

```python
q, k, v = q_proj(x), k_proj(x), v_proj(x)
beta = sigmoid(b_proj(x))                    # [T, 64] 每头写入门
g1 = f_b_proj(f_a_proj(x))                   # 遗忘门低秩瓶颈
gate = g_b_proj(g_a_proj(x))                 # 输出门
```

### 4.2 短卷积（[:1760-1811](../glm5_next.py#L1760-L1811)）——注意这段的实现风格

Prefill 路径：**按请求的 Python for 循环**，每请求：
```python
conv_input = cat([conv_state[slot, :, :3], seq_x])      # 状态拼接
res = conv1d(conv_input, w, groups=channels)            # 分组卷积
res = silu(res[..., -seq_len:])                          # 只留新 token 部分
conv_state[slot, :, :3] = conv_input[..., -3:]           # 更新状态
```
Decode 路径：逐 token 循环，一次 conv 出一个位置。

> 观察点（AGENTS.md 的 NPU 准则活案例）：循环里有 `.item()`（[:1768-1779](../glm5_next.py#L1768-L1779)）造成 D2H 同步——正确性优先的原型实现，后续显然该换成融合算子。读代码时能识别这类"性能债"是成熟度标志。

### 4.3 核心递推：prefill 与 decode 两条路

```python
g = fused_kda_gate(g1, self.A_log, head_dim, g_bias=self.dt_bias,
                   safe_gate=self.kda_safe_gate, lower_bound=-5.0)   # :1823
if num_prefills > 0:
    initial_state = recurrent_state[indices]          # 续传请求取非零状态
    core_out, last_state = chunk_kda(q_conv, k_conv, v_conv, g, beta,
        initial_state, output_final_state=True,
        use_qk_l2norm_in_kernel=True, cu_seqlens=...)  # 分块并行（Triton）
    recurrent_state[indices] = last_state
else:  # decode：单步递推
    core_out, _ = fused_recurrent_kda_fwd(..., scale=head_dim**-0.5,
        initial_state=recurrent_state, inplace_final_state=True,
        ssm_state_indices=indices, use_qk_l2norm_in_kernel=True)
```

- `chunk_kda`：prefill 把序列切成 chunk 矩阵化递推（Triton kernel，`ops/triton/kda/kda.py`）
- `fused_recurrent_kda_fwd`：decode 逐步递推，`inplace_final_state=True` 直接原地写回状态省一次拷贝
- `fused_kda_gate` + `safe_gate/lower_bound`：把门压进 `[-5, 0]` 区间再取指数，防溢出
- `use_qk_l2norm_in_kernel=True`：Q/K 在 kernel 内做 L2 归一化（线性注意力稳定性的标准操作）

### 4.4 输出门控（[:1867-1878](../glm5_next.py#L1867-L1878)）

```python
core_out = rms_norm_gated(core_out, gate, o_norm.weight, o_norm.bias,
                          activation="sigmoid", eps=...)
output[:num_actual_tokens] = self.o_proj(core_out)
```

对应第 2 课的 `AscendGlm5NextGatedRMSNormParams`：RMSNorm 的 weight 乘 sigmoid 门。bias 是**非持久化零 buffer**（[:164-168](../glm5_next.py#L164-L168)，注释：匹配 Transformers checkpoint 只存 o_norm.weight 的事实，又不参与严格参数校验）。

## 5. 状态的来源：spec_decode 也算一份

```python
self.num_spec = speculative_config.num_speculative_tokens if ... else 0   # :1574
```

状态 shape 计算把投机解码的额外 token 算进去（MTP draft 也要为每个 draft token 保留 conv 窗口）。hybrid 引擎为每请求分配 `1 + num_spec` 份状态槽。

## 6. 过关自测

1. KDA 层没有 KV cache，那 1M 上下文的请求它在显存上花多少？

   <details><summary>参考答案</summary>

   每请求固定 [64,128,128] 递推状态 + 3 步 conv 状态，与长度无关——这就是混合架构敢上 1M 的另一半底气。

   </details>

2. `has_initial_state` 什么时候为 True？如果错设为 False 会发生什么？

   <details><summary>参考答案</summary>

   chunked prefill 的续传 batch / 投机解码恢复；False 会把之前算好的状态清零，长序列精度崩。

   </details>

3. 画图：一个 token 在 KDA 层内部经过的 6 个投影和 2 个门。

   <details><summary>参考答案</summary>

   6 条投影路 = q / k / v（3 条）+ b_proj（出 β）+ 两条低秩瓶颈 f、g；o_proj 是回 hidden 的出口。2 个门 = **遗忘门 g + 输出门 gate**（对应 §0 三件装备里的两个门）；β 是递推里的第三个控制信号（sigmoid 写入门）。三者分工：**β 管新账写多用力，g 管旧账留多少，gate 管读出放多少出去**。

   ```
   x [T, 4096]
    ├─ q_proj → q_conv1d → L2 norm ─────────────────────┐
    ├─ k_proj → k_conv1d → L2 norm ─────────────────────┤
    ├─ v_proj → v_conv1d ───────────────────────────────┤
    ├─ b_proj → sigmoid = β（写入强度，每头一档）──────────┤
    ├─ f_a→f_b → g1 →(fuse A_log+dt_bias) = 遗忘门 g ────┤
    │                                                    ▼
    │                    核心递推：S ← g⊙S + β·(k⊗v) → core_out [T, 8192]
    │                                                    │
    └─ g_a→g_b → sigmoid = 输出门 gate ─────────→ rms_norm_gated(core_out, gate)
                                                          ↓
                                                o_proj（8192 → 4096）→ 出层
   ```

   </details>

4. `A_log` 为什么存 log 域？

   <details><summary>参考答案</summary>

   遗忘门要经过指数：`g ≈ exp(-exp(A_log) · dt)`。若线性域直接存 A，训练梯度一抖 A 可能变负 → `exp(-负数)` = 衰减变增强（记忆越老越清晰，直接训崩）。存 log 域后 `exp(A_log) > 0` 由构造保证恒正（不靠梯度自觉），衰减永远落在 (0,1)；再配合 `fused_kda_gate` 的 `safe_gate` 把指数压进 [-5,0] 双保险防溢出（§4.3）。这是 Mamba2/GDN 家族的标准做法。顺带：FP32 存储是因为它是全局衰减参数、精度敏感但体积小（每头 1 个标量，[1,1,64,1]）；`dt_bias` 是每通道的衰减偏置，两者在 `fused_kda_gate` 里合成最终 g。

   </details>

---

**下一课**：[06-MLP与MoE.md](06-MLP与MoE.md#L06) —— 最熟悉的陌生人：288 选 8 怎么落地。
