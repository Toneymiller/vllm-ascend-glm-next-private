# GLM5-Next 模型结构教程（零基础友好版）

> 学习对象：`vllm_ascend` 插件中的 `glm5_next` / `glm5_next_multimodal` 模型实现。
> 目标读者：**刚接触 AI infra、没读过模型结构源码**的工程师。
> 学习方式：每课一份 Markdown 课文，先读课文再对照源码（课文里标了行号），每课末尾有自测。卡住就提问，答疑会直接补进对应课文。

---

## 一、这个模型是什么？

GLM5-Next 是一个**混合架构**的大模型，融合了当前业界三条最前沿的技术路线：

| 技术路线 | 来源血统 | 在本仓库的位置 |
|---|---|---|
| MLA 多头潜在注意力 | DeepSeek V2/V3 | `AscendGlm5NextMLAAttention` |
| DSA 稀疏注意力（Lightning Indexer + KPool） | DeepSeek V3.2/V4 | `AscendSparseAttnIndexerKpool` / `AscendGlm5NextIndexer` |
| KDA 线性注意力（Gated DeltaNet） | Kimi K2 / Mamba 系 | `AscendGlm5NextLinearAttention` |
| MoE 混合专家 | DeepSeek / 通用 | `AscendGlm5NextMoE` |
| MHC 超连接（多残差流） | Hyper-Connections 论文 | `DecoderLayer.hc_pre / hc_post` |
| MTP 多 token 预测 | DeepSeek V3 | `glm5_next_mtp.py` |

它不是把上游模型"搬"过来，而是 vllm-ascend 插件为 Ascend NPU **重写**的 NPU 原生实现（自定义算子 + Triton kernel）。

## 二、核心文件地图

```
vllm_ascend/
├── models/
│   ├── glm5_next.py              # 文本模型主体（2532 行）★ 主战场
│   ├── glm5_next_multimodal.py   # 多模态：视觉塔 + ForConditionalGeneration（706 行）
│   ├── glm5_next_mtp.py          # MTP 投机解码模块（363 行）
│   └── learn/                    # ← 本教程目录
├── transformers_utils/
│   ├── configs/glm5_next.py      # 三个配置类（424 行）
│   └── processors/glm5_next.py   # 多模态输入预处理器（635 行）
├── ops/
│   ├── glm5_next_lightning_indexer.py   # Indexer NPU 算子
│   ├── glm5_next_kpool_state_compress.py# KPool 状态压缩算子
│   └── triton/glm5_next_*.py            # Triton kernel 实现
└── patch/platform/patch_glm5_next_*.py  # 调度器/配置补丁
```

## 三、课程表（按顺序，不要跳）

| 课号 | 文件 | 读完你能 | 预计时长 |
|---|---|---|---|
| 00 | `00-第0课-预备知识从零开始.md` | 手算迷你 Transformer、KV cache 公式、TP/EP/PP 术语——**没基础必须先过这课** | 2~3 h |
| 01 | `01-模型全景与配置解析.md` | 读懂 config.json，知道 45 层的异构布局由谁决定 | 1~2 h |
| 02 | `02-模型骨架与MHC超连接.md` | 画出 forward 调用树；讲清 MHC 四残差流怎么分叉合并 | 2 h |
| 03 | `03-MLA注意力.md` | 算清 MLA 的 KV cache 省了多少显存、cache 里存什么 | 2 h |
| 04 | `04-DSA稀疏注意力与KPool.md` | 讲清一条新 token 进来后 Indexer 的完整流水 | 3~4 h（最难） |
| 05 | `05-KDA线性注意力.md` | 区分两种 state，讲清无 KV cache 的注意力怎么工作 | 2~3 h |
| 06 | `06-MLP与MoE.md` | 手算一个 token 的路由过程，读懂 FusedMoE 的参数 | 1 h |
| 07 | `07-多模态结构.md` | 算清一张图变多少个 token，讲清图文怎么拼 | 1.5 h |
| 08 | `08-MTP投机解码.md` | 讲清草稿-验证流程和 eh_proj 的两个来源 | 1 h |
| 09 | `09-权重加载与模型注册.md` | 给任意权重名定位到模块，看懂映射表 | 1.5 h |
| 10 | `10-模型结构总图.md` | **全套结构图**（总览+逐模块放大+cache 家底），随时翻阅 | 随用随查 |

总计约 15~20 小时的认真投入。**节奏建议**：每天 1~2 课，一周到 10 天完成；不求快，求每课末尾自测全对。

## 四、学习方法（第 0 课 §0.8 的浓缩版）

1. **先骨架后血肉**：00→01→02 建立心智图，03-05 逐个击破三大注意力。
2. **形状是第一语言**：每读一个 forward，草稿上标张量形状 `[T, 4096]`。
3. **`named_parameters()` 是地图**：对照 checkpoint 权重名读代码，永远不迷路。
4. **对照上游认亲**：大量模块是 DeepSeek V4 / Kimi GDN 的 NPU 适配版。
5. **单测当文档**：`tests/ut/models/` 下 20+ 个 glm5_next 测试，输入输出写死了。

## 五、毕业标准

1. 能给别人讲清：这个模型为什么要混三种注意力？各自省在哪里？
2. 给一个随机权重名（如 `layers.12.self_attn.indexer.k_cache`），能立刻说出它属于哪个模块、形状多少、谁管理。
3. 能画出一条 token 从 embedding 到 logits 的完整旅程（含 MHC 分叉合并）。
4. 能回答"1M 上下文的 KV cache 显存怎么构成"这类容量规划问题。
