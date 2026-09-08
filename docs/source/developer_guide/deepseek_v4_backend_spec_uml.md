# DeepSeek V4 自定义 Backend 与 Spec Decode 架构图

本文基于以下两份当前工作区源码整理：

- vLLM：`E:\projects\GLM-Next\vllm`
- vLLM Ascend：当前仓库

这里的 “backend” 容易混淆，实际至少有三层：

1. **模型实现 backend**：同一个 `DeepseekV4ForCausalLM` 在 NVIDIA、ROCm、XPU、Ascend 上对应不同模型类。
2. **Attention backend**：定义 KV-cache 布局、metadata builder、kernel 能力以及 attention 执行。
3. **Compilation backend**：Inductor、eager 等编译后端，不是本文重点。

## 1. DeepSeek V4 模型与 Attention Backend 类图

```mermaid
classDiagram
direction LR

class ModelRegistry {
  +register_model(architecture, class_path)
  +load_model_cls(architecture)
}

class DeepseekV4EntryPoint {
  <<vllm.models.deepseek_v4>>
  +DeepseekV4ForCausalLM
  +DeepSeekV4MTP
}

class DeepseekV4ForCausalLM {
  +model: DeepseekV4Model
  +lm_head
  +forward()
  +compute_logits()
  +get_mtp_target_hidden_states()
  +load_weights()
}

class DeepseekV4Model {
  +layers
  +topk_indices_buffer
  +_mtp_hidden_buffer
  +forward()
}

class DeepseekV4DecoderLayer {
  +attn: DeepseekV4Attention
  +ffn: DeepseekV4MoE
  +forward()
}

class DeepseekV4Attention {
  <<abstract>>
  +backend_cls
  +swa_cache_layer
  +compressor
  +indexer
  +forward()
  +attention_impl()
  +get_attn_backend()
  +get_kv_cache_spec()
  +forward_mqa()*
  +_o_proj()*
}

class DeepseekV4FlashMLAAttention {
  +backend_cls = DeepseekV4FlashMLABackend
  +forward_mqa()
  +_o_proj()
}

class DeepseekV4FlashInferMLAAttention {
  +backend_cls = DeepseekV4FlashInferMLASparseBackend
  +forward_mqa()
  +_o_proj()
}

class DeepseekV4ROCMAiterMLAAttention {
  +backend_cls = DeepseekV4ROCMAiterMLASparseBackend
  +forward_mqa()
  +_o_proj()
}

class DeepseekV4XPUAttention {
  +backend_cls = DeepseekV4XPUSparseBackend
  +forward_mqa()
  +_o_proj()
}

class AttentionLayerBase {
  <<interface>>
  +get_attn_backend()
  +get_kv_cache_spec()
}

class AttentionBackend {
  <<abstract>>
  +get_name()*
  +get_builder_cls()*
  +get_impl_cls()*
  +get_kv_cache_shape()*
  +validate_configuration()
}

class DeepseekV4FlashMLABackend {
  +get_builder_cls()
  +get_kv_cache_shape()
  +is_mla()
  +is_sparse()
}

class DeepseekV4FlashInferMLASparseBackend
class DeepseekV4ROCMAiterMLASparseBackend
class DeepseekV4XPUSparseBackend

class AttentionMetadataBuilder {
  <<abstract>>
  +build(common_metadata)*
}

class DeepseekV4FlashMLAMetadataBuilder {
  +build()
  +_build_c128a_metadata()
}

class DeepseekV4FlashMLAMetadata {
  +block_table
  +decode_topk_indices
  +decode_topk_lens
}

ModelRegistry ..> DeepseekV4EntryPoint : architecture lookup
DeepseekV4EntryPoint ..> DeepseekV4ForCausalLM : platform export
DeepseekV4ForCausalLM *-- DeepseekV4Model
DeepseekV4Model *-- "N" DeepseekV4DecoderLayer
DeepseekV4DecoderLayer *-- DeepseekV4Attention
DeepseekV4DecoderLayer *-- DeepseekV4MoE

DeepseekV4Attention ..|> AttentionLayerBase
DeepseekV4FlashMLAAttention --|> DeepseekV4Attention
DeepseekV4FlashInferMLAAttention --|> DeepseekV4Attention
DeepseekV4ROCMAiterMLAAttention --|> DeepseekV4Attention
DeepseekV4XPUAttention --|> DeepseekV4Attention

DeepseekV4FlashMLABackend --|> AttentionBackend
DeepseekV4FlashInferMLASparseBackend --|> DeepseekV4FlashMLABackend
DeepseekV4ROCMAiterMLASparseBackend --|> DeepseekV4FlashMLABackend
DeepseekV4XPUSparseBackend --|> DeepseekV4FlashMLABackend
DeepseekV4FlashMLABackend ..> DeepseekV4FlashMLAMetadataBuilder
DeepseekV4FlashMLAMetadataBuilder --|> AttentionMetadataBuilder
DeepseekV4FlashMLAMetadataBuilder ..> DeepseekV4FlashMLAMetadata : creates
```

### 关键解读

DeepSeek V4 是 **model-driven attention backend**：

- `DeepseekV4Attention.forward()` 自己完成 Q/KV 投影、RMSNorm、RoPE、cache insert、Indexer、Compressor。
- 平台子类通过 `forward_mqa()` 直接调用 FlashMLA、FlashInfer、AITER 或 XPU kernel。
- `DeepseekV4FlashMLABackend.get_impl_cls()` 故意不提供独立 `AttentionImpl`；backend 主要承担 metadata 和 KV-cache 契约。
- `_select_dsv4_attn_cls()` 根据 `attention_config.backend` 与设备 capability 选择的不是普通 `AttentionImpl`，而是整个 `DeepseekV4Attention` 子类。

因此，仅实现一个通用 `AttentionBackend` 并注册为 `CUSTOM`，**不足以**让 DeepSeek V4 使用它；还必须让 DeepSeek V4 的模型层选中与它配套的 `DeepseekV4Attention` 子类。

## 2. Ascend 插件覆盖关系

```mermaid
classDiagram
direction LR

class ModelRegistry
class AscendDeepseekV4ForCausalLM {
  +model: DeepseekV4Model
  +forward()
  +compute_logits()
  +get_mtp_target_hidden_states()
}
class DeepseekV4Model {
  <<vllm_ascend.models>>
}
class DeepseekV4Attention {
  <<vllm_ascend.models>>
  +dsa_attn: AscendDeepseekSparseAttention
  +forward()
}
class AscendDeepseekSparseAttention
class DSAAttention {
  +attn_backend = AscendDSABackend
  +impl: DSAAttentionImpl
  +get_attn_backend()
  +get_kv_cache_spec()
}
class AscendDSABackend {
  +get_builder_cls()
  +get_impl_cls()
  +get_kv_cache_shape()
}
class DSAAttentionImpl {
  +forward()
}
class AscendDeepseekV4IndexerCache
class AscendDeepseekV4SWACache
class AscendCompressorStateCache

ModelRegistry ..> AscendDeepseekV4ForCausalLM : plugin override
AscendDeepseekV4ForCausalLM *-- DeepseekV4Model
DeepseekV4Model *-- DeepseekV4Attention
DeepseekV4Attention *-- AscendDeepseekSparseAttention
AscendDeepseekSparseAttention *-- DSAAttention
DSAAttention ..> AscendDSABackend
DSAAttention *-- DSAAttentionImpl
AscendDSABackend ..> DSAAttentionImpl : get_impl_cls
DeepseekV4Model *-- AscendDeepseekV4IndexerCache
DeepseekV4Model *-- AscendDeepseekV4SWACache
DeepseekV4Model *-- AscendCompressorStateCache
```

Ascend 当前没有走上游 `vllm.models.deepseek_v4.__init__` 的硬件分支，而是通过：

```text
ModelRegistry.register_model(
    "DeepseekV4ForCausalLM",
    "vllm_ascend.models.deepseek_v4:AscendDeepseekV4ForCausalLM",
)
```

覆盖模型入口。这个方式比修改上游 `deepseek_v4/__init__.py` 更符合硬件插件边界。

## 3. Backend 初始化与单次 Forward 时序图

```mermaid
sequenceDiagram
autonumber
actor User
participant Config as VllmConfig
participant Registry as ModelRegistry
participant Loader as ModelLoader
participant Model as DeepseekV4ForCausalLM
participant Layer as DeepseekV4DecoderLayer
participant Attn as DeepseekV4Attention subclass
participant KV as KV Cache Planner
participant Builder as MetadataBuilder
participant Runner as GPU/NPU ModelRunner
participant Kernel as Platform Kernel

User->>Config: attention_backend / cache_dtype / block_size
Config->>Registry: architecture = DeepseekV4ForCausalLM
Registry-->>Loader: platform model class
Loader->>Model: construct(vllm_config)
Model->>Layer: construct N decoder layers
Layer->>Layer: select attention class
Note over Layer: CUDA 使用 _select_dsv4_attn_cls()<br/>Ascend 由插件模型直接装配 DSA
Layer->>Attn: construct backend-specific layer
Attn->>Config: register static_forward_context[prefix]

KV->>Attn: get_kv_cache_spec()
Attn-->>KV: MLA + SWA + Indexer/Compressor specs
KV->>Attn: get_attn_backend()
Attn-->>KV: backend_cls
KV->>Builder: backend.get_builder_cls()
Builder-->>KV: per-group metadata builder
KV->>KV: allocate and bind KV-cache tensors

Runner->>Builder: build(CommonAttentionMetadata)
Builder-->>Runner: layer-keyed attention metadata
Runner->>Runner: set_forward_context(metadata, slot_mapping)
Runner->>Model: forward(input_ids, positions)
Model->>Layer: forward(hidden_states)
Layer->>Attn: forward(positions, hidden_states)
Attn->>Attn: projections + norm + RoPE + cache insert
Attn->>Attn: Indexer + Compressor
Attn->>Kernel: forward_mqa(q, cache, metadata)
Kernel-->>Attn: sparse MLA output
Attn->>Attn: inverse RoPE + output projection
Attn-->>Layer: attention output
Layer-->>Model: hidden states
Model-->>Runner: final hidden states
```

### 自定义 Attention Backend 的最小闭环

| 层次 | 必须实现或修改 | DeepSeek V4 特别要求 |
|---|---|---|
| Backend 契约 | `AttentionBackend` 子类 | 校验 dtype、head size、block size、capability；定义 KV shape |
| Metadata | `AttentionMetadata` 与 `AttentionMetadataBuilder` | 同时覆盖 prefill、decode、MTP 的 `query_len > 1` |
| 模型 attention | `DeepseekV4Attention` 平台子类 | 实现 `forward_mqa()`、`_o_proj()`、head padding 和 `backend_cls` |
| 选择逻辑 | `_select_dsv4_attn_cls()` 或插件模型装配 | 只注册 `AttentionBackendEnum.CUSTOM` 不会自动进入 DeepSeek V4 |
| KV-cache | `get_kv_cache_spec()` 与 backend `get_kv_cache_shape()` | MLA、SWA、Indexer、Compressor 的 block/stride/dtype 必须一致 |
| 图执行 | static forward context、metadata 稳定地址 | 验证 eager、piecewise/full graph 及 dummy/profile |

Ascend 上推荐沿用当前插件模式：注册 Ascend 模型类，让模型内部装配 `AscendDSABackend` 和 `DSAAttentionImpl`，不要在上游 NVIDIA/ROCm/XPU 入口中追加 Ascend 分支。

## 4. Spec Decode / MTP 类图

```mermaid
classDiagram
direction LR

class SpeculativeConfig {
  +method
  +model
  +num_speculative_tokens
  +draft_model_config
  +attention_backend
  +hf_config_override()
}

class GPUModelRunnerV2 {
  +model
  +speculator
  +rejection_sampler
  +execute_model()
  +sample_tokens()
}

class BaseSpeculator {
  <<abstract>>
  +load_model()
  +set_attn()
  +propose()*
  +capture()*
}

class DraftModelSpeculator {
  +model
  +draft_logits
  +sample_draft()
  +load_draft_model()*
}

class AutoRegressiveSpeculator {
  +propose()
  +_prefill()
  +_generate_draft()
}

class MTPSpeculator {
  +load_draft_model()
}

class EagleSpeculator
class AscendEagleSpeculator {
  +propose()
  +set_attn()
  +_generate_draft()
  +_run_model()
}

class DeepSeekV4MTP {
  +model: DeepSeekV4MultiTokenPredictor
  +forward()
  +compute_logits()
  +load_weights()
}

class DeepSeekV4MultiTokenPredictor {
  +layers
  +embed_tokens
  +forward(spec_step_idx)
  +compute_logits(spec_step_idx)
}

class DeepSeekV4MultiTokenPredictorLayer {
  +e_proj
  +h_proj
  +mtp_block
  +shared_head
  +forward()
}

class DeepseekV4DecoderLayer
class RejectionSampler {
  +forward(target_logits, draft_tokens, draft_logits)
}

SpeculativeConfig ..> GPUModelRunnerV2
GPUModelRunnerV2 *-- BaseSpeculator
GPUModelRunnerV2 *-- RejectionSampler
DraftModelSpeculator --|> BaseSpeculator
AutoRegressiveSpeculator --|> DraftModelSpeculator
MTPSpeculator --|> AutoRegressiveSpeculator
EagleSpeculator --|> AutoRegressiveSpeculator
AscendEagleSpeculator --|> EagleSpeculator
MTPSpeculator ..> DeepSeekV4MTP : load_eagle_model
AscendEagleSpeculator ..> DeepSeekV4MTP : registry lookup
DeepSeekV4MTP *-- DeepSeekV4MultiTokenPredictor
DeepSeekV4MultiTokenPredictor *-- DeepSeekV4MultiTokenPredictorLayer
DeepSeekV4MultiTokenPredictorLayer *-- DeepseekV4DecoderLayer : reuses full block
```

DeepSeek V4 MTP 的 draft model 不是一个缩小版完整 LLM，而是 checkpoint 内的 MTP 层：

- `SpeculativeConfig.hf_config_override()` 把 `deepseek_v4` 改写成 `deepseek_mtp`。
- architecture 改写为 `DeepSeekV4MTPModel`，再由 Model Registry 加载 MTP 类。
- MTP layer 复用一个完整 `DeepseekV4DecoderLayer`，并额外包含 embedding projection、hidden projection 与 shared head。
- target model 在 forward 时把 **pre-hc_head residual** 写入 `_mtp_hidden_buffer`。
- runner 通过 `get_mtp_target_hidden_states()` 把这个特殊 hidden state 交给 drafter。
- `load_eagle_model()` 还会共享 embedding、LM head 与 `topk_indices_buffer`。

## 5. 一轮 MTP 推测解码时序图

```mermaid
sequenceDiagram
autonumber
actor Scheduler
participant Runner as ModelRunner V2
participant Target as DeepseekV4ForCausalLM
participant Reject as RejectionSampler
participant Spec as MTP/Eagle Speculator
participant Draft as DeepSeekV4MTP
participant State as RequestState

Scheduler->>Runner: schedule target tokens plus prior draft tokens
Runner->>Target: forward(input_ids, positions)
Target->>Target: run decoder layers
Target->>Target: save pre-hc_head residual to _mtp_hidden_buffer
Target-->>Runner: target hidden states

alt previous step already has draft tokens
  Runner->>Target: compute_logits for verification positions
  Runner->>Reject: verify target logits against draft tokens
  Reject-->>Runner: accepted tokens + recovery token + rejected count
else first step or speculation skipped
  Runner->>Target: compute_logits
  Runner->>Runner: normal sample
end

Runner->>State: postprocess accepted/sampled tokens
Runner->>Target: get_mtp_target_hidden_states()
Target-->>Runner: pre-hc_head residual
Runner->>Spec: propose(input batch, metadata, residual, sample state)

loop spec_step_idx = 0..K-1
  Spec->>Draft: forward(token, position, previous_hidden, spec_step_idx)
  Draft->>Draft: embedding projection + hidden projection
  Draft->>Draft: DeepseekV4DecoderLayer forward
  Draft-->>Spec: next pre-hc hidden state
  Spec->>Draft: compute_logits(hidden, spec_step_idx)
  Draft-->>Spec: draft logits
  Spec->>Spec: greedy or probabilistic draft sample
end

Spec-->>Runner: K draft token ids
Runner->>State: store draft tokens
Runner-->>Scheduler: accepted output and next-step drafts
Note over Scheduler,Reject: 下一轮 target 一次 forward 并行验证这些 draft token
```

## 6. 如何自定义 Spec

### 路线 A：自定义训练型 MTP/EAGLE（适合 DeepSeek V4）

优先复用 V2 的 `DraftModelSpeculator` / `AutoRegressiveSpeculator`：

1. 在 `SpeculativeConfig` 中识别 method，并生成 `draft_model_config`。
2. 为 draft architecture 注册模型类。
3. draft model 至少实现 `forward()`、`compute_logits()`、`load_weights()`；需要时实现独立 embedding/LM head。
4. 如果 target 交给 draft 的不是最终 hidden state，像 DeepSeek V4 一样提供显式 hook，例如 `get_mtp_target_hidden_states()`。
5. 若生成方式仍是逐 token 自回归，只需复用或继承 `MTPSpeculator`；如果 position、KV 更新、并行 drafting 或 tree layout 不同，再派生新的 speculator。
6. 在 `init_speculator()` 中注册 method 到 speculator 的映射。
7. Ascend 还要同步扩展 `vllm_ascend.worker.v2.spec_decode...init_speculator()`，并处理 NPU metadata 与 ACLGraph。

### 路线 B：无模型或任意算法 proposer

当前源码的 `method="custom_class"` 只完整接到了旧 V1 runner：

```text
constructor:
    MyProposer(vllm_config)

required method:
    propose(
        sampled_token_ids,
        num_tokens_no_spec,
        token_ids_cpu,
        slot_mappings=...,
    )
```

V2 的 `init_speculator()` 没有 `custom_class` 分支，而且要求的是更严格的 `BaseSpeculator` 生命周期：

- `propose()`
- `init_cudagraph_manager()`
- `capture()`
- draft model 场景还包括 `load_model()` 与 `set_attn()`

所以面向 DeepSeek V4 / Ascend V2 时，不应把 V1 `custom_class` 当成最终扩展接口。应实现一个 `BaseSpeculator` 子类，并同时接入：

```text
vllm/v1/worker/gpu/spec_decode/__init__.py
vllm_ascend/worker/v2/spec_decode/eagle/__init__.py
```

无论 proposer 如何产生 draft token，最终分布正确性仍由 target logits 和 `RejectionSampler` 保证；自定义 proposer 不应绕过 target verification。

## 7. 扩展点选择

```mermaid
flowchart TD
    A[我要自定义什么] --> B{只是换 attention kernel?}
    B -->|是| C[实现 Backend + MetadataBuilder]
    C --> D[实现 DeepseekV4Attention 平台子类]
    D --> E[接入模型层选择或插件模型装配]

    B -->|否| F{新增硬件平台模型实现?}
    F -->|是| G[插件 ModelRegistry 覆盖]
    G --> H[实现平台 Model Attention MoE Weight Loader]

    F -->|否| I{自定义 speculative 算法?}
    I -->|训练型 MTP/EAGLE| J[注册 draft architecture]
    J --> K[复用或派生 AutoRegressiveSpeculator]
    I -->|无模型 proposer| L{使用哪个 runner?}
    L -->|V1| M[custom_class proposer]
    L -->|V2 / Ascend| N[BaseSpeculator 子类 + factory 接线]
```

## 8. 重点源码索引

### vLLM

- 模型注册：`vllm/model_executor/models/registry.py`
- 平台入口：`vllm/models/deepseek_v4/__init__.py`
- DeepSeek V4 attention 基类：`vllm/models/deepseek_v4/attention.py`
- CUDA attention 类选择：`vllm/models/deepseek_v4/nvidia/model.py`
- FlashMLA backend 与 metadata：`vllm/models/deepseek_v4/sparse_mla.py`
- FlashMLA 执行层：`vllm/models/deepseek_v4/nvidia/flashmla.py`
- FlashInfer 执行层：`vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`
- Attention backend 抽象：`vllm/v1/attention/backend.py`
- Backend registry：`vllm/v1/attention/backends/registry.py`
- Metadata 初始化：`vllm/v1/worker/gpu/attn_utils.py`
- Spec 配置与 DeepSeek V4 MTP config 改写：`vllm/config/speculative.py`
- V2 speculator factory：`vllm/v1/worker/gpu/spec_decode/__init__.py`
- V2 speculator 抽象：`vllm/v1/worker/gpu/spec_decode/speculator.py`
- 自回归 proposer：`vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py`
- MTP speculator：`vllm/v1/worker/gpu/spec_decode/mtp/speculator.py`
- DeepSeek V4 MTP model：`vllm/models/deepseek_v4/nvidia/mtp.py`
- Target/MTP 权重与 buffer 共享：`vllm/v1/worker/gpu/spec_decode/eagle/utils.py`
- V2 target forward、verification、propose 主链：`vllm/v1/worker/gpu/model_runner.py`
- V1 custom proposer：`vllm/v1/spec_decode/custom_class_proposer.py`

### vLLM Ascend

- 模型覆盖注册：`vllm_ascend/models/__init__.py`
- Ascend DeepSeek V4：`vllm_ascend/models/deepseek_v4.py`
- Ascend DeepSeek V4 MTP：`vllm_ascend/models/deepseek_v4_mtp.py`
- Ascend DSA attention layer：`vllm_ascend/models/layer/attention/layer.py`
- Ascend V2 Model Runner：`vllm_ascend/worker/v2/model_runner.py`
- Ascend speculator factory：`vllm_ascend/worker/v2/spec_decode/eagle/__init__.py`
- Ascend speculator：`vllm_ascend/worker/v2/spec_decode/eagle/speculator.py`
