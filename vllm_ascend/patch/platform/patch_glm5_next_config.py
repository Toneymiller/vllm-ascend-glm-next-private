from transformers import AutoConfig
from vllm.transformers_utils.model_arch_config_convertor import (
    MODEL_ARCH_CONFIG_CONVERTORS,
    ModelArchConfigConvertorBase,
)

from vllm_ascend.transformers_utils.configs.glm5_next import (
    Glm5NextConfig,
    Glm5NextTextConfig,
    Glm5NextVisionConfig,
)


class Glm5NextModelArchConfigConvertor(ModelArchConfigConvertorBase):
    """让 vLLM 0.23.0 按 MLA 语义解析 GLM5-Next 的模型结构。"""

    def is_deepseek_mla(self) -> bool:
        # 新版 vLLM 已把 glm5_next/glm5_next_text 加入 MLA 白名单；
        # 0.23.0 尚未包含它们，因此需要在插件侧补齐相同语义。
        return getattr(self.hf_text_config, "kv_lora_rank", None) is not None

AutoConfig.register("glm5_next", Glm5NextConfig, exist_ok=True)
AutoConfig.register("glm5_next_text", Glm5NextTextConfig, exist_ok=True)
AutoConfig.register("glm5_next_vision", Glm5NextVisionConfig, exist_ok=True)

# ModelArchConfigConvertor 在 ModelConfig 构造阶段决定 use_mla、head_size
# 和每卡 KV head 数。注册专用 convertor 后，GLM5-Next 使用
# (kv_lora_rank + qk_rope_head_dim) 作为 MLA cache head_size，且 KV heads=1。
MODEL_ARCH_CONFIG_CONVERTORS["glm5_next"] = Glm5NextModelArchConfigConvertor
MODEL_ARCH_CONFIG_CONVERTORS["glm5_next_text"] = Glm5NextModelArchConfigConvertor
