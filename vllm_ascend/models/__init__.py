from vllm import ModelRegistry


def register_model():
    ModelRegistry.register_model("DeepseekV4ForCausalLM", "vllm_ascend.models.deepseek_v4:AscendDeepseekV4ForCausalLM")

    ModelRegistry.register_model("DeepSeekV4MTPModel", "vllm_ascend.models.deepseek_v4_mtp:DeepSeekV4MTP")
    ModelRegistry.register_model(
        "LlamaForCausalLMVwnEagle3", "vllm_ascend.models.llama_eagle3_vwn:Eagle3VwnLlamaForCausalLM"
    )
    ModelRegistry.register_model(
        "Glm5NextForCausalLM", "vllm_ascend.models.glm5_next:AscendGlm5NextForCausalLM"
    )
    # Glm5NextForConditionalGeneration -> AscendGlm5NextForConditionalGeneration (多模态)
    # Glm5NextForCausalLM -> AscendGlm5NextForCausalLM (纯文本)
    ModelRegistry.register_model(
        "Glm5NextForConditionalGeneration",
        "vllm_ascend.models.glm5_next_multimodal:AscendGlm5NextForConditionalGeneration",
    )
    ModelRegistry.register_model(
        "Glm5NextMTPModel",
        "vllm_ascend.models.glm5_next_mtp:AscendGlm5NextMTP",
    )
    ModelRegistry.register_model(
        "Glm5NextMTP",
        "vllm_ascend.models.glm5_next_mtp:AscendGlm5NextMTP",
    )
