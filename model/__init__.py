import os

import torch

from transformers import AutoConfig, Qwen2_5OmniProcessor

def load_pretrained_model_flash(model_name=None, **kwargs):
    use_outro = kwargs.get("use_outro", False)
    cache_dir = kwargs.get("cache_dir") or os.environ.get("MODEL_CACHE_DIR")
    if use_outro:
        config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
        text_config = config.thinker_config.text_config
        text_config.use_outro = True
        # The public switch always enables the complete method.
        text_config.outro_apply_once = True
        text_config.outro_enable_relu_tanh = True
        text_config.outro_gamma = 3.0
        text_config.outro_temp = 0.1
        text_config.outro_rescale = True
        from .modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
    else:
        from transformers import Qwen2_5OmniForConditionalGeneration
        config = None

    model_kwargs = {
        "config": config,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "attn_implementation": "flash_attention_2",
    }
    if cache_dir:
        model_kwargs["cache_dir"] = cache_dir

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
    model.disable_talker()
    model.eval()

    processor_kwargs = {}
    if cache_dir:
        processor_kwargs["cache_dir"] = cache_dir
    processor = Qwen2_5OmniProcessor.from_pretrained(model_name, **processor_kwargs)

    return model, processor
