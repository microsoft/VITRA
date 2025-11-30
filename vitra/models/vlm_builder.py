import copy
import transformers
import torch

def build_vlm(vlm_config):
    vlm_config = copy.deepcopy(vlm_config)
    model_path = vlm_config.get("pretrained_model_name_or_path")
    model_name = vlm_config.get("name")
    model_type = vlm_config.get("type", "AutoModel")
    if model_name == "paligemma":
        from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration

        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            # attn_implementation="eager",
            # revision="bfloat16",
        )
        processor = PaliGemmaProcessor.from_pretrained(model_path)
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")

    return processor, model
