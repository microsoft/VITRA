import os
import copy
import json
import torch
import transformers
from pathlib import Path
from tqdm.auto import tqdm
from huggingface_hub import hf_hub_download

import vitra.models.vla as vla
from vitra.utils.data_utils import read_dataset_statistics, GaussianNormalizer


def build_vla(configs):
    model_fn = getattr(vla, configs["vla_name"])
    model = model_fn(
        configs=configs,
        train_setup_configs=configs["train_setup"],
        act_model_configs=configs["action_model"],
        fwd_pred_next_n=configs["fwd_pred_next_n"],
        repeated_diffusion_steps=configs.get("repeated_diffusion_steps", 8),
        use_state=configs["use_state"],
        use_fov=configs.get("use_fov", True),
    )
    return model

def load_model(configs):
    model_load_path = configs.get("model_load_path", None)
    model = build_vla(
        configs=configs,
    )
    if model_load_path is not None:
        # Check if model_load_path is a file or directory
        if os.path.isfile(model_load_path):
            # If it's a file, load it directly
            checkpoint_path = model_load_path
        else:
            # If it's a directory, look for weights.pt inside
            checkpoint_path = os.path.join(model_load_path, "weights.pt")
        
        model = load_vla_checkpoint(model, checkpoint_path)

    return model

def load_vla_checkpoint(model, checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    with torch.no_grad():
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=True)
    print("Checkpoint loaded")
    return model

