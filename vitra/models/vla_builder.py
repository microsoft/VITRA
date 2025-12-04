import os
import copy
import json
import torch
import transformers
from pathlib import Path
from tqdm.auto import tqdm
from huggingface_hub import hf_hub_download, list_repo_files

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
        # Check if model_load_path is a Hugging Face repo (format: "username/repo-name")
        if "/" in model_load_path and not os.path.exists(model_load_path):
            # Load from Hugging Face Hub
            print(f"Loading model from Hugging Face Hub: {model_load_path}")
            # List all files in the repo
            repo_files = list_repo_files(repo_id=model_load_path)
            # Find .pt files in checkpoints/ folder
            checkpoint_files = [f for f in repo_files if f.startswith("checkpoints/") and f.endswith(".pt")]
            if not checkpoint_files:
                # Fallback: look for any .pt file in the repo
                checkpoint_files = [f for f in repo_files if f.endswith(".pt")]
            if not checkpoint_files:
                raise ValueError(f"No .pt checkpoint files found in {model_load_path}")
            # Use the first .pt file found
            checkpoint_filename = checkpoint_files[0]
            print(f"Found checkpoint: {checkpoint_filename}")
            checkpoint_path = hf_hub_download(
                repo_id=model_load_path,
                filename=checkpoint_filename,
                cache_dir=configs.get("hf_cache_dir", None)
            )
        # Check if model_load_path is a file or directory
        elif os.path.isfile(model_load_path):
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

