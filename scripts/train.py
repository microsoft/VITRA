"""
train.py

Main training script for VITRA Vision-Language-Action (VLA) models.
Supports distributed training with FSDP (Fully Sharded Data Parallel) strategy.
"""

import argparse
import copy
import datetime
import faulthandler
import json
import os
import random
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import wandb
from torch.utils.data import DataLoader

from vitra.datasets.materialize import get_vla_dataset_and_collator
from vitra.models.vla_builder import build_vla, load_vla_checkpoint
from vitra.training import VLAMetrics
from vitra.utils import (
    find_last_checkpoint,
    get_epoch_and_step_from_checkpoint,
    set_global_seed,
    setup_seed,
)
from vitra.training.fsdp import VLAFSDPStrategy
from vitra.utils.config_utils import load_config
from vitra.utils.overwatch import initialize_overwatch

# === Environment Configuration ===
# Disable tokenizers parallelism to avoid deadlocks in multi-process data loading
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Enable TF32 for faster training on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

def experiment(variant):
    """
    Main training experiment function for VITRA VLA models.
    
    Args:
        variant: Configuration dictionary containing all training parameters including:
            - Model architecture settings
            - Training hyperparameters
            - Dataset configurations
            - Logging and checkpoint paths
    """
    # === Device Setup ===
    torch.cuda.set_device(device_id := overwatch.local_rank())
    torch.cuda.empty_cache()
    
    # === Weights & Biases Setup ===
    overwatch.info("VITRA VLA Training :: Creating Folders", ctx_level=1)
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key is None:
        raise ValueError("Please set the WANDB_API_KEY environment variable.")
    wandb.login(key=wandb_api_key)
 
    # === Directory Setup ===
    os.makedirs(variant["log_root"], exist_ok=True)
    os.makedirs(variant["output_root"], exist_ok=True)
    os.makedirs(variant["cache_root"], exist_ok=True)
    
    # === Run ID and Checkpoint Directory ===
    # Create unique run identifier based on task name and batch configuration
    run_id = variant["task_name"] if "task_name" in variant else None
    batch_size = variant["batch_size"]
    total_batch_size = variant["total_batch_size"]
    run_id = f"{run_id}_TB{total_batch_size}_B{batch_size}_bf16{variant['use_bf16']}"
    
    checkpoint_dir = os.path.join(variant["output_root"], run_id)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # === Random Seed Setup ===
    worker_init_fn = set_global_seed(variant["seed"], get_worker_init_fn=True)

    # === Configuration Serialization ===
    def posix_to_str(d):
        if isinstance(d, dict):
            return {k: posix_to_str(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [posix_to_str(v) for v in d]
        elif isinstance(d, Path):
            return str(d)
        else:
            return d
    
    variant_str = copy.deepcopy(variant)
    copied_variant = posix_to_str(variant_str)

    if overwatch.rank() == 0:
        with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
            json.dump(copied_variant, f, indent=2)
        overwatch.info(f"Config saved to {checkpoint_dir}", ctx_level=1)
        print(json.dumps(copied_variant, indent=2))

    dist.barrier()
    
    # === Model Loading and Checkpoint Resume ===
    overwatch.info("Loading model", ctx_level=1)
    resume_step = 0
    resume_epoch = 0
    model_load_path = variant["model_load_path"]
    
    # Handle checkpoint resumption
    if variant["resume"]:
        # Auto-discover last checkpoint if path not specified
        if model_load_path is None:
            model_load_path = find_last_checkpoint(checkpoint_dir)
        
        # Parse resume epoch and step from checkpoint path
        if model_load_path is not None:
            resume_epoch, resume_step = get_epoch_and_step_from_checkpoint(model_load_path)
            if overwatch.rank() == 0:
                overwatch.info(
                    f"Resume from {model_load_path}, epoch: {resume_epoch}, step: {resume_step}",
                    ctx_level=1
                )

    # Build VLA model from configuration
    model = build_vla(configs=variant)
    pretrain_path = variant.get("pretrain_path", None)
    if variant['resume'] and model_load_path is not None:
        model = load_vla_checkpoint(model, os.path.join(model_load_path, "weights.pt"))
    elif pretrain_path is not None:
        if os.path.isdir(pretrain_path):
            model = load_vla_checkpoint(model, os.path.join(pretrain_path, "weights.pt"))
        else:
            model = load_vla_checkpoint(model, pretrain_path)

    model = model.train()
    model.trainable_params_setup()
    model.model.use_bf16 = variant["use_bf16"]
    model.use_bf16 = variant["use_bf16"]

    # Debug mode: freeze all parameters for testing
    if variant.get("debug", False):
        for p in model.model.parameters():
            p.requires_grad = False

    # Log parameter counts
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    if overwatch.rank() == 0:
        overwatch.info(f"Trainable Model Parameters: {total_params/1e6:.2f}M/{all_params/1e6:.2f}M")
    
    processor = model.processor

    # === Dataset Creation ===
    # Create VLA dataset with distributed data sharding
    vla_dataset, collator, batch_sampler = get_vla_dataset_and_collator(
        variant["train_dataset"]["data_root_dir"],
        variant["train_dataset"]["data_mix"],
        augmentation=variant["train_dataset"]["augmentation"],
        shard_num=dist.get_world_size(),  # Total number of distributed processes
        shard_index=dist.get_rank(),  # Current process rank
        seed=variant["seed"],
        future_action_window_size=variant["fwd_pred_next_n"] - 1,
        processor=processor,
        batch_size=batch_size,
        normalization=variant["train_dataset"].get("normalization", True),
        flip_augmentation=variant["train_dataset"].get("flip_augmentation", 1.0),
        set_none_ratio=variant["train_dataset"].get("set_none_ratio", 0.0),
        action_type=variant["train_dataset"].get('action_type', 'angle'),
        use_rel=variant["train_dataset"].get('use_rel', False),
        rel_mode=variant["train_dataset"].get('rel_mode', "step"),
        clip_len=variant["train_dataset"].get('clip_len', None),
        state_mask_prob=variant["train_dataset"].get('state_mask_prob', 0.1),
    )
    
    # === Training Strategy Setup ===
    # Initialize FSDP (Fully Sharded Data Parallel) training strategy
    training_strategy = VLAFSDPStrategy(
        vla=model,
        device_id=overwatch.local_rank(),
        stage=None,
        epochs=variant["trainer"]["max_epochs"],
        max_steps=variant["trainer"]["max_steps"],
        global_batch_size=variant["total_batch_size"],
        per_device_batch_size=batch_size,
        learning_rate=variant["trainer"]["learning_rate"],
        weight_decay=variant["trainer"]["weight_decay"],
        max_grad_norm=variant["trainer"]["gradient_clip_val"],
        lr_scheduler_type=variant["trainer"]["lr_scheduler_type"],
        warmup_ratio=variant["trainer"]["warmup_ratio"],
        enable_gradient_checkpointing=variant["trainer"]["enable_gradient_checkpointing"],
        enable_mixed_precision_training=variant["trainer"]["enable_mixed_precision_training"],
        reduce_in_full_precision=variant["trainer"]["reduce_in_full_precision"],
        action_model_learning_rate=variant["trainer"].get("action_model_learning_rate", None),
        action_model_weight_decay=variant["trainer"].get("action_model_weight_decay", None),
        sharding_strategy=variant["trainer"].get("sharding_strategy", "shard-grad-op"),
        cognition_token_weight_decay=variant["trainer"].get("cognition_token_weight_decay", True),
        llm_freeze_step=variant["trainer"].get("llm_freeze_step", 0),
        move_word_embedding_to_action_model=variant["trainer"].get("move_word_embedding_to_action_head", False),
        optimizer_betas=variant["trainer"].get("optimizer_betas", (0.9, 0.999)),
    )
    
    # === FSDP Wrapping and Checkpointing Policies ===
    # Define which modules should be wrapped by FSDP and which should use activation checkpointing
    if variant["vla_name"] == "VITRA_Paligemma":
        auto_wrap_policy, checkpointing_policy = get_fsdp_wrap_policy_and_checkpointing(variant["trainer"])
    else:
        raise NotImplementedError(f"Unsupported VLA name: {variant['vla_name']}")
    
    # Initialize FSDP wrapping, optimizer, and learning rate scheduler
    training_strategy.run_setup(
        run_dir=checkpoint_dir,
        n_train_examples=len(vla_dataset),
        auto_wrap_policy_modules=auto_wrap_policy,
        checkpointing_policy_modules=checkpointing_policy,
    )
    
    # Load optimizer and scheduler state if resuming from checkpoint
    if variant["resume"] == True and model_load_path is not None:
        training_strategy.load_optimizer_and_scheduler(model_load_path)
    
    # === Metrics Tracking Setup ===
    # Initialize metrics logging with Weights & Biases
    trackers = ["wandb"]
    overwatch.info(f"Creating Metrics with Active Trackers => `{trackers}`")
    metrics = VLAMetrics(
        trackers,
        hparams=variant_str,
        run_id=run_id,
        run_dir=checkpoint_dir,
        wandb_project=variant["wandb_project"],
        wandb_entity=variant["wandb_entity"],
        resume_step=resume_step,
        resume_epoch=resume_epoch,
    )
    
    # === DataLoader Creation ===
    overwatch.info("Creating Dataloader", ctx_level=1)
    
    num_workers = variant["num_workers"] if variant["num_workers"] is not None else variant["train_dataset"]["num_workers"]
    prefetch_factor = variant["prefetch_factor"] if variant["prefetch_factor"] is not None else variant["train_dataset"]["prefetch_factor"]

    if num_workers == 0 or prefetch_factor == 0:
        prefetch_factor = None

    if overwatch.rank() == 0:
        print(f"num_workers: {num_workers}, prefetch_factor: {prefetch_factor}")
    
    # Set batch sampler epoch for proper data shuffling when resuming
    batch_sampler.set_epoch(resume_epoch, resume_step * training_strategy.grad_accumulation_steps)

    setup_seed(variant["seed"], rank=torch.distributed.get_rank())

    # Create PyTorch DataLoader with multi-process data loading
    dataloader = DataLoader(
        vla_dataset,
        batch_sampler=batch_sampler,
        collate_fn=collator,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        worker_init_fn=worker_init_fn,
        persistent_workers=num_workers > 0,
        pin_memory=num_workers > 0,
    )

    # === Training Execution ===
    overwatch.info("Starting VLA Training Loop")
    training_strategy.run_training(
        dataloader,
        metrics,
        save_interval=variant["save_steps"],
        start_global_step=resume_step,
        start_epoch=resume_epoch,
    )

    # === Training Finalization ===
    overwatch.info("Done with Training =>> Finalizing Metrics")
    metrics.finalize()

    # === Cleanup ===
    overwatch.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()

def get_fsdp_wrap_policy_and_checkpointing(configs):
    """
    Get FSDP auto-wrapping policy and activation checkpointing policy for PaliGemma models.
    
    The auto-wrap policy determines which module types should be individually wrapped by FSDP,
    allowing for efficient memory usage and communication in distributed training.
    
    The checkpointing policy determines which modules should use activation checkpointing
    (gradient checkpointing) to trade computation for memory during training.
    
    Args:
        configs: Trainer configuration dictionary containing strategy settings
        
    Returns:
        Tuple of (auto_wrap_policy, checkpointing_policy):
            - auto_wrap_policy: Set of module classes to wrap with FSDP
            - checkpointing_policy: Set of module classes to apply gradient checkpointing, or None
    """
    if 'strategy' not in configs or configs['strategy'] == 'ddp':
        raise NotImplementedError("FSDP strategy not specified or DDP selected.")
    
    # Import model layer classes for wrapping
    from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer
    from transformers.models.paligemma.modeling_paligemma import PaliGemmaMultiModalProjector
    from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer, SiglipVisionTransformer
    
    from vitra.models.action_model import DiT
    from vitra.utils.nn_utils import MLPProjector
    
    # Define which module types should be wrapped by FSDP
    policy = {
        SiglipEncoderLayer,  # Vision encoder layers
        SiglipVisionTransformer,  # Vision transformer
        DiT,  # Diffusion Transformer for action model
        Gemma2DecoderLayer,  # Language model decoder layers
        PaliGemmaMultiModalProjector,  # Vision-language projection layer
        MLPProjector  # MLP projection layers
    }
    
    # Enable gradient checkpointing for Gemma2 layers if specified
    checkpointing_policy = (
        {Gemma2DecoderLayer}
        if configs["strategy"] == "fsdp_paligemma_with_checkpointing"
        else None
    )
    
    return policy, checkpointing_policy

def update_configs(configs, args):
    """
    Update configuration dictionary with command-line arguments.
    
    Command-line arguments take precedence over config file values. This function
    handles both top-level parameters and nested dictionaries (e.g., trainer settings).
    
    Args:
        configs: Base configuration dictionary loaded from YAML/JSON config file
        args: Parsed command-line arguments dictionary
        
    Returns:
        Updated configuration dictionary with command-line overrides applied
    """
    if args["task_name"] is not None:
        configs["task_name"] = args["task_name"]
    
    configs["use_bf16"] = (
        args["use_bf16"]
        if args["use_bf16"] is not None
        else configs.get("use_bf16", False)
    )

    if args["data_mix"] is not None:
        configs["train_dataset"]["data_mix"] = args["data_mix"]
    
    configs["output_root"] = Path(configs["output_root"])
    configs["log_root"] = Path(configs["log_root"])
    configs["cache_root"] = Path(configs["cache_root"]) / configs["model"]

    # Update remaining arguments (handles both flat and nested dictionaries)
    for k, v in args.items():
        if k not in configs:
            print(f"{k} not in config. The value is {v}.")
            configs[k] = v
        elif isinstance(v, dict):
            for sub_k, sub_v in v.items():
                if sub_v is not None:
                    configs[k][sub_k] = sub_v
        elif v is not None:
            configs[k] = v
    
    return configs

def parse_args():
    """
    Parse command-line arguments for training configuration.
    
    Arguments are organized into two groups:
    1. Global arguments (experiment settings, paths, data configuration)
    2. Trainer arguments (training hyperparameters and strategy)
    
    Returns:
        Dictionary with structure:
        {
            'config': str,
            'seed': int,
            ...other global args...,
            'trainer': {
                'strategy': str,
                'gradient_clip_val': float,
                ...other trainer args...
            }
        }
    """
    parser = argparse.ArgumentParser(description="VITRA VLA Training Script")
    
    # === Global Arguments ===
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML/JSON configuration file for training"
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--log_root",
        default=None,
        type=str,
        help="Root directory for logging"
    )
    parser.add_argument(
        "--output_root",
        default=None,
        type=str,
        help="Root directory for checkpoints and outputs"
    )
    parser.add_argument(
        "--model_load_path",
        default=None,
        type=str,
        help="Path to checkpoint for resuming training"
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        help="Unique identifier for this training run"
    )
    parser.add_argument(
        "--use_bf16",
        default=None,
        action="store_true",
        help="Enable bfloat16 mixed precision training"
    )
    parser.add_argument(
        "--data_mix",
        default=None,
        type=str,
        help="Dataset mixture configuration"
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Enable debug mode (freezes model parameters)"
    )
    parser.add_argument(
        "--fwd_pred_next_n",
        default=None,
        type=int,
        help="Number of future action steps to predict"
    )
    parser.add_argument(
        "--batch_size",
        default=None,
        type=int,
        help="Per-device batch size"
    )
    parser.add_argument(
        "--total_batch_size",
        default=None,
        type=int,
        help="Global batch size across all devices"
    )
    parser.add_argument(
        "--num_workers",
        default=None,
        type=int,
        help="Number of data loading workers per process"
    )
    parser.add_argument(
        "--prefetch_factor",
        default=None,
        type=int,
        help="Number of batches to prefetch per worker"
    )
    
    # Capture global argument names before adding trainer group
    global_names = set(vars(parser.parse_known_args()[0]).keys())

    # === Trainer Arguments Group ===
    trainer_parser = parser.add_argument_group("trainer", "Training strategy and hyperparameters")
    trainer_parser.add_argument(
        "--strategy",
        default=None,
        type=str,
        help="Training strategy (e.g., 'fsdp')"
    )
    trainer_parser.add_argument(
        "--gradient_clip_val",
        default=None,
        type=float,
        help="Maximum gradient norm for clipping"
    )
    trainer_parser.add_argument(
        "--max_steps",
        default=None,
        type=int,
        help="Maximum number of training steps (overrides epochs)"
    )
    
    # Capture trainer argument names (difference from global)
    trainer_names = set(vars(parser.parse_known_args()[0]).keys()) - global_names

    # === Parse and Organize Arguments ===
    args = {}
    trainer_args = {}
    temp_args = vars(parser.parse_args())
    
    # Separate global and trainer arguments
    for k, v in temp_args.items():
        if k in global_names:
            args[k] = v
        elif k in trainer_names:
            trainer_args[k] = v

    # Nest trainer arguments under 'trainer' key
    args["trainer"] = trainer_args

    return args


if __name__ == "__main__":
    # Enable fault handler for better debugging of segmentation faults
    faulthandler.enable()

    args = parse_args()

    configs = load_config(args.get("config"))
    configs = update_configs(configs, args)
    
    # Initialize distributed training backend (NCCL for NVIDIA GPUs)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    experiment(variant=configs)