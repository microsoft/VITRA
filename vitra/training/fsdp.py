"""
fsdp.py

Core class definition for a strategy implementing Torch native Fully Sharded Data Parallel Training (with support for
fine-grained control over wrapping policies and mixed precision per component).
"""
import gc
import json
import math
import threading
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullOptimStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import AdamW
from tqdm import tqdm
from transformers.optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
)

from vitra.training.base_strategy import TrainingStrategy
from vitra.training.metrics import VLAMetrics
from vitra.utils.overwatch import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)  


def get_constant_schedule_with_freeze_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a learning rate scheduler that is zero for the first `num_warmup_steps` steps, then constant."""
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return 0.0
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def split_modality_collator(
    vla,
    cognition_token_weight_decay: bool = False,
    move_word_embedding_to_action_model: bool = False,
    verbose: bool = True
):
    """
    Split model parameters into vlm backbone and other (action model) groups with separate decay settings.
    
    Returns:
        Tuple of (backbone_decay, backbone_no_decay, other_decay, other_no_decay) parameter lists
    """
    backbone_decay, backbone_no_decay, other_decay, other_no_decay = [], [], [], []
    
    def is_backbone_param(name: str) -> bool:
        """Check if the parameter is part of the vision or text backbone."""
        if move_word_embedding_to_action_model and "embed_tokens" in name:
            return False
        return "backbone" in name

    for name, param in vla.named_parameters():
        if not param.requires_grad:
            continue
        
        # Check parameters that should not have weight decay
        no_weight_decay = param.ndim <= 1 or name.endswith(".bias")
        if "cognition_token" in name:
            no_weight_decay = not cognition_token_weight_decay
        
        # Categorize parameters
        if no_weight_decay:
            if is_backbone_param(name):
                backbone_no_decay.append(param)
                if verbose:
                    overwatch.info(f"Parameter `{name}` is part of the backbone and has no decay; added to `backbone_no_decay`")
            else:
                other_no_decay.append(param)
                if verbose:
                    overwatch.info(f"Parameter `{name}` is not part of the backbone and has no decay; added to `other_no_decay`")
        else:
            if is_backbone_param(name):
                backbone_decay.append(param)
                if verbose:
                    overwatch.info(f"Parameter `{name}` is part of the backbone and has decay; added to `backbone_decay`")
            else:
                other_decay.append(param)
                if verbose:
                    overwatch.info(f"Parameter `{name}` is not part of the backbone and has decay; added to `other_decay`")
    
    return backbone_decay, backbone_no_decay, other_decay, other_no_decay


class VLAFSDPStrategy(TrainingStrategy):
    """FSDP (Fully Sharded Data Parallel) training strategy for VLA models."""

    def __init__(
        self,
        vla,
        device_id: int,
        stage: str,
        epochs: int,
        max_steps: Optional[int],
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warmup_ratio: float,
        enable_gradient_checkpointing: bool = True,
        enable_mixed_precision_training: bool = True,
        reduce_in_full_precision: bool = False,
        action_model_learning_rate: Optional[float] = None,
        action_model_weight_decay: Optional[float] = None,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        sharding_strategy: str = "shard-grad-op",
        state_dict_type: StateDictType = StateDictType.FULL_STATE_DICT,
        cognition_token_weight_decay: bool = False,
        llm_freeze_step: int = 0,
        move_word_embedding_to_action_model: bool = False,
        optimizer_betas: tuple = (0.9, 0.999),
    ) -> None:
        super().__init__(
            vla=vla,
            device_id=device_id,
            stage=stage,
            epochs=epochs,
            max_steps=max_steps,
            global_batch_size=global_batch_size,
            per_device_batch_size=per_device_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
            enable_mixed_precision_training=enable_mixed_precision_training,
            reduce_in_full_precision=reduce_in_full_precision,
            mixed_precision_dtype=mixed_precision_dtype,
        )
        # Action model specific parameters
        self.action_model_learning_rate = action_model_learning_rate if action_model_learning_rate is not None else learning_rate
        self.action_model_weight_decay = action_model_weight_decay if action_model_weight_decay is not None else weight_decay
        self.cognition_token_weight_decay = cognition_token_weight_decay
        self.llm_freeze_step = llm_freeze_step
        self.move_word_embedding_to_action_model = move_word_embedding_to_action_model
        self.optimizer_betas = optimizer_betas

        # FSDP-specific parameters
        if sharding_strategy == "shard-grad-op":
            self.fsdp_sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
        elif sharding_strategy == "full-shard":
            self.fsdp_sharding_strategy = ShardingStrategy.HYBRID_SHARD
        else:
            raise ValueError(f"FSDP sharding strategy '{sharding_strategy}' is not supported!")

        assert state_dict_type == StateDictType.FULL_STATE_DICT, "Sharded state saving is not yet implemented!"
        self.fsdp_state_dict_type = state_dict_type
        self.fsdp_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        self.fsdp_save_optimizer_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)

    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        only_trainable: bool = True,
        is_epoch_end: bool = False,
    ) -> None:
        """Save a checkpoint to the `run_dir` only containing the state_dicts for trainable parameters by default."""
        assert isinstance(self.vla, FSDP), "FSDPStrategy.save_checkpoint assumes VLM is already wrapped in FSDP!"
        if is_epoch_end:
            checkpoint_name = f"epoch={epoch}-step={global_step}.end.ckpt"
        else:
            checkpoint_name = f"epoch={epoch}-step={global_step}.ckpt"
        checkpoint_dir = run_dir / "checkpoints"/ checkpoint_name
        if overwatch.is_rank_zero():
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        def save_with_time(state_dict, path):
            overwatch.info(f"Saving state dict to {path} start at {datetime.now()}")
            torch.save(state_dict, path)
            overwatch.info(f"Saving state dict to {path} end at {datetime.now()}")
        
        # Gather full state dictionary from shards
        with FSDP.state_dict_type(self.vla, self.fsdp_state_dict_type, self.fsdp_save_policy, self.fsdp_save_optimizer_policy):
            overwatch.info("Gathering model state")
            model_state = self.vla.state_dict()
            overwatch.info("Preparing save checkpoint")
            overwatch.info("Gathering optimizer state")
            optim_state = FSDP.optim_state_dict(self.vla, self.optimizer)
            meta_state = {
                "epoch": epoch,
                "global_step": global_step
            }
            if overwatch.is_rank_zero():
                with open(checkpoint_dir / "meta.json", "w") as f:
                    json.dump(meta_state, f)
            dist.barrier()
            if overwatch.is_rank_zero():
                threading.Thread(target=save_with_time, args=(model_state, checkpoint_dir / 'weights.pt')).start()
                threading.Thread(target=save_with_time, args=(optim_state, checkpoint_dir / 'optimizer.pt')).start()
            
            dist.barrier()

    def load_optimizer_and_scheduler(self, checkpoint_folder: str) -> None:
        """Load optimizer and scheduler state from checkpoint."""
        assert isinstance(self.vla, FSDP), "FSDPStrategy.load_optimizer_and_scheduler assumes VLM is already wrapped in FSDP!"
        
        checkpoint_folder = Path(checkpoint_folder)
        optimizer_path = checkpoint_folder / "optimizer.pt"
        
        if not optimizer_path.exists():
            overwatch.warning(f"Optimizer checkpoint not found at {optimizer_path}!")
            return
        
        # Load checkpoint (FSDP handles device placement automatically)
        optim_state_dict = torch.load(optimizer_path, map_location="cpu")
        
        with FSDP.state_dict_type(
            self.vla,
            self.fsdp_state_dict_type,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False)
        ):
            optim_state_dict = FSDP.optim_state_dict_to_load(self.vla, self.optimizer, optim_state_dict)
            # optim_state_dict = FSDP.optim_state_dict_to_load(self.vla, self.optimizer, optim_state_dict["optimizer"])
            self.optimizer.load_state_dict(optim_state_dict)
        
        overwatch.info(f"Loaded optimizer state dict from {optimizer_path}")
        
    def run_setup(
        self,
        run_dir: Path,
        n_train_examples: int,
        auto_wrap_policy_modules,
        checkpointing_policy_modules,
    ) -> None:
        """Setup FSDP training (wrap model, create optimizer, etc.)."""
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy

        auto_wrap_policy = ModuleWrapPolicy(auto_wrap_policy_modules)

        # Configure FSDP mixed precision policy
        if self.enable_mixed_precision_training and self.mixed_precision_dtype == torch.bfloat16:
            reduce_buffer_dtype = torch.bfloat16 if not self.reduce_in_full_precision else torch.float32
            fsdp_precision_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=reduce_buffer_dtype,
                buffer_dtype=reduce_buffer_dtype
            )
        else:
            fsdp_precision_policy = MixedPrecision(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32
            )

        # <FSDP> => note that FSDP will automatically take care of device placement (similar to `autocast`)
        self.vla = FSDP(
            self.vla,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=fsdp_precision_policy,
            sharding_strategy=self.fsdp_sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            use_orig_params=True,
        )
        
        # Setup gradient checkpointing
        if self.enable_gradient_checkpointing:
            # For Gradient Checkpointing under FSDP --> we make the same assumption as in the DDP/other strategies; the
            #   bulk of activation memory is taken up by the LLM activations. However, unlike other strategies, we
            #   cannot rely on the HF Transformers default `gradient_checkpointing_enable()` --> FSDP breaks semantics!
            #
            # Instead, we need to write our own *NO-REENTRANT* wrapper, and apply it to the LLM's Transformer Layer.
            non_reentrant_wrapper = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
            if checkpointing_policy_modules is not None:
                def check_fn(submodule: nn.Module) -> bool:
                    if isinstance(checkpointing_policy_modules, (list, set)):
                        return any(isinstance(submodule, module) for module in checkpointing_policy_modules)
                    return isinstance(submodule, checkpointing_policy_modules)

                # Note that the terms "activation checkpointing" and "gradient checkpointing" are synonymous!
                apply_activation_checkpointing(self.vla, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)

        dist.barrier()

        # Create Optimizer and LR Scheduler =>> note that most of the LR Schedulers we use require `max_steps/epochs`
        #   => Optimizer should only operate on parameters that are *unfrozen* / trainable!
        n_train_examples = math.ceil(n_train_examples / self.global_batch_size) * self.global_batch_size
        if self.max_steps is None:
            num_training_steps = (n_train_examples * self.epochs) // self.global_batch_size
        else:
            num_training_steps = self.max_steps

        backbone_decay, backbone_no_decay, other_decay, other_no_decay = split_modality_collator(
            self.vla,
            cognition_token_weight_decay=self.cognition_token_weight_decay,
            move_word_embedding_to_action_model=self.move_word_embedding_to_action_model,
            verbose=False
        )
        groups = [
            {"params": backbone_decay, "weight_decay": self.weight_decay, "lr": self.learning_rate},
            {"params": backbone_no_decay, "weight_decay": 0.0, "lr": self.learning_rate},
            {"params": other_decay, "weight_decay": self.action_model_weight_decay, "lr": self.action_model_learning_rate},
            {"params": other_no_decay, "weight_decay": 0.0, "lr": self.action_model_learning_rate},
        ]

        # Create Optimizer & LR Scheduler
        self.optimizer = AdamW(groups, betas=self.optimizer_betas)

        if self.lr_scheduler_type == "linear-warmup+cosine-decay" or self.lr_scheduler_type == "warmup_cosine":
            # Set warmup steps (floor) based on `warmup_ratio` (should be 0.03 - 0.05)
            num_warmup_steps = int(num_training_steps * self.warmup_ratio)

            self.lr_scheduler = get_cosine_with_min_lr_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps,
                num_training_steps,
                min_lr_rate=0.1
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = 0.0

        elif self.lr_scheduler_type == "constant":
            num_warmup_steps = 0
            self.lr_scheduler = get_constant_schedule(self.optimizer)

        elif self.lr_scheduler_type == "linear-warmup+constant" or self.lr_scheduler_type == "warmup_constant":
            num_warmup_steps = int(num_training_steps * self.warmup_ratio)
            self.lr_scheduler = get_constant_schedule_with_warmup(
                self.optimizer, num_warmup_steps=num_warmup_steps
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = 0.0

        elif self.lr_scheduler_type == "backbone-freeze-warmup":
            # Backbone uses constant-freeze-warmup, action head uses constant
            num_warmup_steps = self.llm_freeze_step

            # Create separate optimizers for different scheduling
            backbone_groups = [
                {"params": backbone_decay, "weight_decay": self.weight_decay, "lr": self.learning_rate},
                {"params": backbone_no_decay, "weight_decay": 0.0, "lr": self.learning_rate},
            ]
            action_model_groups = [
                {"params": other_decay, "weight_decay": self.action_model_weight_decay, "lr": self.action_model_learning_rate},
                {"params": other_no_decay, "weight_decay": 0.0, "lr": self.action_model_learning_rate},
            ]

            # Create separate optimizers for backbone and action model
            backbone_optimizer = AdamW(backbone_groups, betas=self.optimizer_betas)
            action_model_optimizer = AdamW(action_model_groups, betas=self.optimizer_betas)
            
            # Create schedulers for each component
            backbone_scheduler = get_constant_schedule_with_freeze_warmup(
                backbone_optimizer, num_warmup_steps=num_warmup_steps
            )
            action_model_scheduler = get_constant_schedule(action_model_optimizer)
            
            # Create the multi-group scheduler
            self.lr_scheduler = MultiGroupLRScheduler(
                self.optimizer, backbone_scheduler, action_model_scheduler
            )
        else:
            raise ValueError(f"Learning Rate Schedule with type `{self.lr_scheduler_type}` is not supported!")

        # Finalize Setup =>> Log!
        scheduler_info = f"         |-> LR Scheduler Type = {self.lr_scheduler_type}\n"
        if self.lr_scheduler_type == "backbone-freeze-warmup+action-constant":
            scheduler_info += f"                 |-> Backbone: Constant schedule with freeze warmup ({num_warmup_steps} steps)\n"
            scheduler_info += f"                 |-> Action Head: Constant schedule\n"
        else:
            scheduler_info += f"         |-> LR Scheduler Warmup Steps (Ratio) = {num_warmup_steps} ({self.warmup_ratio})\n"

        overwatch.info(
            "FSDP Full-Shard Strategy =>> Finalized Training Setup:\n"
            f"         |-> Global (Effective) Batch Size = {self.global_batch_size}\n"
            f"         |-> Per-Device Batch Size = {self.per_device_batch_size}\n"
            f"         |-> Distributed World Size = {overwatch.world_size()}\n"
            f"         |-> Gradient Accumulation Steps = {self.grad_accumulation_steps}\n\n"
            f"         |-> LLM Backbone FSDP Gradient Checkpointing = {self.enable_gradient_checkpointing}\n"
            f"         |-> Use FSDP Mixed Precision = {self.enable_mixed_precision_training}\n"
            f"                 |-> Parameter Precision = {fsdp_precision_policy.param_dtype}\n"
            f"                 |-> Reduction Precision = {fsdp_precision_policy.reduce_dtype}\n"
            f"                 |-> Buffer Precision = {fsdp_precision_policy.buffer_dtype}\n\n"
            f"         |-> Default AdamW LR = {self.learning_rate}\n"
            f"         |-> AdamW Weight Decay = {self.weight_decay}\n"
            f"         |-> AdamW Betas = {self.optimizer_betas}\n"
            + scheduler_info +
            f"         |-> LLM Learning Rate = {self.learning_rate}\n"
            f"         |-> Action Model Learning Rate = {self.action_model_learning_rate}\n"
            f"         |-> LLM Weight Decay = {self.weight_decay}\n"
            f"         |-> Action Model Weight Decay = {self.action_model_weight_decay}\n"
            f"         |-> Cognition Token Weight Decay = {self.cognition_token_weight_decay}\n"
            f"         |-> Dataset Size = {n_train_examples} Examples\n"
            f"         |-> Max Steps = {num_training_steps}\n"
        )

    def clip_grad_norm(self) -> None:
        """Clip gradients using FSDP's built-in gradient clipping."""
        self.vla.clip_grad_norm_(max_norm=self.max_grad_norm)

    def run_training(
        self,
        dataloader,
        metrics: VLAMetrics,
        save_interval: int = 2500,
        epoch_save_interval: int = 1,
        start_epoch: int = 0,
        start_global_step: int = 0,
        save_full_model: bool = True,
    ) -> None:
        """Run the VLA training loop for the given dataloader; log losses and action metrics to metrics."""
        vla_dataset = dataloader.dataset
        
        status = metrics.get_status()
        with tqdm(
            total=(self.epochs * (len(dataloader) // self.grad_accumulation_steps)) if self.max_steps is None else self.max_steps,
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
            initial=start_global_step,
        ) as progress:
            train_idx = 0
            for epoch in range(start_epoch, self.epochs):
                self.vla.train()
                self.optimizer.zero_grad()
                for batch_idx, batch in enumerate(dataloader):
                    # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                    #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                    input_ids = batch["input_ids"]
                    rgb = batch["pixel_values"]
                    attention_mask = batch["attention_mask"]
                    action_labels = batch["actions"]
                    action_masks = batch["action_masks"]
                    current_state_mask = batch["current_state_mask"]
                    current_state = batch["current_state"]
                    fov = batch["fov"]

                    prediction = self.vla.forward(
                        rgb,
                        input_ids,
                        attention_mask=attention_mask,
                        action_labels=action_labels,
                        action_masks=action_masks,
                        current_state_mask=current_state_mask,
                        current_state=current_state,
                        data_source=['action'],
                        fov=fov,
                    )
                    loss = prediction["loss"]

                    # Commit loss and backward
                    metrics.commit(
                        loss=loss, 
                        left_hand_6d=prediction["left_hand_6d"],
                        left_hand_joints=prediction["left_hand_joints"],
                        right_hand_6d=prediction["right_hand_6d"],
                        right_hand_joints=prediction["right_hand_joints"],
                    )
                    
                    normalized_loss = loss / self.grad_accumulation_steps
                    normalized_loss.backward()

                    # === Gradient Step ===
                    # Step =>> Only if Done w/ Gradient Accumulation
                    if (train_idx + 1) % self.grad_accumulation_steps == 0:
                        # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality-assumptions
                        self.clip_grad_norm()

                        # Optimizer & LR Scheduler Step
                        self.optimizer.step()
                        self.lr_scheduler.step()

                        self.optimizer.zero_grad()
                        # Compute epoch value using number of completed gradient steps
                        # epoch = (metrics.global_step + 1) // (len(vla_dataset) // self.global_batch_size)

                        # Prepare learning rate metrics
                        lr_dict = {}
                        # Get the appropriate learning rate for logging
                        if isinstance(self.lr_scheduler, MultiGroupLRScheduler):
                            # For multi-group scheduler, log multiple learning rates
                            lr = self.lr_scheduler.get_last_lr()
                            lr_dict['backbone_decay_lr'] = lr[0]       # backbone decay learning rate
                            lr_dict['backbone_no_decay_lr'] = lr[1]    # backbone no decay learning rate
                            lr_dict['action_decay_lr'] = lr[2]         # action decay learning rate
                            lr_dict['action_no_decay_lr'] = lr[3]      # action no decay learning rate
                            current_lr = lr_dict['backbone_decay_lr']  # backbone learning rate
                        else:
                            current_lr = self.lr_scheduler.get_last_lr()[0]
                        
                        metrics.commit(update_step_time=True, global_step=metrics.global_step + 1, epoch=epoch, lr=current_lr, **lr_dict)
                        status = metrics.push()

                        # Check for Save Interval or Max Steps & Save Checkpoint
                        if (terminate := (self.max_steps is not None and metrics.global_step >= self.max_steps)) or (
                            (metrics.global_step % save_interval) == 0
                        ):
                            self.save_checkpoint(
                                metrics.run_dir, metrics.global_step, epoch, only_trainable=not save_full_model
                            )
                            dist.barrier()

                        if terminate:
                            return
                    train_idx += 1

                    # Update progress bar
                    progress.set_description(status)
                    progress.update()
                    
                # Save epoch checkpoint if needed
                if epoch % epoch_save_interval == 0:
                    self.save_checkpoint(
                        metrics.run_dir, metrics.global_step, epoch, only_trainable=not save_full_model, is_epoch_end=True
                    )
                gc.collect()
                torch.cuda.empty_cache()

# Custom LR Scheduler for different parameter groups
class MultiGroupLRScheduler:
    """
    A custom learning rate scheduler that applies different scheduling strategies
    to different parameter groups in the optimizer.
    """
    def __init__(self, optimizer, backbone_scheduler, action_model_scheduler):
        self.optimizer = optimizer
        self.backbone_scheduler = backbone_scheduler
        self.action_model_scheduler = action_model_scheduler
        
        # Assume first two groups are backbone (decay/no_decay), last two are action model
        self.backbone_group_indices = [0, 1]
        self.action_model_group_indices = [2, 3]
    
    def step(self):
        """Step both schedulers and update the corresponding parameter groups"""
        # Step the schedulers
        self.backbone_scheduler.step()
        self.action_model_scheduler.step()
        
        # Update backbone parameter groups with backbone scheduler's learning rates
        backbone_lrs = self.backbone_scheduler.get_last_lr()
        for i, group_idx in enumerate(self.backbone_group_indices):
            # Both backbone groups should use the same LR from backbone scheduler
            self.optimizer.param_groups[group_idx]['lr'] = backbone_lrs[0] if len(backbone_lrs) == 1 else backbone_lrs[i]
        
        # Update action model parameter groups with action model scheduler's learning rates
        action_model_lrs = self.action_model_scheduler.get_last_lr()
        for i, group_idx in enumerate(self.action_model_group_indices):
            # Both action model groups should use the same LR from action model scheduler
            self.optimizer.param_groups[group_idx]['lr'] = action_model_lrs[0] if len(action_model_lrs) == 1 else action_model_lrs[i]
    
    def get_last_lr(self):
        """Return the last learning rates for all parameter groups"""
        backbone_lrs = self.backbone_scheduler.get_last_lr()
        action_model_lrs = self.action_model_scheduler.get_last_lr()
        
        # Return LRs in the order of parameter groups: [backbone_decay, backbone_no_decay, action_decay, action_no_decay]
        return [
            backbone_lrs[0] if len(backbone_lrs) == 1 else backbone_lrs[0],  # backbone_decay
            backbone_lrs[0] if len(backbone_lrs) == 1 else backbone_lrs[0],  # backbone_no_decay
            action_model_lrs[0] if len(action_model_lrs) == 1 else action_model_lrs[0],  # action_decay
            action_model_lrs[0] if len(action_model_lrs) == 1 else action_model_lrs[0],  # action_no_decay
        ]
