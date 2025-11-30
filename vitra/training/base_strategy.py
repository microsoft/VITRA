"""
base_strategy.py

Abstract base class definition of a (distributed) training strategy, with full annotations of class methods, utility
functions, and initialization logic.

Training Strategies (DDP, FSDP-Grad, FSDP-Full) tend to have a lot of repeated components; this class does the
heavy lifting for common functionality.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

import torch

from vitra.training.metrics import VLAMetrics
from vitra.utils.overwatch import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === Abstract Base Class for an arbitrary Training Strategy ===
class TrainingStrategy(ABC):
    """Abstract base class for training strategies (DDP, FSDP, etc.)."""
    
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
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        repeated_diffusion_steps: int = 4,
        **_: str,
    ) -> None:
        """
        Initialize training strategy.
        
        Args:
            vla: Vision-Language-Action model
            device_id: CUDA device ID
            stage: Training stage identifier
            epochs: Number of training epochs
            max_steps: Maximum training steps (optional)
            global_batch_size: Total batch size across all devices
            per_device_batch_size: Batch size per device
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            max_grad_norm: Maximum gradient norm for clipping
            lr_scheduler_type: Type of learning rate scheduler
            warmup_ratio: Warmup ratio for scheduler
            enable_gradient_checkpointing: Whether to enable gradient checkpointing
            enable_mixed_precision_training: Whether to use mixed precision
            reduce_in_full_precision: Whether to reduce gradients in full precision
            mixed_precision_dtype: Data type for mixed precision training
            repeated_diffusion_steps: Number of repeated diffusion steps
        """
        # Model and device
        self.vla = vla
        self.device_id = device_id
        self.stage = stage

        # Optimization parameters
        self.epochs = epochs
        self.max_steps = max_steps
        self.global_batch_size = global_batch_size
        self.per_device_batch_size = per_device_batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.lr_scheduler_type = lr_scheduler_type
        self.warmup_ratio = warmup_ratio

        # Training strategy parameters
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.reduce_in_full_precision = reduce_in_full_precision
        self.mixed_precision_dtype = mixed_precision_dtype
        self.repeated_diffusion_steps = repeated_diffusion_steps

        # Optimizer and scheduler (initialized in run_setup)
        self.optimizer = None
        self.lr_scheduler = None

        # Validate and compute gradient accumulation steps
        assert (
            self.global_batch_size % self.per_device_batch_size == 0
        ), "Per-device batch size must evenly divide global batch size!"
        
        self.grad_accumulation_steps = self.global_batch_size // self.per_device_batch_size // overwatch.world_size()
        
        if self.grad_accumulation_steps == 0:
            self.grad_accumulation_steps = 1
            overwatch.warning(
                "Global batch size is smaller than per-device batch size; gradient accumulation steps set to 1!"
            )
            overwatch.warning(f"Effective global batch size is now {self.global_batch_size}")

    @abstractmethod
    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None:
        """Save model checkpoint."""
        ...

    @abstractmethod
    def load_optimizer_and_scheduler(self, checkpoint_path: str) -> None:
        """Load optimizer and scheduler state from checkpoint."""
        ...
    
    @abstractmethod
    def run_setup(self, run_dir: Path, n_train_examples: int) -> None:
        """Setup training (wrap model, create optimizer, etc.)."""
        ...

    @abstractmethod
    def clip_grad_norm(self) -> None:
        """Clip gradient norms."""
        ...

    @abstractmethod
    def run_training(
        self,
        dataloader,
        metrics: VLAMetrics,
        save_interval: int = 2500,
        start_global_step: int = 0,
        save_full_model: bool = True,
    ) -> None:
        """Run the training loop."""
        ...