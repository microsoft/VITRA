"""
materialize.py

Dataset and collator materialization for VLA training.
"""

import time
from pathlib import Path
from typing import Tuple

from vitra.datasets.dataset import MultipleWeightedDataset, MultipleDatasetWeightedDistributedBatchSampler
from vitra.utils.data_utils import PaddedCollatorForHandPrediction

def get_vla_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    padding_side: str = "right",
    augmentation: bool = False,
    shard_num: int = 1,
    shard_index: int = 0,
    seed: int = 42,
    future_action_window_size: int = 0,
    batch_size: int = 32,
    processor=None,
    normalization: bool = True,
    flip_augmentation: float = 1.0,
    set_none_ratio: float = 0.0,
    statistics_type: str = 'ori',
    action_type: str = "angle",
    use_rel: bool = False,
    rel_mode: str = "step",
    clip_len: int = None,
    state_mask_prob: float = 0.1,
) -> Tuple[MultipleWeightedDataset, PaddedCollatorForHandPrediction, MultipleDatasetWeightedDistributedBatchSampler]:
    """
    Create VLA dataset, batch sampler, and collator for training.
    
    Args:
        data_root_dir: Root directory containing datasets
        data_mix: Comma-separated list of dataset names to mix
        padding_side: Token padding side ("right" or "left")
        augmentation: Whether to apply data augmentation
        shard_num: Number of distributed shards
        shard_index: Index of current shard
        seed: Random seed for sampling
        future_action_window_size: Number of future actions to predict
        batch_size: Batch size per device
        processor: Text processor with tokenizer
        normalization: Whether to normalize actions
        flip_augmentation: Probability of flip augmentation
        set_none_ratio: Ratio of setting actions to None
        statistics_type: Type of statistics to use
        action_type: Type of action representation
        use_rel: Whether to use relative actions
        rel_mode: Relative action mode
    
    Returns:
        Tuple of (dataset, collator, batch_sampler)
    """
    assert 0 <= shard_index < shard_num, "Shard index must be in [0, shard_num)."

    multi_dataset = MultipleWeightedDataset.load_datasets(
        data_root_dir,
        data_mix, 
        action_past_window_size=0,
        action_future_window_size=future_action_window_size,
        image_past_window_size=0,
        augmentation=augmentation,
        normalization = normalization,
        processor = processor,
        flip_augmentation = flip_augmentation,
        set_none_ratio = set_none_ratio,
        action_type = action_type, 
        use_rel = use_rel, 
        rel_mode = rel_mode,
        clip_len=clip_len, 
        state_mask_prob=state_mask_prob
    )

    batch_sampler = MultipleDatasetWeightedDistributedBatchSampler(
        multi_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        seed=seed,
        num_replicas=shard_num,
        rank=shard_index,
    )

    # Create collator for padding
    collator = PaddedCollatorForHandPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side=padding_side
    )

    return multi_dataset, collator, batch_sampler