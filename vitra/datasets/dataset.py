import bisect
import copy
import json
import math
import os
import random
import time
from functools import lru_cache
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
from tqdm import tqdm

from vitra.datasets.data_mixture import HAND_MIXTURES
from vitra.datasets.robot_dataset import RoboDatasetCore
from vitra.datasets.human_dataset import EpisodicDatasetCore

class FrameDataset(Dataset):
    def __init__(self, dataset_folder, dataset_name,
                 image_past_window_size=0, image_future_window_size=0, action_past_window_size=0, action_future_window_size=0,
                 augmentation=False, normalization=True, processor=None, flip_augmentation=1.0, set_none_ratio=0.0,
                 action_type="angle", use_rel=False, rel_mode='step', load_images=True, data_type='human', clip_len=None, state_mask_prob=0.1):
        # only support image_past_window_size=0 now (in the post transform)
        """Both past and future window size does not include the current frame"""
        self.image_past_window_size = image_past_window_size
        self.image_future_window_size = image_future_window_size
        self.action_past_window_size = action_past_window_size
        self.action_future_window_size = action_future_window_size
        self.dataset_name = dataset_name
        self.augmentation = augmentation
        self.normalization = normalization
        self.load_images = load_images
        self.data_type = data_type

        self.data_statistics = None
        self.processor = processor
        self.action_type = action_type
        self.rel_mode = rel_mode # 'step'
        training_path = None
        assert action_type == 'angle' and use_rel == False and rel_mode == 'step', "Please recalculate the statistics and update the path here with other action representations."
        if dataset_name == 'ego4d_cooking_and_cleaning':
            annotation_file = os.path.join(dataset_folder, "Annotation/ego4d_cooking_and_cleaning/episode_frame_index.npz")
            label_folder = os.path.join(dataset_folder, "Annotation/ego4d_cooking_and_cleaning/episodic_annotations")
            statistics_path = os.path.join(dataset_folder, "Annotation/statistics/ego4d_angle_statistics.json")
            video_root = os.path.join(dataset_folder, 'Video/Ego4D_root')
        elif dataset_name == 'egoexo4d':
            annotation_file = os.path.join(dataset_folder, "Annotation/egoexo4d/episode_frame_index.npz")
            label_folder = os.path.join(dataset_folder, "Annotation/egoexo4d/episodic_annotations")
            statistics_path = os.path.join(dataset_folder, "Annotation/statistics/egoexo4d_angle_statistics.json")
            video_root = os.path.join(dataset_folder, 'Video/EgoExo4D_root')
        elif dataset_name == 'epic':
            annotation_file = os.path.join(dataset_folder, "Annotation/epic/episode_frame_index.npz")
            label_folder = os.path.join(dataset_folder, "Annotation/epic/episodic_annotations")
            statistics_path = os.path.join(dataset_folder, "Annotation/statistics/epic_angle_statistics.json")
            video_root = os.path.join(dataset_folder, 'Video/Epic-Kitchen_root')
        elif dataset_name == 'ssv2':
            annotation_file = os.path.join(dataset_folder, "Annotation/ssv2/episode_frame_index.npz")
            label_folder = os.path.join(dataset_folder, "Annotation/ssv2/episodic_annotations")
            statistics_path = os.path.join(dataset_folder, "Annotation/statistics/ssv2_angle_statistics.json")
            video_root = os.path.join(dataset_folder, 'Video/Somethingsomething-v2_root')
        elif dataset_name == 'ego4d_other':
            annotation_file = os.path.join(dataset_folder, "Annotation/ego4d_cooking_and_cleaning/episode_frame_index.npz")
            label_folder = os.path.join(dataset_folder, "Annotation/ego4d_cooking_and_cleaning/episodic_annotations")
            statistics_path = os.path.join(dataset_folder, "Annotation/statistics/ego4d_angle_statistics.json")
            video_root = os.path.join(dataset_folder, 'Video/Ego4D_root')
        elif dataset_name == 'robo_dataset':
            root_dir = os.path.join(dataset_folder, "TeleData")
            statistics_path = os.path.join(dataset_folder, "teledata_statistics.json")
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")
            
        # Warn if statistics file is missing but images are to be loaded
        if statistics_path is None or not os.path.exists(statistics_path):
            if load_images:
                print(f"Warning: statistics file '{statistics_path}' does not exist. Please calculate statistics first if you plan to train a model.")
            else:
                statistics_path = None  # Allow None when calculating statistics only

        if data_type == 'human':
            self.episodic_dataset_core = EpisodicDatasetCore(
                video_root=video_root, 
                annotation_file=annotation_file, 
                label_folder=label_folder, 
                training_path=training_path, 
                statistics_path=statistics_path,
                augmentation=augmentation, 
                flip_augmentation=flip_augmentation, 
                set_none_ratio=set_none_ratio, 
                action_type=action_type, 
                use_rel=use_rel,
                clip_len=clip_len,
                state_mask_prob=state_mask_prob,
                action_past_window_size=self.action_past_window_size,
                action_future_window_size=self.action_future_window_size,
                image_past_window_size=self.image_past_window_size,
                image_future_window_size=self.image_future_window_size,
                rel_mode=self.rel_mode,  # 'step'
                load_images=self.load_images
            )
        else:
            self.episodic_dataset_core = RoboDatasetCore(
                root_dir=root_dir,
                statistics_path=statistics_path,
                action_past_window_size=self.action_past_window_size,
                action_future_window_size=self.action_future_window_size,
                image_past_window_size=self.image_past_window_size,
                image_future_window_size=self.image_future_window_size,
                load_images=self.load_images
            )

        self._length = len(self.episodic_dataset_core)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):

        sample = self.episodic_dataset_core.__getitem__(idx)
        sample = self.episodic_dataset_core.transform_trajectory(sample, self.normalization) if self.load_images else sample
        return self.post_transform(sample) if self.load_images else sample

    def post_transform(self, data):
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        img = data["image_list"][-1]
        img = Image.fromarray(img)
        lang = data["instruction"]
        imgs = []
        imgs.append(img)
        # can be modified to multiple images
        # for raw in image_list:
        #     img = Image.fromarray(raw)
        #     imgs.append(img)
        # # imgs = [img0, img1, img3]
        lang = '<image>' * len(imgs) + lang
        model_inputs = self.processor(text=lang, images=imgs, return_tensors="pt").to(torch.float32)
        image_mask = torch.tensor(np.asarray(data["image_mask"]), dtype=torch.bool)
        input_ids = model_inputs["input_ids"]
        pixel_values = model_inputs["pixel_values"]
        input_ids = input_ids.squeeze(0) 

        # pixel_values.shape need to be [num_img, 3, 224, 224]

        fov = torch.tensor(data["fov"], dtype=torch.float32)
        intrinsics = torch.tensor(data["intrinsics"], dtype=torch.float32)
        labels = None

        return_dict = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            dataset_name=self.dataset_name,
            actions=data["action_list"],
            action_masks=data["action_mask"],
            current_state_mask=data["current_state_mask"],
            current_state=data["current_state"],
            fov = fov,
            intrinsics = intrinsics,
        )

        return return_dict

    def find_index_bf(self, idx):
        episode_id = 0
        for i in range(len(self.episodic_lengths)):
            if idx < self.episodic_lengths[i]:
                episode_id = i
                break
            idx -= self.episodic_lengths[i]
        return episode_id, idx

class MultipleWeightedDataset(Dataset):
    def __init__(self, datasets, weights=None):
        self.datasets = datasets
        if weights is None:
            weights = [1] * len(datasets)
        self.weights = weights
        self._length = sum([len(dataset) for dataset in datasets])
        self._accumulate_lengths = [0]
        for dataset in datasets:
            self._accumulate_lengths.append(self._accumulate_lengths[-1] + len(dataset))
        print("Dataset lengths:", [len(dataset) for dataset in datasets])
    
    def __len__(self):
        return self._length
    
    def __getitem__(self, index):
        if isinstance(index, int):
            dataset_id = bisect.bisect_right(self._accumulate_lengths, index) - 1
            idx = index - self._accumulate_lengths[dataset_id]
            assert 0 <= idx < len(self.datasets[dataset_id]), f"Index {idx} out of range for dataset {dataset_id} with length {len(self.datasets[dataset_id])}"
            return self.datasets[dataset_id][idx]
        elif isinstance(index, tuple):
            dataset_id, idx = index
            assert 0 <= idx < len(self.datasets[dataset_id]), f"Index {idx} out of range for dataset {dataset_id} with length {len(self.datasets[dataset_id])}"
            return self.datasets[dataset_id][idx]

    @staticmethod
    def save_mixed_dataset_statistics(dataset_folder, data_mix, action_type, data_statistics):
        # Convert numpy arrays to lists for JSON serialization
        data_statistics_json = {
            'dataset_name': f"{data_mix}_{action_type}",
            'state_left': {
            'mean': data_statistics['state_left_mean'].tolist(),
            'std': data_statistics['state_left_std'].tolist()
            },
            'action_left': {
            'mean': data_statistics['action_left_mean'].tolist(),
            'std': data_statistics['action_left_std'].tolist()
            },
            'state_right': {
            'mean': data_statistics['state_right_mean'].tolist(),
            'std': data_statistics['state_right_std'].tolist()
            },
            'action_right': {
            'mean': data_statistics['action_right_mean'].tolist(),
            'std': data_statistics['action_right_std'].tolist()
            }
        }
        # Save to local file
        with open(os.path.join(dataset_folder, f"{data_mix}_{action_type}_weighted_statistics.json"), "w") as f:
            json.dump(data_statistics_json, f, indent=2)

    @staticmethod
    def weighted_average_statistics(datasets, weights):
        """
        Calculate weighted average of data_statistics across multiple datasets.

        Args:
            datasets: List of datasets, each containing dataset.episodic_dataset_core.data_statistics
            weights: List of corresponding weights for each dataset
        
        Returns:
            Dictionary of merged statistics with weighted averages
        """
        assert len(datasets) == len(weights), "datasets and weights must have the same length"

        # Calculate raw weights = len(dataset) * weights[i]
        raw_weights = np.array([
            len(ds.episodic_dataset_core) * w
            for ds, w in zip(datasets, weights)
        ], dtype=np.float64)

        # Normalize to ensure total weight equals 1
        norm_weights = raw_weights / raw_weights.sum()

        # Get all keys from the first dataset
        keys = datasets[0].episodic_dataset_core.data_statistics.keys()

        # Store merged results
        merged_statistics = {}

        for key in keys:
            # Initialize as None
            total = None
            for ds, weight in zip(datasets, norm_weights):
                value = ds.episodic_dataset_core.data_statistics[key]
                if total is None:
                    total = weight * value
                else:
                    total += weight * value
            merged_statistics[key] = total

        return merged_statistics

    @classmethod
    def load_datasets(cls, dataset_folder, data_mix,
                      image_past_window_size=0, image_future_window_size=0, action_past_window_size=0, action_future_window_size=0,
                      augmentation=False, normalization=True, processor = None, flip_augmentation=1.0, set_none_ratio=0.0,
                      action_type="angle", use_rel=False, rel_mode='step', clip_len=None, state_mask_prob=0.1):
        dataset_weight_list = []
        if data_mix in HAND_MIXTURES:
            dataset_weight_list = HAND_MIXTURES[data_mix]
        else:
            dataset_weight_list = [(data_mix, 1)]
        
        datasets = []
        weights = []
        for dataset_name, weight in dataset_weight_list:
            print("Loading dataset:", dataset_name)
            # Auto-detect data_type based on dataset_name
            if dataset_name.startswith('robo_'):
                data_type = 'robot'
            else:
                data_type = 'human'
            dataset = FrameDataset(os.path.join(dataset_folder), dataset_name, 
                                   image_past_window_size, image_future_window_size, 
                                   action_past_window_size, action_future_window_size,
                                   augmentation, normalization, processor, flip_augmentation, set_none_ratio,
                                   action_type, use_rel, rel_mode, load_images=True, data_type=data_type, clip_len=clip_len, state_mask_prob=state_mask_prob)
            datasets.append(dataset)
            weights.append(weight)
        data_statistics = cls.weighted_average_statistics(datasets, weights)
        cls.save_mixed_dataset_statistics(dataset_folder, data_mix, action_type, data_statistics)
        for dataset in datasets:
            dataset.episodic_dataset_core.set_global_data_statistics(data_statistics)
        return cls(datasets, weights)

    @classmethod
    def self_check(cls, dataset):
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        for i, idx in enumerate(indices):
            if i % 1000 == 0:
                print(f"Checking {i} / {len(indices)}")
            item = dataset[idx]
            if item is None:
                print(f"Dataset {dataset} item {idx} is None")
        print(f"Dataset {dataset} is valid")

class MultipleDatasetWeightedDistributedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, 
                 dataset, batch_size,  
                 drop_last: bool = False, 
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 shuffle: bool = True,
                 seed: int = 0):
        self.dataset = dataset
        self.epoch = 0
        self.step = 0
        self.drop_last = drop_last
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")
        self.num_replicas = num_replicas
        self.rank = rank
        print(f"Creating Distributed Batch Sampler with rank {rank} and num_replicas {num_replicas}")
        self.prepare_sample_size()
    
    def prepare_sample_size(self):
        self._dataset_lengths = [len(dataset) for dataset in self.dataset.datasets]
        self.weights = self.dataset.weights
        self._sample_size = [int(weight * length) for weight, length in zip(self.weights, self._dataset_lengths)]
        self.total_size = sum(self._sample_size)
        iter_size = self.batch_size * self.num_replicas
        if self.drop_last:
            self.num_iters = self.total_size // iter_size
        else:
            self.num_iters = (self.total_size + iter_size - 1) // iter_size
        self.num_samples = self.num_iters * iter_size
    
    def create_indices(self, dataset_id, epoch):
        indices = list(range(self._dataset_lengths[dataset_id]))
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed((self.seed + epoch) * len(self.dataset.datasets) + dataset_id)
            indices = torch.randperm(len(indices), generator=g).tolist()
        return indices
    
    def create_indices_range(self, dataset_id, start, end):
        dataset_length = self._dataset_lengths[dataset_id]
        start_epoch = start // dataset_length
        start_idx = start % dataset_length
        end_epoch = end // dataset_length
        end_idx = end % dataset_length
        if end_idx == 0:
            end_epoch -= 1
            end_idx = dataset_length
        assert start_epoch <= end_epoch
        indices = []
        for epoch in range(start_epoch, end_epoch + 1):
            epoch_indices = self.create_indices(dataset_id, epoch)
            if epoch == start_epoch and epoch == end_epoch:
                epoch_indices = epoch_indices[start_idx:end_idx]
            elif epoch == start_epoch:
                epoch_indices = epoch_indices[start_idx:]
            elif epoch == end_epoch:
                epoch_indices = epoch_indices[:end_idx]
            else:
                epoch_indices = epoch_indices
            indices.extend(epoch_indices)
        assert len(indices) == end - start
        return indices
    
    def shuffle_dataset_indices(self, indices):
        dataset_id = []
        for i in range(len(indices)):
            dataset_id.extend([i] * len(indices[i]))
        rng = random.Random(self.seed + self.epoch)
        rng.shuffle(dataset_id)
        dataset_index_list = []
        dataset_count = [0] * len(self.dataset.datasets)
        for i in range(len(dataset_id)):
            di = dataset_id[i]
            si = indices[di][dataset_count[di]]
            dataset_index_list.append((di, si))
            dataset_count[di] += 1
        for i in range(len(dataset_count)):
            assert dataset_count[i] == len(indices[i])
        return dataset_index_list
            

    def prepare_indices(self):
        indices = []
        for i in range(len(self.dataset.datasets)):
            start = self.epoch * self._sample_size[i]
            end = (self.epoch + 1) * self._sample_size[i]
            indices.append(self.create_indices_range(i, start, end))
        dataset_index_list = self.shuffle_dataset_indices(indices)
        return dataset_index_list
    
    def dataset_statistics(self, dataset_index_list):
        dataset_count = [0] * len(self.dataset.datasets)
        for di, si in dataset_index_list:
            dataset_count[di] += 1
        s = sum(dataset_count)
        for i in range(len(dataset_count)):
            print(f"Dataset {i} count: {dataset_count[i]}, ratio: {dataset_count[i] / s:.4f}")

    def __iter__(self):
        dataset_index_list = self.prepare_indices()

        if not self.drop_last:
            padding_size = self.num_samples - len(dataset_index_list)
            if padding_size <= len(dataset_index_list):
                dataset_index_list += dataset_index_list[:padding_size]
            else:
                dataset_index_list += (dataset_index_list * math.ceil(padding_size / len(dataset_index_list)))[:padding_size]
        else:
            dataset_index_list = dataset_index_list[:self.num_samples]
        assert len(dataset_index_list) == self.num_samples

        dataset_index_list = dataset_index_list[self.rank:self.num_samples:self.num_replicas]
        assert len(dataset_index_list) == self.num_iters * self.batch_size

        self.dataset_statistics(dataset_index_list)
        print(f"Batch Sampler in rank {self.rank} start from {self.step} to {self.num_iters} at epoch {self.epoch}")
        print(f"First batch: {dataset_index_list[self.step * self.batch_size:(self.step + 1) * self.batch_size]}")
        for i in range(self.step, self.num_iters):
            # print("Iterating", i, dataset_index_list[i * self.batch_size:(i + 1) * self.batch_size])
            yield dataset_index_list[i * self.batch_size:(i + 1) * self.batch_size]
        self.set_epoch(self.epoch + 1)
        print("Epoch", self.epoch, "completed")

    def __len__(self):
        return self.num_iters

    def set_epoch(self, epoch, step=0):
        assert epoch >= 0
        step = step % self.num_iters
        assert 0 <= step < self.num_iters
        self.epoch = epoch
        self.step = step