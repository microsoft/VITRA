
import numpy as np
import json
from tqdm import tqdm
import argparse
import os

from torch.utils.data import DataLoader
from vitra.utils.config_utils import load_config
from vitra.datasets.dataset import FrameDataset


def compute_statistics(dataset, num_workers=16, batch_size=128, save_folder='./'):
    """
    Compute mean and standard deviation of the 'state' in the dataset using multi-threading.

    Args:
        dataset (FrameDataset): The dataset to process.
        num_workers (int): Number of worker threads for DataLoader.
        batch_size (int): Batch size for DataLoader.
        save_folder (str): Folder to save the computed statistics.

    Returns:
        dict: A dictionary containing the mean and standard deviation of the state.
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    train_num = 0

    state_left_list = []
    state_right_list = []
    action_left_list = []
    action_right_list = []
    for batch in tqdm(dataloader, desc="Processing dataset"):
        current_state = batch['current_state'].numpy()
        current_state_mask = batch['current_state_mask'].numpy()
        action_list = batch['action_list'].numpy()
        action_mask = batch['action_mask'].numpy()
        dim = action_list.shape[-1] // 2
        half_state_dim = current_state.shape[-1] // 2
        action_left_list.extend(action_list[action_mask[:, :, 0], :dim].tolist())
        action_right_list.extend(action_list[action_mask[:, :, 1], dim:].tolist())
        state_left_list.extend(current_state[current_state_mask[:, 0], :half_state_dim].tolist())
        state_right_list.extend(current_state[current_state_mask[:, 1], half_state_dim:].tolist())
        train_num += batch['current_state'].shape[0]
    
    del dataloader
    action_right_np = np.array(action_right_list)
    state_right_np = np.array(state_right_list)
    action_left_np = np.array(action_left_list)
    state_left_np = np.array(state_left_list)
    action_left_mean = np.mean(action_left_np, axis=0)
    action_left_std = np.std(action_left_np, axis=0)
    state_left_mean = np.mean(state_left_np, axis=0)
    state_left_std = np.std(state_left_np, axis=0)
    action_right_mean = np.mean(action_right_np, axis=0)
    action_right_std = np.std(action_right_np, axis=0)
    state_right_mean = np.mean(state_right_np, axis=0)
    state_right_std = np.std(state_right_np, axis=0)

    my_dict = {
        'dataset_name': f"{dataset.dataset_name}_{dataset.action_type}_statistics.json",
        'state_left': { 'mean': state_left_mean.tolist(), 'std': state_left_std.tolist() },
        'action_left': { 'mean': action_left_mean.tolist(), 'std': action_left_std.tolist() },
        'state_right': { 'mean': state_right_mean.tolist(), 'std': state_right_std.tolist() },
        'action_right': { 'mean': action_right_mean.tolist(), 'std': action_right_std.tolist() },
        'num_traj': train_num
        }
    os.makedirs(save_folder, exist_ok=True)
    with open(os.path.join(save_folder, f"{dataset.dataset_name}_{dataset.action_type}_statistics.json"), "w") as outfile:
        json.dump(my_dict, outfile, indent=4)
    print(f"Statistics saved for dataset '{dataset.dataset_name}' at '{os.path.join(save_folder, f'{dataset.dataset_name}_{dataset.action_type}_statistics.json')}'")

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Calculate statistics for hand manipulation dataset')
 
    # Dataset parameters
    parser.add_argument('--dataset_folder', type=str, 
                        default="/home/t-qixiuli/eai_blob/hand_pretrain_prepare",
                        help='Path to the dataset folder')
    parser.add_argument('--dataset_name', type=str, default='all',
                        help='Name of the dataset (ego4d, egoexo4d, epic, ssv2, ego4d_beyond, all)')

    # Augmentation parameters
    parser.add_argument('--augmentation', action='store_true', default=True,
                        help='Enable data augmentation')
    parser.add_argument('--flip_augmentation', type=float, default=1.0,
                        help='Probability of flip augmentation (0.0 to 1.0)')
    parser.add_argument('--set_none_ratio', type=float, default=0.0,
                        help='Ratio for setting None in augmentation')

    # Action parameters
    # Note that action_future_window_size can be set to 0 if rel_mode is 'step'.
    # Then you can use the same statistics for varying action_future_window_size during training/inference.
    # They will be very close for every single step.
    parser.add_argument('--action_future_window_size', type=int, default=0,
                        help='Future window size for actions, set to be 0 if rel_mode is \'step\'')
    parser.add_argument('--action_type', type=str, default='angle', 
                        choices=['angle', 'keypoints'],
                        help='Type of action representation')
    parser.add_argument('--rel_mode', type=str, default='step',
                        choices=['step', 'anchor'],
                        help='Relative mode for actions')
    parser.add_argument('--use_rel', action='store_true', default=False,
                        help='Use relative MANO pose')
    # Other parameters
    parser.add_argument('--config', type=str, default='configs/debug.json',
                        help='Path to config file')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of DataLoader workers')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for processing')
    parser.add_argument('--save_folder', type=str, default='./statistics/',
                        help='Folder to save computed statistics')

    args = parser.parse_args()
    if args.dataset_name == 'all':
        for dataset_name in ['ego4d_cooking_and_cleaning', 'egoexo4d', 'epic', 'ssv2', 'ego4d_other']:
            print(f"Calculating statistics for dataset: {dataset_name}")
            dataset = FrameDataset(
                dataset_folder=args.dataset_folder, 
                dataset_name=dataset_name, 
                action_past_window_size=0, 
                action_future_window_size=args.action_future_window_size,
                augmentation=args.augmentation,
                flip_augmentation=args.flip_augmentation,
                set_none_ratio=0.0,
                action_type=args.action_type, 
                use_rel=args.use_rel,
                rel_mode=args.rel_mode, 
                load_images=False,
                state_mask_prob=0.0,
            )
            compute_statistics(dataset, args.num_workers, args.batch_size, args.save_folder)
    else:
        print(f"Calculating statistics for dataset: {args.dataset_name}")
        dataset = FrameDataset(
            dataset_folder=args.dataset_folder, 
            dataset_name=args.dataset_name, 
            action_past_window_size=0, 
            action_future_window_size=args.action_future_window_size,
            augmentation=args.augmentation,
            flip_augmentation=args.flip_augmentation,
            set_none_ratio=0.0,
            action_type=args.action_type, 
            use_rel=args.use_rel,
            rel_mode=args.rel_mode, 
            load_images=False,
        )
        compute_statistics(dataset, args.num_workers, args.batch_size, args.save_folder)