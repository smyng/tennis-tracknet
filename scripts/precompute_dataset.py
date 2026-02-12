#!/usr/bin/env python3
"""
Precompute processed frames for TrackNet training.

Converts raw PNGs + background subtraction into pre-processed NPZ files
per rally, eliminating redundant image loading/resizing/subtraction during
training. Speeds up data loading by ~50-100x.

Usage:
    python precompute_dataset.py --split train
    python precompute_dataset.py --split val
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add project root to path when running from scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.general import get_rally_dirs, HEIGHT, WIDTH, IMG_FORMAT


def precompute_rally(rally_dir, data_dir, bg_mode='subtract_concat'):
    """
    Precompute all processed frames for a single rally.

    Saves resized frames + background subtraction as a single NPZ file,
    so __getitem__ can do a fast array lookup instead of disk I/O.

    Args:
        rally_dir: Path like 'train/match1/frame/1_01_00'
        data_dir: Root data directory
        bg_mode: Background mode ('subtract_concat', 'subtract', etc.)

    Returns:
        Path to saved NPZ file
    """
    full_rally_dir = os.path.join(data_dir, rally_dir)

    # Find all frame files
    frame_files = sorted([
        f for f in os.listdir(full_rally_dir)
        if f.endswith(f'.{IMG_FORMAT}') and f[0].isdigit()
    ], key=lambda f: int(f.split('.')[0]))

    if not frame_files:
        return None

    # Load median background
    median_img = None
    if bg_mode:
        # Parse match_dir and rally_id from rally_dir
        # rally_dir format: 'train/match1/frame/1_01_00'
        parts = rally_dir.split('/')
        split = parts[0]
        match = parts[1]
        rally_id = parts[3]
        match_dir = os.path.join(data_dir, split, match)

        median_file = os.path.join(match_dir, 'frame', rally_id, 'median.npz')
        if not os.path.exists(median_file):
            median_file = os.path.join(match_dir, 'median.npz')

        if os.path.exists(median_file):
            median_img = np.load(median_file)['median']
        else:
            print(f"  Warning: No median file for {rally_dir}, skipping bg subtraction")
            bg_mode = ''

    # Process all frames
    processed_frames = []  # Will be (N, C, H, W) where C=4 for subtract_concat

    for fname in frame_files:
        img = Image.open(os.path.join(full_rally_dir, fname))

        if bg_mode == 'subtract_concat':
            # Background subtracted channel
            diff_img = Image.fromarray(
                np.sum(np.absolute(np.array(img) - median_img), 2).astype('uint8')
            )
            diff_img = np.array(diff_img.resize(size=(WIDTH, HEIGHT)))
            diff_img = diff_img.reshape(1, HEIGHT, WIDTH)

            # RGB channels
            img = np.array(img.resize(size=(WIDTH, HEIGHT)))
            img = np.moveaxis(img, -1, 0)  # (3, H, W)

            # Concatenate: (4, H, W)
            frame = np.concatenate((img, diff_img), axis=0)

        elif bg_mode == 'subtract':
            diff_img = Image.fromarray(
                np.sum(np.absolute(np.array(img) - median_img), 2).astype('uint8')
            )
            diff_img = np.array(diff_img.resize(size=(WIDTH, HEIGHT)))
            frame = diff_img.reshape(1, HEIGHT, WIDTH)

        else:
            img = np.array(img.resize(size=(WIDTH, HEIGHT)))
            frame = np.moveaxis(img, -1, 0)  # (3, H, W)

        processed_frames.append(frame)

    # Stack into single array: (N_frames, C, H, W)
    frames_array = np.stack(processed_frames).astype(np.uint8)

    # Save as compressed NPZ
    output_dir = os.path.join(data_dir, 'precomputed', bg_mode or 'raw')
    os.makedirs(output_dir, exist_ok=True)

    # Use rally path as filename (replace / with _)
    safe_name = rally_dir.replace('/', '_')
    output_file = os.path.join(output_dir, f'{safe_name}.npz')

    np.savez_compressed(output_file, frames=frames_array)

    return output_file


def main():
    parser = argparse.ArgumentParser(description='Precompute dataset frames')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--bg_mode', type=str, default='subtract_concat',
                        choices=['', 'subtract', 'subtract_concat', 'concat'])
    args = parser.parse_args()

    rally_dirs = get_rally_dirs(args.data_dir, args.split)
    print(f"Precomputing {len(rally_dirs)} rallies for split='{args.split}', bg_mode='{args.bg_mode}'")
    print(f"Output: {args.data_dir}/precomputed/{args.bg_mode}/")
    print()

    total_frames = 0
    total_bytes = 0

    for rally_dir in tqdm(rally_dirs, desc='Processing rallies'):
        output_file = precompute_rally(rally_dir, args.data_dir, args.bg_mode)
        if output_file:
            size = os.path.getsize(output_file)
            total_bytes += size
            data = np.load(output_file)
            total_frames += len(data['frames'])

    print()
    print(f"Done! Precomputed {total_frames} frames")
    print(f"Total size: {total_bytes / (1024**3):.2f} GB")
    print()
    print("To use during training, the dataset needs to be patched to load from")
    print("precomputed NPZ files instead of raw PNGs. Run:")
    print(f"  python patch_dataset_precomputed.py")


if __name__ == '__main__':
    main()
