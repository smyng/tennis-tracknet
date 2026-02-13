#!/usr/bin/env python3
"""Precompute resized+background-subtracted frames for faster training.

Generates per-rally .npy files matching the __getitem__ processing in dataset.py.
This avoids redundant PIL.resize + background subtraction on every epoch.

Usage:
    python precompute_frames.py --height 576 --width 1024 --bg_mode subtract_concat
    python precompute_frames.py --height 288 --width 512 --bg_mode subtract_concat
"""

import os
import sys
import argparse
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm


def get_rally_dirs(data_dir, split):
    """Find all rally directories for a split."""
    split_dir = os.path.join(data_dir, split)
    rallies = []
    for match_dir in sorted(glob.glob(os.path.join(split_dir, 'match*'))):
        frame_dir = os.path.join(match_dir, 'frame')
        if not os.path.isdir(frame_dir):
            continue
        for rally_dir in sorted(os.listdir(frame_dir)):
            full_path = os.path.join(frame_dir, rally_dir)
            if os.path.isdir(full_path):
                rallies.append(full_path)
    return rallies


def load_median(rally_dir):
    """Load the median background image for a rally."""
    # Check rally dir first, then parent match dir
    median_file = os.path.join(rally_dir, 'median.npz')
    if not os.path.exists(median_file):
        match_dir = os.path.dirname(os.path.dirname(rally_dir))
        median_file = os.path.join(match_dir, 'median.npz')
    if os.path.exists(median_file):
        return np.load(median_file)['median']
    return None


def precompute_rally(rally_dir, height, width, bg_mode):
    """Precompute all frames for one rally. Returns (N, C, H, W) uint8 array."""
    # Find all frame files
    frame_files = sorted(
        glob.glob(os.path.join(rally_dir, '*.png')) +
        glob.glob(os.path.join(rally_dir, '*.jpg')),
        key=lambda f: int(os.path.basename(f).split('.')[0])
    )
    if not frame_files:
        return None

    n_frames = int(os.path.basename(frame_files[-1]).split('.')[0]) + 1

    median = load_median(rally_dir) if bg_mode else None

    if bg_mode == 'subtract_concat':
        channels = 4  # RGB + diff
    elif bg_mode == 'subtract':
        channels = 1  # diff only
    elif bg_mode == 'concat':
        channels = 3  # RGB (median prepended separately)
    else:
        channels = 3  # RGB

    frames = np.zeros((n_frames, channels, height, width), dtype=np.uint8)

    for fpath in frame_files:
        fn = int(os.path.basename(fpath).split('.')[0])
        img = Image.open(fpath)

        if bg_mode == 'subtract' and median is not None:
            diff = Image.fromarray(
                np.sum(np.absolute(np.array(img).astype(np.int16) - median.astype(np.int16)), 2)
                .clip(0, 255).astype('uint8'))
            diff = np.array(diff.resize((width, height)))
            frames[fn, 0] = diff
        elif bg_mode == 'subtract_concat' and median is not None:
            img_arr = np.array(img)
            diff = np.sum(
                np.absolute(img_arr.astype(np.int16) - median.astype(np.int16)), 2
            ).clip(0, 255).astype('uint8')
            diff_resized = np.array(Image.fromarray(diff).resize((width, height)))
            img_resized = np.array(img.resize((width, height)))
            # channels: R, G, B, diff
            frames[fn, :3] = np.moveaxis(img_resized, -1, 0)
            frames[fn, 3] = diff_resized
        else:
            img_resized = np.array(img.resize((width, height)))
            frames[fn, :3] = np.moveaxis(img_resized, -1, 0)

    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--bg_mode', type=str, default='subtract_concat',
                        choices=['', 'subtract', 'subtract_concat', 'concat'])
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'val'])
    args = parser.parse_args()

    res_suffix = f'_{args.height}x{args.width}'
    out_dir = os.path.join(args.data_dir, 'precomputed',
                           (args.bg_mode or 'raw') + res_suffix)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    for split in args.splits:
        rally_dirs = get_rally_dirs(args.data_dir, split)
        print(f"\n{split}: {len(rally_dirs)} rallies")

        for rally_dir in tqdm(rally_dirs, desc=f"  {split}"):
            # Build safe filename matching dataset.py convention
            rel_path = os.path.relpath(rally_dir, args.data_dir)
            safe_name = rel_path.replace(os.sep, '_')
            npy_path = os.path.join(out_dir, f'{safe_name}.npy')

            if os.path.exists(npy_path):
                continue  # skip already computed

            frames = precompute_rally(
                rally_dir, args.height, args.width, args.bg_mode)
            if frames is not None:
                np.save(npy_path, frames)

    # Summary
    files = glob.glob(os.path.join(out_dir, '*.npy'))
    total_bytes = sum(os.path.getsize(f) for f in files)
    print(f"\nDone: {len(files)} files, {total_bytes/1024/1024/1024:.1f} GB")


if __name__ == '__main__':
    main()
