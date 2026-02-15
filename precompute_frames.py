#!/usr/bin/env python3
"""Precompute resized+background-subtracted frames for faster training.

Generates per-rally .npy files matching the __getitem__ processing in dataset.py.
This avoids redundant PIL.resize + background subtraction on every epoch.

Usage:
    python precompute_frames.py --height 576 --width 1024 --bg_mode subtract_concat
    python precompute_frames.py --height 288 --width 512 --bg_mode subtract_concat
    python precompute_frames.py --height 576 --width 1024 --workers 4  # parallel
"""

import os
import sys
import argparse
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


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
    median_file = os.path.join(rally_dir, 'median.npz')
    if not os.path.exists(median_file):
        match_dir = os.path.dirname(os.path.dirname(rally_dir))
        median_file = os.path.join(match_dir, 'median.npz')
    if os.path.exists(median_file):
        return np.load(median_file)['median']
    return None


def _process_frame(args):
    """Process a single frame (used by thread pool)."""
    fpath, height, width, bg_mode, median = args
    fn = int(os.path.basename(fpath).split('.')[0])
    img = Image.open(fpath)

    if bg_mode == 'subtract' and median is not None:
        diff = Image.fromarray(
            np.sum(np.absolute(np.array(img).astype(np.int16) - median.astype(np.int16)), 2)
            .clip(0, 255).astype('uint8'))
        diff = np.array(diff.resize((width, height)))
        return fn, diff[np.newaxis]  # (1, H, W)
    elif bg_mode == 'subtract_concat' and median is not None:
        img_arr = np.array(img)
        diff = np.sum(
            np.absolute(img_arr.astype(np.int16) - median.astype(np.int16)), 2
        ).clip(0, 255).astype('uint8')
        diff_resized = np.array(Image.fromarray(diff).resize((width, height)))
        img_resized = np.array(img.resize((width, height)))
        channels = np.concatenate([
            np.moveaxis(img_resized, -1, 0),  # (3, H, W)
            diff_resized[np.newaxis],           # (1, H, W)
        ], axis=0)  # (4, H, W)
        return fn, channels
    else:
        img_resized = np.array(img.resize((width, height)))
        return fn, np.moveaxis(img_resized, -1, 0)  # (3, H, W)


def precompute_rally(rally_dir, height, width, bg_mode, io_threads=4):
    """Precompute all frames for one rally. Returns (N, C, H, W) uint8 array."""
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
        channels = 4
    elif bg_mode == 'subtract':
        channels = 1
    else:
        channels = 3

    frames = np.zeros((n_frames, channels, height, width), dtype=np.uint8)

    # Parallel frame loading + processing via threads (I/O bound)
    task_args = [(f, height, width, bg_mode, median) for f in frame_files]

    if io_threads > 1 and len(frame_files) > 1:
        with ThreadPoolExecutor(max_workers=io_threads) as pool:
            for fn, data in pool.map(_process_frame, task_args):
                frames[fn] = data
    else:
        for args in task_args:
            fn, data = _process_frame(args)
            frames[fn] = data

    return frames


def _process_rally(args):
    """Process one rally end-to-end (used by process pool)."""
    rally_dir, out_dir, data_dir, height, width, bg_mode, io_threads = args
    rel_path = os.path.relpath(rally_dir, data_dir)
    safe_name = rel_path.replace(os.sep, '_')
    npy_path = os.path.join(out_dir, f'{safe_name}.npy')

    if os.path.exists(npy_path):
        return npy_path, True  # skipped

    frames = precompute_rally(rally_dir, height, width, bg_mode, io_threads)
    if frames is not None:
        np.save(npy_path, frames)
    return npy_path, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--bg_mode', type=str, default='subtract_concat',
                        choices=['', 'subtract', 'subtract_concat', 'concat'])
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'val'])
    parser.add_argument('--workers', type=int, default=1,
                        help='number of parallel rally workers (processes)')
    parser.add_argument('--io_threads', type=int, default=4,
                        help='threads per rally for frame I/O')
    args = parser.parse_args()

    res_suffix = f'_{args.height}x{args.width}'
    out_dir = os.path.join(args.data_dir, 'precomputed',
                           (args.bg_mode or 'raw') + res_suffix)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    for split in args.splits:
        rally_dirs = get_rally_dirs(args.data_dir, split)
        print(f"\n{split}: {len(rally_dirs)} rallies")

        if args.workers > 1:
            # Multi-process: parallelize across rallies
            task_args = [
                (rd, out_dir, args.data_dir, args.height, args.width,
                 args.bg_mode, args.io_threads)
                for rd in rally_dirs
            ]
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for npy_path, skipped in tqdm(
                    pool.map(_process_rally, task_args),
                    total=len(rally_dirs), desc=f"  {split}"
                ):
                    pass
        else:
            # Single process with threaded I/O per rally
            for rally_dir in tqdm(rally_dirs, desc=f"  {split}"):
                rel_path = os.path.relpath(rally_dir, args.data_dir)
                safe_name = rel_path.replace(os.sep, '_')
                npy_path = os.path.join(out_dir, f'{safe_name}.npy')

                if os.path.exists(npy_path):
                    continue

                frames = precompute_rally(
                    rally_dir, args.height, args.width, args.bg_mode,
                    args.io_threads)
                if frames is not None:
                    np.save(npy_path, frames)

    # Summary
    files = glob.glob(os.path.join(out_dir, '*.npy'))
    total_bytes = sum(os.path.getsize(f) for f in files)
    print(f"\nDone: {len(files)} files, {total_bytes/1024/1024/1024:.1f} GB")


if __name__ == '__main__':
    main()
