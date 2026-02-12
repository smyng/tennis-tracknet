"""
Convert TrackNet v1 Tennis Dataset to TrackNetV3 Format

TrackNet v1 format:
    data/game{1-10}/Clip{N}/
        - Label.csv (columns: file name, visibility, x-coordinate, y-coordinate, status)
        - {frame_id}.jpg

TrackNetV3 format:
    data/train/match{1-10}/
        - csv/{rally_id}_ball.csv (columns: Frame, X, Y, Visibility)
        - frame/{rally_id}/{frame_id}.png
"""

import os
import pandas as pd
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm
import shutil


def convert_game_to_match(input_dir: str, output_dir: str, game_id: int, verbose: bool = True):
    """
    Convert a single game directory from v1 to v3 format.

    Args:
        input_dir: Path to v1 dataset (e.g., 'tracknet-dataset/')
        output_dir: Path to v3 dataset (e.g., 'tracknet-v3-tennis/data/')
        game_id: Game number (1-10)
        verbose: Print progress
    """
    game_name = f'game{game_id}'
    match_name = f'match{game_id}'

    game_path = Path(input_dir) / game_name
    if not game_path.exists():
        if verbose:
            print(f"⚠ Warning: {game_path} does not exist, skipping")
        return 0, 0

    # Get all clips in this game
    clips = sorted([d for d in game_path.iterdir() if d.is_dir()])

    total_frames = 0
    total_rallies = 0

    for clip_idx, clip_path in enumerate(clips):
        rally_id = f'{game_id}_{clip_idx+1:02d}_00'

        if verbose:
            print(f"  Processing {game_name}/{clip_path.name} → {match_name}/{rally_id}")

        # Read v1 label file
        label_file = clip_path / 'Label.csv'
        if not label_file.exists():
            if verbose:
                print(f"    ⚠ Warning: {label_file} not found, skipping")
            continue

        labels_v1 = pd.read_csv(label_file)

        # Convert to v3 format
        labels_v3 = pd.DataFrame({
            'Frame': range(len(labels_v1)),
            'X': labels_v1['x-coordinate'],
            'Y': labels_v1['y-coordinate'],
            'Visibility': labels_v1['visibility']
        })

        # Create output directories
        match_dir = Path(output_dir) / 'train' / match_name
        csv_dir = match_dir / 'csv'
        frame_dir = match_dir / 'frame' / rally_id

        csv_dir.mkdir(parents=True, exist_ok=True)
        frame_dir.mkdir(parents=True, exist_ok=True)

        # Save v3 CSV
        csv_file = csv_dir / f'{rally_id}_ball.csv'
        labels_v3.to_csv(csv_file, index=False)

        # Copy/convert frame images
        for idx, row in labels_v1.iterrows():
            frame_name = row['file name']
            src_img = clip_path / frame_name

            # Convert to PNG if JPG (v3 uses PNG by default)
            dst_name = f'{idx}.png'
            dst_img = frame_dir / dst_name

            if src_img.exists():
                # Read and save as PNG
                img = cv2.imread(str(src_img))
                if img is not None:
                    cv2.imwrite(str(dst_img), img)
                    total_frames += 1
                else:
                    if verbose:
                        print(f"    ⚠ Warning: Could not read {src_img}")
            else:
                if verbose:
                    print(f"    ⚠ Warning: {src_img} not found")

        total_rallies += 1

    if verbose:
        print(f"  ✓ {game_name}: {total_rallies} rallies, {total_frames} frames")

    return total_rallies, total_frames


def create_test_split(output_dir: str, test_game_ids: list = [9, 10]):
    """
    Move specified games from train/ to test/ directory.

    Args:
        output_dir: Path to v3 dataset root
        test_game_ids: List of game IDs to use for testing (default: game 9, 10)
    """
    output_path = Path(output_dir)
    train_dir = output_path / 'train'
    test_dir = output_path / 'test'

    test_dir.mkdir(parents=True, exist_ok=True)

    for game_id in test_game_ids:
        match_name = f'match{game_id}'
        src = train_dir / match_name
        dst = test_dir / match_name

        if src.exists():
            shutil.move(str(src), str(dst))
            print(f"  Moved {match_name} to test split")


def generate_background_images(data_dir: str, split: str = 'train', sample_frames: int = 100):
    """
    Generate median background images for each rally.

    Args:
        data_dir: Path to v3 dataset root
        split: 'train' or 'test'
        sample_frames: Number of frames to sample for median calculation
    """
    import numpy as np  # Import here for background generation

    data_path = Path(data_dir) / split
    if not data_path.exists():
        print(f"⚠ Warning: {data_path} does not exist")
        return

    print(f"\nGenerating background images for {split} split...")

    matches = sorted([d for d in data_path.iterdir() if d.is_dir()])

    for match_dir in matches:
        frame_dir = match_dir / 'frame'
        bg_dir = match_dir / 'background'
        bg_dir.mkdir(exist_ok=True)

        rallies = sorted([d for d in frame_dir.iterdir() if d.is_dir()])

        for rally_dir in tqdm(rallies, desc=f"  {match_dir.name}"):
            rally_id = rally_dir.name
            bg_file = bg_dir / f'{rally_id}.png'

            if bg_file.exists():
                continue  # Skip if already exists

            # Get all frames in rally
            frames = sorted(rally_dir.glob('*.png'))

            if len(frames) == 0:
                continue

            # Sample frames evenly
            if len(frames) > sample_frames:
                step = len(frames) // sample_frames
                sampled_frames = frames[::step][:sample_frames]
            else:
                sampled_frames = frames

            # Load frames
            frame_stack = []
            for frame_file in sampled_frames:
                img = cv2.imread(str(frame_file))
                if img is not None:
                    frame_stack.append(img)

            if len(frame_stack) > 0:
                # Compute median background
                frame_stack = np.array(frame_stack)
                bg_image = np.median(frame_stack, axis=0).astype(np.uint8)
                cv2.imwrite(str(bg_file), bg_image)


def main():
    parser = argparse.ArgumentParser(description='Convert TrackNet v1 Tennis Dataset to TrackNetV3 Format')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to TrackNet v1 dataset directory (containing game1/, game2/, etc.)')
    parser.add_argument('--output', type=str, default='data',
                       help='Output directory for TrackNetV3 format (default: data/)')
    parser.add_argument('--test-games', type=int, nargs='+', default=[9, 10],
                       help='Game IDs to use for test split (default: 9 10)')
    parser.add_argument('--no-background', action='store_true',
                       help='Skip background image generation')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Print progress information')

    args = parser.parse_args()

    print("=" * 80)
    print("Converting TrackNet v1 Tennis Dataset → TrackNetV3 Format")
    print("=" * 80)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Test split: games {args.test_games}")
    print()

    # Convert all games
    total_rallies = 0
    total_frames = 0

    for game_id in range(1, 11):
        rallies, frames = convert_game_to_match(args.input, args.output, game_id, verbose=args.verbose)
        total_rallies += rallies
        total_frames += frames

    print()
    print(f"✓ Conversion complete!")
    print(f"  Total rallies: {total_rallies}")
    print(f"  Total frames: {total_frames}")

    # Create test split
    print()
    print("Creating train/test split...")
    create_test_split(args.output, test_game_ids=args.test_games)

    # Generate background images
    if not args.no_background:
        import numpy as np  # Import here to avoid dependency if skipped
        generate_background_images(args.output, split='train')
        generate_background_images(args.output, split='test')

    print()
    print("=" * 80)
    print("✓ Dataset ready for TrackNetV3 training!")
    print("=" * 80)
    print()
    print("Directory structure:")
    print("  data/")
    print("    ├── train/")
    print("    │   ├── match1/")
    print("    │   │   ├── csv/{rally_id}_ball.csv")
    print("    │   │   ├── frame/{rally_id}/{frame}.png")
    print("    │   │   └── background/{rally_id}.png")
    print("    │   ├── match2/")
    print("    │   └── ...")
    print("    └── test/")
    print("        ├── match9/")
    print("        └── match10/")
    print()


if __name__ == '__main__':
    main()
