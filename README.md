# Tennis Ball Tracking with TrackNet

Tennis ball detection and trajectory tracking using TrackNet/TrackNetV4, adapted from [TrackNetV3](https://github.com/qaz812345/TrackNetV3) (originally for badminton shuttlecock tracking).

## Features

- **TrackNet & TrackNetV4** architectures for ball detection
- **InpaintNet** for trajectory rectification (fills gaps from occlusions)
- **Background subtraction** to filter out static elements (players, court lines)
- **Mac GPU (MPS) support** for training and inference
- **Precomputed dataset** pipeline for fast data loading

## Setup

See [SETUP.md](SETUP.md) for detailed instructions including:
- Environment setup and dependencies
- Downloading and converting the tennis dataset
- Mac GPU (MPS) verification

Quick start:

```bash
pip install -r requirements.txt
```

## Training

```bash
# Train TrackNet (ball detection)
python train.py \
  --exp_id tennis_v1 \
  --num_epochs 30 \
  --batch_size 4 \
  --lr 1.0 \
  --seq_len 8 \
  --bg_mode subtract_concat \
  --frame_alpha 0.5

# Resume training
python train.py --exp_id tennis_v1 --resume_training
```

Training logs and checkpoints are saved to `exps/<exp_id>/`.

## Inference

```bash
# Predict ball positions from a video
python predict.py \
  --video_file input.mp4 \
  --tracknet_file exps/tennis_v1/model_best.pt \
  --save_dir prediction_tennis \
  --output_video
```

## Evaluation

```bash
# Evaluate on test set
python test.py \
  --split test \
  --tracknet_file exps/tennis_v1/model_best.pt
```

## Model Architecture

- **TrackNet**: Encoder-decoder CNN that takes a sequence of frames (with background subtraction) and predicts ball heatmaps
- **TrackNetV4**: Enhanced architecture with improved feature extraction
- **InpaintNet**: Trajectory rectification network that fills in missing detections using surrounding trajectory context

## Project Structure

```
├── README.md                  # This file
├── SETUP.md                   # Setup and dataset guide
├── requirements.txt           # Python dependencies
├── model.py                   # TrackNet, TrackNetV4, InpaintNet
├── dataset.py                 # Dataset class and data loading
├── train.py                   # Training script
├── test.py                    # Evaluation script
├── predict.py                 # Inference/prediction
├── utils/
│   ├── general.py             # Utilities, model loading
│   ├── metric.py              # Evaluation metrics
│   └── visualize.py           # Visualization helpers
└── scripts/
    ├── convert_tennis_dataset.py  # Dataset format conversion
    ├── precompute_dataset.py      # Precompute processed frames
    ├── bench_batch_size.py        # Batch size benchmarking
    └── check_training.py          # Training progress monitor
```

Data, checkpoints, and experiment outputs are gitignored (see `.gitignore`).

## Acknowledgments

Based on [TrackNetV3](https://github.com/qaz812345/TrackNetV3) by Yu-Huan Wu et al.
- Paper: [TrackNetV3: Enhancing ShuttleCock Tracking](https://dl.acm.org/doi/10.1145/3595916.3626370)

## License

[MIT](LICENSE)
