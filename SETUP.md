# TrackNetV3 for Tennis - Complete Setup Guide

## ✅ What's Already Done

- [x] TrackNetV3 cloned and configured
- [x] Mac GPU (MPS) support enabled and tested
- [x] Pretrained badminton weights downloaded (132 MB)
- [x] Dataset conversion script ready (`convert_tennis_dataset.py`)
- [x] MPS verification test passed ✅

## 🎾 Getting the Tennis Dataset

### Option 1: Download Original TrackNet Tennis Dataset

1. **Download from Google Drive:**
   - URL: https://drive.google.com/drive/folders/11r0RUaQHX7I3ANkaYG4jOxXK1OYo01Ut
   - Download all game folders (game1, game2, ..., game10)
   - Save to: `~/Downloads/tracknet-tennis-dataset/`

2. **Dataset Structure (Original v1):**
   ```
   tracknet-tennis-dataset/
   ├── game1/
   │   ├── Clip1/
   │   │   ├── Label.csv
   │   │   ├── 0000.jpg
   │   │   ├── 0001.jpg
   │   │   └── ...
   │   ├── Clip2/
   │   └── ...
   ├── game2/
   └── ...
   ```

3. **Dataset Stats:**
   - 10 broadcast tennis videos
   - 19,835 labeled frames total
   - Resolution: 1280×720
   - Frame rate: 30 fps

### Option 2: Use Your Own Tennis Videos

If you have your own tennis videos, you can label them using:
- [CVAT](https://github.com/opencv/cvat) - Open source annotation tool
- [LabelImg](https://github.com/tzutalin/labelImg) - Simple image annotation
- Export as CSV with columns: `Frame, X, Y, Visibility`

## 🔄 Converting Dataset to TrackNetV3 Format

Once you have the dataset:

```bash
cd tracknet-v3-tennis

# Convert TrackNet v1 → TrackNetV3 format
python convert_tennis_dataset.py \
  --input ~/Downloads/tracknet-tennis-dataset \
  --output data \
  --test-games 9 10 \
  --verbose

# This will create:
#   data/train/match{1-8}/  (training data)
#   data/test/match{9-10}/  (test data)
```

**What the conversion does:**
1. Reorganizes directory structure (game → match, Clip → rally)
2. Converts CSV format (v1 → v3 column names)
3. Converts images (JPG → PNG)
4. Generates background images (median filtering)
5. Splits into train/test (80/20)

**Expected output:**
```
data/
├── train/
│   ├── match1/
│   │   ├── csv/
│   │   │   └── 1_01_00_ball.csv
│   │   ├── frame/
│   │   │   └── 1_01_00/
│   │   │       ├── 0.png
│   │   │       └── ...
│   │   └── background/
│   │       └── 1_01_00.png
│   ├── match2/
│   └── ...
└── test/
    ├── match9/
    └── match10/
```

## 🚀 Training on Mac GPU

### Quick Start (Recommended Settings for Mac)

```bash
# Train TrackNet (Stage 1: Ball Detection)
python train.py \
  --exp_id tennis_tracknet_v1 \
  --num_epochs 30 \
  --batch_size 4 \
  --lr 1.0 \
  --seq_len 8 \
  --bg_mode subtract_concat \
  --frame_alpha 0.5

# Expected: ~30-60 min/epoch on M1/M2 Mac
# Total: ~15-30 hours for 30 epochs
```

### Training Parameters Explained

- `--batch_size 4`: Optimized for Mac GPU memory (vs 8-16 on NVIDIA)
- `--seq_len 8`: Number of consecutive frames to process
- `--bg_mode subtract_concat`: Use background subtraction (helps filter players!)
- `--frame_alpha 0.5`: Mixup augmentation strength
- `--lr 1.0`: Learning rate (same as original)

### Advanced: Training InpaintNet (Stage 2: Trajectory Refinement)

After TrackNet training:

```bash
# Generate predictions for training InpaintNet
python test.py \
  --split train \
  --tracknet_file exps/tennis_tracknet_v1/model_best.pt \
  --save_dir inpaint_training_data

# Train InpaintNet
python train.py \
  --exp_id tennis_inpaint_v1 \
  --model_name InpaintNet \
  --num_epochs 20 \
  --batch_size 8 \
  --pred_dir inpaint_training_data
```

## 📊 Monitoring Training

Training logs are saved to:
- `exps/tennis_tracknet_v1/plots/` - TensorBoard logs
- `exps/tennis_tracknet_v1/model_best.pt` - Best model checkpoint
- `exps/tennis_tracknet_v1/model_last.pt` - Latest checkpoint

**View with TensorBoard:**
```bash
pip install tensorboardX
tensorboard --logdir exps/tennis_tracknet_v1/plots
```

## 🧪 Testing Trained Model

### Inference on New Video

```bash
python predict.py \
  --video_file ../test-clips/test_30sec.mp4 \
  --tracknet_file exps/tennis_tracknet_v1/model_best.pt \
  --inpaintnet_file exps/tennis_inpaint_v1/model_best.pt \
  --save_dir results \
  --output_video

# Output:
#   results/test_30sec_ball.csv  (ball coordinates)
#   results/test_30sec.mp4       (video with trajectory overlay)
```

### Evaluate on Test Set

```bash
python test.py \
  --split test \
  --tracknet_file exps/tennis_tracknet_v1/model_best.pt \
  --inpaintnet_file exps/tennis_inpaint_v1/model_best.pt
```

## 🔧 Troubleshooting

### Out of Memory

Reduce batch size:
```bash
python train.py --batch_size 2 ...
```

### Slow Training

Check device usage:
```bash
python test_mps.py  # Should show "mps" as device
```

### Dataset Conversion Issues

Run with verbose mode:
```bash
python convert_tennis_dataset.py --input ... --verbose
```

## 📈 Expected Results

**TrackNetV3 (on badminton):**
- Accuracy: 97.51%
- Recall: 99.33%
- F1: 98.56%

**On Tennis (expected after fine-tuning):**
- Accuracy: ~95-97% (slightly lower due to different sport dynamics)
- Much better than current v1: fewer false positives on players
- Improved far-court detection with background subtraction

## 🎯 Next Steps

1. **Download tennis dataset** (Option 1 or 2 above)
2. **Convert dataset** using `convert_tennis_dataset.py`
3. **Start training** on Mac GPU (15-30 hours)
4. **Evaluate** on your test videos
5. **Fine-tune** if needed (adjust learning rate, augmentation)

## 💡 Tips for Best Results

1. **Data augmentation**: Default mixup (α=0.5) is good
2. **Background mode**: `subtract_concat` works best for filtering players
3. **Sequence length**: 8 frames is optimal (balances context vs memory)
4. **Early stopping**: Monitor validation loss, stop if plateaus
5. **Ensemble**: Train multiple models with different seeds, ensemble predictions

---

**Mac GPU Status:** ✅ Working
**Dataset Converter:** ✅ Ready
**Pretrained Weights:** ✅ Downloaded
**Ready to Train:** ⏳ Need dataset
