# SwiftFormer: Engineering Symbol Classification

This project implements a complete pipeline for classifying Geometric Dimensioning and Tolerancing (GD&T) symbols using the SwiftFormer architecture. It includes synthetic dataset generation, training (optimized for macOS/MPS), and a rich HTML-based evaluation tool.

## Features

- **Synthetic Dataset Generation**: Generates varied GD&T symbol images with realistic augmentations (noise, blur, perspective transforms, lighting gradients).
- **SwiftFormer Integration**: Fine-tunes the `MBZUAI/swiftformer-s` model for high-efficiency classification.
- **macOS Optimized**: Training and inference scripts are configured to leverage the Apple Silicon GPU (MPS).
- **Rich Evaluation**: Generates a side-by-side HTML report comparing Ground Truth vs. Predicted symbols with latency tracking.

---

## Setup

### Prerequisites
Ensure you have Python 3.9+ installed and the following dependencies:

```bash
pip install torch torchvision transformers opencv-python Pillow numpy
```

### Fonts
The dataset generator requires GD&T fonts to be placed in the `Fonts/` directory:
- `Fonts/Y145m.ttf`
- `Fonts/GDT Regular.ttf`

---

## Project Structure

- `generate_swiftformer_dataset.py`: Synthetic data engine.
- `TrainSwiftFormer.py`: Fine-tuning script for SwiftFormer-S.
- `TestSwiftFormerMPS.py`: Evaluation script using Metal Performance Shaders (MPS).
- `TestSwiftFormerCPU.py`: Evaluation script for CPU-only environments.
- `best_swiftformer_gdtFCF_20EPOCH.pth`: Pre-trained weights (20 Epochs).

---

## Dataset Generation

Produce a PyTorch-compatible `ImageFolder` dataset for 14 different GD&T symbols.

```bash
python generate_swiftformer_dataset.py --count 625 --output swiftformer_dataset
```

- **`--count`**: Number of images per symbol (default: 625).
- **`--output`**: Target directory for the dataset.

The script applies random severity (mild to extreme), including 70% probability of drawing Feature Control Frame (FCF) boxes around symbols and various image degradation techniques to simulate real-world scanned engineering drawings.

---

## Training

Train the SwiftFormer-S model on your generated dataset.

```bash
python TrainSwiftFormer.py
```

> [!IMPORTANT]
> Change the `data_dir` variable in `TrainSwiftFormer.py` (Line 52) to point to your generated dataset path before running.

**Configuration Details:**
- **Optimization**: AdamW with Weight Decay (0.05).
- **Scheduler**: Cosine annealing with linear warmup.
- **Normalization**: Uses SwiftFormer-S specific mean/std from `transformers`.

---

## Testing & Evaluation

Run inference on a validation set and generate a detailed HTML report.

```bash
python TestSwiftFormerMPS.py <input_folder> --model_path <path_to_model>.pth --output_html report.html
```

### Example:
```bash
python TestSwiftFormerMPS.py swiftformer_dataset/val --model_path best_swiftformer_gdtFCF_20EPOCH.pth --output_html my_report.html
```

The report includes:
- **Accuracy Metrics**: Overall and per-class breakdown.
- **Latency**: Average and per-image inference speed.
- **Visual Comparison**: Side-by-side view of the input image, ground truth, and predicted symbol.
