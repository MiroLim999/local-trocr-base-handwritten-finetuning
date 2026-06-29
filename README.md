# Local TrOCR Handwritten Fine-Tuning

Fine-tune Microsoft's [TrOCR](https://huggingface.co/microsoft/trocr-base-handwritten) (`microsoft/trocr-base-handwritten`) on your own handwritten text images, then evaluate and run predictions locally. Training runs on GPU automatically when CUDA is available and falls back to CPU otherwise.

## Features

- Fine-tune `trocr-base-handwritten` on a custom image + label dataset
- Mixed-precision (AMP) training with per-epoch validation and best-checkpoint saving
- Dependency-free CER / WER / exact-match metrics, plus saved bar-chart PNGs
- Batch prediction over a folder of images (no labels required)
- Baseline evaluation of the original model for before/after comparison

## Project Structure

```
.
├── download_trocr.py     # Download the base TrOCR model + processor from Hugging Face
├── train_trocr.py        # Fine-tune the model on your dataset
├── test_trocr.py         # Evaluate the BASE model and save metrics
├── test_finetuned.py     # Evaluate the FINE-TUNED model and save metrics
├── predict.py            # Run predictions on a folder of new images
├── metrics.py            # CER / WER / accuracy helpers + metrics PNG export
├── requirements.txt
├── Evaluation Metrics/   # Saved metric charts (base/ and finetuned/)
├── dataset/              # Your images + label CSVs (not tracked in git)
└── trocr-finetuned/      # Saved fine-tuned model (not tracked in git)
```

> Note: `dataset/`, `trocr-finetuned/`, and `venv/` are intentionally excluded from the repo via `.gitignore` because of their size. See [Model & Data](#model--data) below.

## Setup

Requires Python 3.10+ (developed on 3.13). A CUDA-capable GPU is recommended but optional.

```bash
# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

For GPU support, install the CUDA build of PyTorch that matches your system from the [official PyTorch install guide](https://pytorch.org/get-started/locally/) instead of the default CPU wheel.

## Dataset Format

Place images in split folders and describe them with a manifest CSV:

```
dataset/
├── split_manifest.csv     # columns: filename,label,split,source
├── train/syn_000001.png
└── val/syn_000004.png
```

The manifest CSV needs at least these columns:

| column     | description                                   |
|------------|-----------------------------------------------|
| `filename` | image file name (matched within the split dir)|
| `label`    | ground-truth text for the image               |
| `split`    | `train` or `val`                              |
| `source`   | optional origin tag                           |

Rows with empty labels or the label `UNREADABLE` are skipped automatically.

## Usage

### 1. Download the base model

```bash
python download_trocr.py
```

Downloads `microsoft/trocr-base-handwritten` (~1.2 GB) and the processor, and prints a CUDA availability check.

### 2. (Optional) Evaluate the base model

```bash
python test_trocr.py
```

Runs the original model over your evaluation images and saves a chart to `Evaluation Metrics/base/`.

### 3. Fine-tune

```bash
python train_trocr.py
```

Hyperparameters live in the `CONFIG` section near the top of `train_trocr.py` (epochs, batch size, learning rate, max label length, output dir). The best checkpoint by validation loss is saved to `trocr-finetuned/`.

### 4. Evaluate the fine-tuned model

```bash
python test_finetuned.py
```

Saves a chart to `Evaluation Metrics/finetuned/` so you can compare against the base run.

### 5. Predict on new images

```bash
python predict.py                          # uses the default new_images/ folder
python predict.py --folder path/to/images  # custom folder
```

Predictions are printed and written to a `predictions.csv` inside the image folder.

## Metrics

`metrics.py` computes corpus-level scores:

- CER (Character Error Rate)
- WER (Word Error Rate)
- Exact-match accuracy (whole string)

Each evaluation also exports a timestamped PNG bar chart under `Evaluation Metrics/`.

## Model & Data

The fine-tuned model (`trocr-finetuned/`, ~1.3 GB) and the `dataset/` folder are not stored in this repo because they exceed GitHub's file-size limits. To share them, consider:

- Pushing the model to the [Hugging Face Hub](https://huggingface.co/docs/hub/models-uploading)
- Tracking large files with [Git LFS](https://git-lfs.com/)
- Hosting the dataset externally and linking to it

## Continued Fine-Tuning

To keep training from an existing fine-tuned checkpoint instead of the base model, point the loaders in `train_trocr.py` at your saved directory:

```python
processor = TrOCRProcessor.from_pretrained("trocr-finetuned", local_files_only=True)
model = VisionEncoderDecoderModel.from_pretrained("trocr-finetuned", local_files_only=True)
```

When continuing, a lower learning rate and mixing in earlier data help reduce catastrophic forgetting.

## License

No license specified. Add one if you intend others to reuse this code.
