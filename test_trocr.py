"""
test_trocr.py
Loads the TrOCR model and predicts text from ALL images in the dataset/ folder.
Automatically uses GPU if available, otherwise falls back to CPU.
"""

import os
import sys
import warnings
import logging

# --- Suppress ALL noisy warnings/logs BEFORE any library imports ---
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import pandas as pd
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from metrics import compute_metrics, print_metrics, save_metrics_png

# Supported image extensions
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")


def main():
    model_name = "microsoft/trocr-base-handwritten"
    dataset_folder = os.path.join("dataset", "val2")
    manifest_csv = os.path.join("dataset", "split_manifest1.csv")

    print("=" * 50)
    print("TrOCR HANDWRITTEN TEXT RECOGNITION TEST")
    print("=" * 50)

    # --- Step 1: Find all images in the dataset folder ---
    if not os.path.isdir(dataset_folder):
        print(f"\nERROR: Folder '{dataset_folder}' not found.")
        return

    image_files = sorted([
        f for f in os.listdir(dataset_folder)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ])

    if not image_files:
        print(f"\nERROR: No images found in '{dataset_folder}/'")
        print(f"Supported formats: {IMAGE_EXTENSIONS}")
        return

    print(f"Found {len(image_files)} image(s) in '{dataset_folder}/'")
    print()

    # --- Load ground-truth labels (if manifest is available) ---
    labels = {}
    if os.path.exists(manifest_csv):
        gt = pd.read_csv(manifest_csv, keep_default_na=False)
        labels = dict(zip(gt["filename"], gt["label"].astype(str)))
        print(f"Loaded {len(labels)} ground-truth labels from '{manifest_csv}'")
    else:
        print(f"NOTE: '{manifest_csv}' not found — metrics will be skipped.")
    print()

    # --- Step 2: Select device (GPU or CPU) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # --- Step 3: Load processor and model (once) ---
    print("Loading processor...")
    processor = TrOCRProcessor.from_pretrained(model_name)

    print("Loading model...")
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    print("Model loaded and ready.\n")

    # --- Step 4: Process each image ---
    print("=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"{'#':<4} {'Filename':<22} {'Ground Truth':<22} {'Prediction'}")
    print("-" * 80)

    references = []
    predictions = []

    for i, filename in enumerate(image_files, start=1):
        image_path = os.path.join(dataset_folder, filename)
        ground_truth = labels.get(filename)
        try:
            image = Image.open(image_path).convert("RGB")
            pixel_values = processor(images=image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)

            with torch.no_grad():
                generated_ids = model.generate(pixel_values, max_new_tokens=64)

            predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            gt_display = ground_truth if ground_truth is not None else "(no label)"
            print(f"{i:<4} {filename:<22} {gt_display:<22} {predicted_text}")

            if ground_truth is not None:
                references.append(str(ground_truth).strip())
                predictions.append(predicted_text)

        except Exception as e:
            print(f"{i:<4} {filename:<22} {'ERROR':<22} {e}")

    print("-" * 80)
    print(f"\nDone. Processed {len(image_files)} image(s).")

    # --- Evaluation metrics ---
    if references:
        results = compute_metrics(references, predictions)
        print()
        print_metrics(results, title="BASE MODEL — EVALUATION METRICS")
        save_metrics_png(results, subfolder="base", model_label="microsoft/trocr-base-handwritten")
    else:
        print("\nNo ground-truth labels matched these images — metrics skipped.")


if __name__ == "__main__":
    main()
