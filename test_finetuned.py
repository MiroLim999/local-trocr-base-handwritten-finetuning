"""
test_finetuned.py
Tests the fine-tuned TrOCR model on images from the test set.
Compares predictions against ground truth labels.
"""

import os
import sys
import warnings
import logging

# --- Suppress warnings ---
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

# ============================================================
# CONFIG
# ============================================================
FINETUNED_DIR = "trocr-finetuned"        # Path to your fine-tuned model
MANIFEST_CSV = os.path.join("dataset", "manifest.csv")
TEST_IMG_DIR = os.path.join("dataset", "test")   # Folder of images to evaluate
NUM_SAMPLES = 6000
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")
# ============================================================


def main():
    print("=" * 60)
    print("FINE-TUNED TrOCR — TEST RESULTS")
    print("=" * 60)

    # --- Verify paths ---
    if not os.path.isdir(FINETUNED_DIR):
        print(f"\nERROR: Fine-tuned model not found at '{FINETUNED_DIR}/'")
        print("Run train_trocr.py first to fine-tune the model.")
        return

    if not os.path.exists(MANIFEST_CSV):
        print(f"\nERROR: Manifest CSV not found at '{MANIFEST_CSV}'")
        return

    if not os.path.isdir(TEST_IMG_DIR):
        print(f"\nERROR: Image folder not found at '{TEST_IMG_DIR}/'")
        return

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # --- Load fine-tuned model ---
    print("Loading fine-tuned model...")
    processor = TrOCRProcessor.from_pretrained(FINETUNED_DIR)
    model = VisionEncoderDecoderModel.from_pretrained(FINETUNED_DIR)
    model.to(device)
    model.eval()
    print("Model loaded.\n")

    # --- Load labels and list images in the test folder ---
    df = pd.read_csv(MANIFEST_CSV, keep_default_na=False)
    labels = dict(zip(df["filename"], df["label"].astype(str)))

    image_files = sorted([
        f for f in os.listdir(TEST_IMG_DIR)
        if f.lower().endswith(IMAGE_EXTENSIONS) and f in labels
    ])

    # Skip empty / UNREADABLE labels
    image_files = [
        f for f in image_files
        if labels[f].strip() != "" and labels[f].strip().upper() != "UNREADABLE"
    ]

    # Limit number of samples
    image_files = image_files[:NUM_SAMPLES]

    # --- Run predictions ---
    print(f"{'#':<4} {'Filename':<20} {'Ground Truth':<20} {'Prediction':<20} {'Match'}")
    print("-" * 85)

    references = []
    predictions = []
    total = 0

    for filename in image_files:
        ground_truth = str(labels[filename]).strip()
        img_path = os.path.join(TEST_IMG_DIR, filename)

        if not os.path.exists(img_path):
            continue

        image = Image.open(img_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_new_tokens=32)

        predicted = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        references.append(ground_truth)
        predictions.append(predicted)
        total += 1

        match = "OK" if predicted == ground_truth else "x"
        print(f"{total:<4} {filename:<20} {ground_truth:<20} {predicted:<20} {match}")

    print("-" * 85)

    # --- Evaluation metrics ---
    if total > 0:
        results = compute_metrics(references, predictions)
        print()
        print_metrics(results, title="FINE-TUNED MODEL — EVALUATION METRICS")
        save_metrics_png(results, subfolder="finetuned", model_label="trocr-finetuned")
    else:
        print("\nNo samples were evaluated (no matching images found).")


if __name__ == "__main__":
    main()
