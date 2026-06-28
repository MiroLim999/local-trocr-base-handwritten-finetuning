"""
predict.py
Predicts handwritten text from ALL images in a given folder
using your fine-tuned TrOCR model.

No CSV or labels needed — just drop images in a folder and run.

Usage:
  python predict.py                          (uses default folder: new_images/)
  python predict.py --folder path/to/images  (uses custom folder)
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

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# ============================================================
# CONFIG
# ============================================================
FINETUNED_DIR = "trocr-finetuned"         # Path to your fine-tuned model
DEFAULT_FOLDER = "new_images"             # Default folder for new images
# ============================================================

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")


def main():
    # Allow custom folder via command line argument
    if "--folder" in sys.argv:
        idx = sys.argv.index("--folder")
        if idx + 1 < len(sys.argv):
            image_folder = sys.argv[idx + 1]
        else:
            print("ERROR: --folder requires a path argument")
            return
    else:
        image_folder = DEFAULT_FOLDER

    print("=" * 60)
    print("TrOCR PREDICTION — NEW IMAGES")
    print("=" * 60)

    # --- Verify paths ---
    if not os.path.isdir(FINETUNED_DIR):
        print(f"\nERROR: Fine-tuned model not found at '{FINETUNED_DIR}/'")
        print("Run train_trocr.py first to fine-tune the model.")
        return

    if not os.path.isdir(image_folder):
        print(f"\nERROR: Image folder not found at '{image_folder}/'")
        print(f"Create the folder and add your images there.")
        print(f"\nExample:")
        print(f"  mkdir {image_folder}")
        print(f"  (copy your images into {image_folder}/)")
        print(f"  python predict.py")
        return

    # --- Find all images ---
    image_files = sorted([
        f for f in os.listdir(image_folder)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ])

    if not image_files:
        print(f"\nERROR: No images found in '{image_folder}/'")
        print(f"Supported formats: {IMAGE_EXTENSIONS}")
        return

    print(f"Folder: {os.path.abspath(image_folder)}")
    print(f"Images found: {len(image_files)}")
    print()

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

    # --- Run predictions ---
    print("=" * 60)
    print("PREDICTIONS")
    print("=" * 60)
    print(f"{'#':<4} {'Filename':<35} {'Predicted Text'}")
    print("-" * 70)

    results = []

    for i, filename in enumerate(image_files, start=1):
        img_path = os.path.join(image_folder, filename)

        try:
            image = Image.open(img_path).convert("RGB")
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

            with torch.no_grad():
                generated_ids = model.generate(pixel_values, max_new_tokens=32)

            predicted = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            print(f"{i:<4} {filename:<35} {predicted}")
            results.append((filename, predicted))

        except Exception as e:
            print(f"{i:<4} {filename:<35} ERROR: {e}")
            results.append((filename, f"ERROR: {e}"))

    print("-" * 70)
    print(f"\nDone. Predicted {len(results)} image(s).")

    # --- Save results to CSV ---
    output_csv = os.path.join(image_folder, "predictions.csv")
    with open(output_csv, "w", encoding="utf-8") as f:
        f.write("FILENAME,PREDICTION\n")
        for filename, prediction in results:
            # Escape commas in predictions
            prediction_clean = prediction.replace('"', '""')
            f.write(f'{filename},"{prediction_clean}"\n')

    print(f"Results saved to: {output_csv}")


if __name__ == "__main__":
    main()
