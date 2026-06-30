"""
predict.py
Predicts handwritten text from ALL images in a given folder
using your fine-tuned TrOCR model.

No CSV or labels needed — just drop images in a folder and run.
Each prediction includes a confidence score (the model's certainty in its own
output, 0-100%). This is not true accuracy — there are no ground-truth labels
here — but low confidence is a useful flag for predictions worth reviewing.

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

import math

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# ============================================================
# CONFIG
# ============================================================
# FINETUNED_DIR = "microsoft/trocr-base-handwritten" # Default Model
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
    # FINETUNED_DIR may be a local folder (e.g. "trocr-finetuned") OR a Hugging
    # Face model name (e.g. "microsoft/trocr-base-handwritten"). Only error out
    # for a local-looking path that doesn't exist; let HF names pass through to
    # from_pretrained (which loads from cache / downloads).
    looks_like_local_path = os.path.sep in FINETUNED_DIR or FINETUNED_DIR.startswith(".")
    is_hf_name = "/" in FINETUNED_DIR and not looks_like_local_path
    if not os.path.isdir(FINETUNED_DIR) and not is_hf_name:
        print(f"\nERROR: Model not found at '{FINETUNED_DIR}/'")
        print("Set FINETUNED_DIR to your fine-tuned folder (e.g. 'trocr-finetuned')")
        print("or a Hugging Face name (e.g. 'microsoft/trocr-base-handwritten').")
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
    print(f"Loading model: {FINETUNED_DIR}")
    processor = TrOCRProcessor.from_pretrained(FINETUNED_DIR)
    model = VisionEncoderDecoderModel.from_pretrained(FINETUNED_DIR)
    model.to(device)
    model.eval()
    print("Model loaded.\n")

    # --- Run predictions ---
    print("=" * 70)
    print("PREDICTIONS")
    print("=" * 70)
    print("Note: 'Confidence' is the model's certainty in its own output")
    print("      (not true accuracy, since these images have no ground-truth labels).")
    print()
    print(f"{'#':<4} {'Filename':<35} {'Conf %':<8} {'Predicted Text'}")
    print("-" * 80)

    results = []
    # eos id can live in different places depending on the model; fall back safely.
    eos_id = (
        getattr(model.generation_config, "eos_token_id", None)
        or getattr(model.config, "eos_token_id", None)
        or getattr(model.config.decoder, "eos_token_id", None)
        or processor.tokenizer.sep_token_id
    )

    def sequence_confidence(gen_output):
        """Geometric mean of per-token probabilities, up to the first EOS, as a %."""
        try:
            scores = model.compute_transition_scores(
                gen_output.sequences, gen_output.scores, normalize_logits=True
            )[0]  # log-probs for each generated step (batch of 1)
            # sequences[0] = [start_token, tok1, tok2, ...]; align generated tokens with scores
            gen_tokens = gen_output.sequences[0][1:1 + len(scores)]
            log_probs = []
            for tok, lp in zip(gen_tokens, scores):
                if not torch.isfinite(lp):
                    continue
                log_probs.append(lp.item())
                if tok.item() == eos_id:
                    break
            if not log_probs:
                return 0.0
            return math.exp(sum(log_probs) / len(log_probs)) * 100.0
        except Exception:
            return float("nan")

    confidences = []

    for i, filename in enumerate(image_files, start=1):
        img_path = os.path.join(image_folder, filename)

        try:
            image = Image.open(img_path).convert("RGB")
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

            with torch.no_grad():
                gen_output = model.generate(
                    pixel_values,
                    max_new_tokens=32,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            generated_ids = gen_output.sequences
            predicted = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            conf = sequence_confidence(gen_output)
            confidences.append(conf)

            print(f"{i:<4} {filename:<35} {conf:<8.1f} {predicted}")
            results.append((filename, predicted, conf))

        except Exception as e:
            print(f"{i:<4} {filename:<35} {'-':<8} ERROR: {e}")
            results.append((filename, f"ERROR: {e}", float("nan")))

    print("-" * 80)
    print(f"\nDone. Predicted {len(results)} image(s).")

    # --- Average confidence summary ---
    valid_conf = [c for c in confidences if not math.isnan(c)]
    if valid_conf:
        avg_conf = sum(valid_conf) / len(valid_conf)
        low = sum(1 for c in valid_conf if c < 80.0)
        print(f"Average confidence: {avg_conf:.1f}%")
        print(f"Low-confidence (<80%) predictions to review: {low}/{len(valid_conf)}")

    # --- Save results to CSV ---
    output_csv = os.path.join(image_folder, "predictions.csv")
    with open(output_csv, "w", encoding="utf-8") as f:
        f.write("FILENAME,PREDICTION,CONFIDENCE\n")
        for filename, prediction, conf in results:
            # Escape quotes in predictions
            prediction_clean = prediction.replace('"', '""')
            conf_str = "" if (isinstance(conf, float) and math.isnan(conf)) else f"{conf:.1f}"
            f.write(f'{filename},"{prediction_clean}",{conf_str}\n')

    print(f"Results saved to: {output_csv}")


if __name__ == "__main__":
    main()
