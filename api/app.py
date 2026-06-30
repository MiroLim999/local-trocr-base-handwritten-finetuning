"""
app.py
Flask OCR API that serves the fine-tuned TrOCR model.

The web frontend sends one or more cropped field images (as PNG data URLs).
For each crop, this API returns the predicted text and a confidence score
(the model's certainty in its own output, 0-100%).

Run:
  python app.py
Then the API is available at http://127.0.0.1:5000

Endpoints:
  GET  /health   -> simple status + which model is loaded
  POST /ocr      -> { "fields": [ { "name": "...", "image": "data:image/png;base64,..." } ] }
                    returns { "results": [ { "name", "text", "confidence" } ] }
"""

import os
import io
import math
import base64
import warnings
import logging

# --- Quiet down HF / transformers noise ---
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import torch
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# ============================================================
# CONFIG
# ============================================================
# Resolve the fine-tuned model folder relative to the repo root
# (this file lives in <repo>/api/app.py).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINETUNED_DIR = os.path.join(REPO_ROOT, "trocr-finetuned")
FALLBACK_MODEL = "microsoft/trocr-base-handwritten"
MAX_NEW_TOKENS = 32
# ============================================================

app = Flask(__name__)
CORS(app)  # allow the PHP/JS frontend (different port) to call this API

# Loaded lazily on first request so the server starts instantly.
_state = {"model": None, "processor": None, "device": None, "eos_id": None, "label": None}


def _load_model():
    if _state["model"] is not None:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.isdir(FINETUNED_DIR):
        model_src = FINETUNED_DIR
        label = "trocr-finetuned (local)"
    else:
        model_src = FALLBACK_MODEL
        label = FALLBACK_MODEL + " (fallback)"

    print(f"[ocr-api] Loading model: {model_src}  on  {device}")
    processor = TrOCRProcessor.from_pretrained(model_src)
    model = VisionEncoderDecoderModel.from_pretrained(model_src)
    model.to(device)
    model.eval()

    eos_id = (
        getattr(model.generation_config, "eos_token_id", None)
        or getattr(model.config, "eos_token_id", None)
        or getattr(model.config.decoder, "eos_token_id", None)
        or processor.tokenizer.sep_token_id
    )

    _state.update(model=model, processor=processor, device=device,
                  eos_id=eos_id, label=label)
    print("[ocr-api] Model ready.")


def _sequence_confidence(model, gen_output, eos_id):
    """Geometric mean of per-token probabilities up to the first EOS, as a %."""
    try:
        scores = model.compute_transition_scores(
            gen_output.sequences, gen_output.scores, normalize_logits=True
        )[0]
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
        return round(math.exp(sum(log_probs) / len(log_probs)) * 100.0, 1)
    except Exception:
        return 0.0


def _decode_data_url(data_url):
    """Turn a 'data:image/png;base64,...' string into a PIL RGB image."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    return Image.open(io.BytesIO(raw)).convert("RGB")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": _state["model"] is not None,
        "model": _state["label"] or ("trocr-finetuned" if os.path.isdir(FINETUNED_DIR) else FALLBACK_MODEL),
        "device": str(_state["device"]) if _state["device"] else "not-loaded",
    })


@app.route("/ocr", methods=["POST"])
def ocr():
    data = request.get_json(silent=True) or {}
    fields = data.get("fields", [])
    if not isinstance(fields, list) or not fields:
        return jsonify({"error": "Send a non-empty 'fields' list."}), 400

    _load_model()
    model = _state["model"]
    processor = _state["processor"]
    device = _state["device"]
    eos_id = _state["eos_id"]

    results = []
    for field in fields:
        name = field.get("name", "field")
        image_data = field.get("image", "")
        try:
            image = _decode_data_url(image_data)
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                gen_output = model.generate(
                    pixel_values,
                    max_new_tokens=MAX_NEW_TOKENS,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            text = processor.batch_decode(
                gen_output.sequences, skip_special_tokens=True
            )[0].strip()
            conf = _sequence_confidence(model, gen_output, eos_id)
            results.append({"name": name, "text": text, "confidence": conf})
        except Exception as e:
            results.append({"name": name, "text": "", "confidence": 0.0, "error": str(e)})

    return jsonify({"results": results, "model": _state["label"]})


if __name__ == "__main__":
    # threaded=False keeps PyTorch inference predictable for this prototype.
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=False)
