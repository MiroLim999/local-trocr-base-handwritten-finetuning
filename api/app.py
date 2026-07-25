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
  GET  /health     -> status + which models are available/loaded
  GET  /models     -> selectable models for the frontend dropdown
  POST /ocr        -> { "fields": [ { "name": "...", "image": "data:image/png;base64,..." } ],
                       "model": "<key>" }  returns { "results": [ { "name", "text", "confidence" } ] }
  POST /add_model  -> multipart upload (name + files) saved into Models/<name>/
  POST /delete_model -> { "model": "<key>" } removes that folder from Models/
  POST /rename_model -> { "model": "<key>", "newName": "<name>" } renames the folder
"""

import os
import io
import re
import math
import shutil
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
# Resolve paths relative to the repo root (this file lives in <repo>/api/app.py).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# All fine-tuned models live in <repo>/Models/<name>/. Drop a model folder in
# there (with config.json + model.safetensors) and it shows up automatically.
MODELS_DIR = os.path.join(REPO_ROOT, "Models")

# The un-fine-tuned base model, always offered as an option.
FALLBACK_MODEL = "microsoft/trocr-base-handwritten"
BASE_MODEL_KEY = "base"
BASE_MODEL_LABEL = "TrOCR base (not fine-tuned)"

MAX_NEW_TOKENS = 32

# Optional friendly labels for known folders. Any folder not listed here gets a
# label auto-generated from its name.
MODEL_LABELS = {
    "trocr-finetuned":    "TrOCR fine-tuned (v1)",
    "v2-finetuned-model": "TrOCR fine-tuned (v2)",
}
# Preferred default model when the frontend doesn't specify one.
PREFERRED_DEFAULT = "trocr-finetuned"
# ============================================================

app = Flask(__name__)
CORS(app)  # allow the PHP/JS frontend (different port) to call this API

# Models are loaded lazily and cached by key so the server starts instantly
# and each model is only loaded once.
_device = None
_models = {}  # cache_key -> {"model", "processor", "eos_id", "label"}


def _get_device():
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def _looks_like_model(path):
    """A folder is usable if it has a config.json and a weights file."""
    if not os.path.isdir(path):
        return False
    has_config = os.path.isfile(os.path.join(path, "config.json"))
    has_weights = any(
        os.path.isfile(os.path.join(path, w))
        for w in ("model.safetensors", "pytorch_model.bin")
    )
    return has_config and has_weights


def _discover_models():
    """Map of {folder_name: abs_path} for every valid model under Models/.

    Re-scanned on each call so newly added folders appear without a restart."""
    found = {}
    if os.path.isdir(MODELS_DIR):
        for name in sorted(os.listdir(MODELS_DIR)):
            path = os.path.join(MODELS_DIR, name)
            if _looks_like_model(path):
                found[name] = path
    return found


def _prettify(name):
    return name.replace("-", " ").replace("_", " ").strip().title()


def _label_for(key):
    if key == BASE_MODEL_KEY:
        return BASE_MODEL_LABEL
    return MODEL_LABELS.get(key, _prettify(key))


def _default_key():
    discovered = _discover_models()
    if PREFERRED_DEFAULT in discovered:
        return PREFERRED_DEFAULT
    if discovered:
        return next(iter(discovered))
    return BASE_MODEL_KEY


def _resolve_key(requested):
    """Choose a usable model key, honoring the request when possible."""
    if requested == BASE_MODEL_KEY:
        return BASE_MODEL_KEY
    if requested in _discover_models():
        return requested
    return _default_key()


def _model_info():
    """Descriptor list of every selectable model for /models and /health."""
    infos = [
        {
            "key": key,
            "label": _label_for(key),
            "available": True,
            "loaded": key in _models,
        }
        for key in _discover_models()
    ]
    # The base model is always available (pulled from the HF cache / hub).
    infos.append({
        "key": BASE_MODEL_KEY,
        "label": BASE_MODEL_LABEL,
        "available": True,
        "loaded": BASE_MODEL_KEY in _models,
    })
    return infos


def _load_model(key):
    """Load (and cache) the model for `key`. Returns its cache entry."""
    device = _get_device()

    if key == BASE_MODEL_KEY:
        model_src, label, cache_key = FALLBACK_MODEL, BASE_MODEL_LABEL, BASE_MODEL_KEY
    else:
        model_src = _discover_models().get(key)
        if model_src is None:
            # Requested folder vanished/invalid -> fall back to the base model.
            model_src, label, cache_key = FALLBACK_MODEL, BASE_MODEL_LABEL, BASE_MODEL_KEY
        else:
            label, cache_key = _label_for(key), key

    if cache_key in _models:
        return _models[cache_key]

    print(f"[ocr-api] Loading model: {label}  ({model_src})  on  {device}")
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

    entry = {"model": model, "processor": processor, "eos_id": eos_id, "label": label}
    _models[cache_key] = entry
    print(f"[ocr-api] Model ready: {label}")
    return entry


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
        "model_loaded": bool(_models),
        "device": str(_device) if _device else "not-loaded",
        "default": _default_key(),
        "models": _model_info(),
    })


@app.route("/models", methods=["GET"])
def models():
    """List selectable models so the frontend can build its dropdown."""
    return jsonify({
        "default": _default_key(),
        "models": _model_info(),
    })


@app.route("/ocr", methods=["POST"])
def ocr():
    data = request.get_json(silent=True) or {}
    fields = data.get("fields", [])
    if not isinstance(fields, list) or not fields:
        return jsonify({"error": "Send a non-empty 'fields' list."}), 400

    key = _resolve_key(data.get("model"))
    entry = _load_model(key)
    model = entry["model"]
    processor = entry["processor"]
    device = _get_device()
    eos_id = entry["eos_id"]

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

    return jsonify({"results": results, "model": entry["label"], "modelKey": key})


# Files we accept inside a model folder.
_ALLOWED_MODEL_FILES = {
    "config.json", "generation_config.json",
    "preprocessor_config.json", "processor_config.json",
    "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json",
    "vocab.json", "merges.txt", "sentencepiece.bpe.model", "spm.model",
    "model.safetensors", "pytorch_model.bin",
}
_ALLOWED_MODEL_EXTS = {".json", ".safetensors", ".bin", ".txt", ".model"}


@app.route("/add_model", methods=["POST"])
def add_model():
    """Save an uploaded model folder into Models/<name>/ so it becomes selectable.

    Multipart form:
      name      -> desired model name (folder name)
      files     -> the model files (config.json, model.safetensors, ...)

    werkzeug streams large uploads to disk, so multi-GB weights are handled
    without buffering the whole request in memory."""
    raw_name = (request.form.get("name") or "").strip()
    if not raw_name:
        return jsonify({"ok": False, "error": "Please provide a model name."}), 400

    # Sanitize into a safe folder name.
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", raw_name).strip("-._")
    if not name or name == BASE_MODEL_KEY:
        return jsonify({"ok": False, "error": "That model name is not allowed."}), 400

    files = request.files.getlist("files") or request.files.getlist("files[]")
    if not files:
        return jsonify({"ok": False, "error": "No files were uploaded."}), 400

    # Validate names/extensions and that it looks like a real model.
    incoming = {}
    for f in files:
        base = os.path.basename((f.filename or "").replace("\\", "/"))
        if not base or ".." in base:
            return jsonify({"ok": False, "error": "Invalid file name in upload."}), 400
        ext = os.path.splitext(base)[1].lower()
        if base not in _ALLOWED_MODEL_FILES and ext not in _ALLOWED_MODEL_EXTS:
            return jsonify({"ok": False, "error": f"File type not allowed: {base}"}), 400
        incoming[base] = f

    if "config.json" not in incoming:
        return jsonify({"ok": False, "error": "Missing config.json."}), 400
    if not ({"model.safetensors", "pytorch_model.bin"} & set(incoming)):
        return jsonify({"ok": False, "error": "Missing weights (model.safetensors or pytorch_model.bin)."}), 400

    os.makedirs(MODELS_DIR, exist_ok=True)
    target = os.path.join(MODELS_DIR, name)
    if os.path.isdir(target):
        return jsonify({"ok": False, "error": f"A model named '{name}' already exists."}), 409

    os.makedirs(target)
    try:
        saved = []
        for base, f in incoming.items():
            f.save(os.path.join(target, base))
            saved.append(base)
    except Exception as e:
        shutil.rmtree(target, ignore_errors=True)  # roll back on failure
        return jsonify({"ok": False, "error": f"Failed to save files: {e}"}), 500

    print(f"[ocr-api] Added model '{name}' with {len(saved)} files.")
    return jsonify({"ok": True, "name": name, "saved": saved})


@app.route("/delete_model", methods=["POST"])
def delete_model():
    """Delete a model folder from Models/. Body: { "model": "<key>" }.

    Refuses to touch the built-in base model and guards against any path that
    resolves outside the Models/ directory."""
    data = request.get_json(silent=True) or {}
    key = (data.get("model") or "").strip()

    if not key:
        return jsonify({"ok": False, "error": "No model specified."}), 400
    if key == BASE_MODEL_KEY:
        return jsonify({"ok": False, "error": "The base model cannot be deleted."}), 400

    discovered = _discover_models()
    if key not in discovered:
        return jsonify({"ok": False, "error": f"Model '{key}' was not found."}), 404

    # Path-safety: the resolved folder must live directly inside Models/.
    target = os.path.realpath(discovered[key])
    models_root = os.path.realpath(MODELS_DIR)
    if os.path.dirname(target) != models_root:
        return jsonify({"ok": False, "error": "Refusing to delete outside the Models folder."}), 400

    # Free the model from memory if it was loaded.
    _models.pop(key, None)

    try:
        shutil.rmtree(target)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Could not delete: {e}"}), 500

    print(f"[ocr-api] Deleted model '{key}'.")
    return jsonify({"ok": True, "deleted": key})


@app.route("/rename_model", methods=["POST"])
def rename_model():
    """Rename a model folder in Models/. Body: { "model": "<key>", "newName": "<name>" }.

    Refuses to touch the base model and guards against paths outside Models/."""
    data = request.get_json(silent=True) or {}
    key = (data.get("model") or "").strip()
    raw_new = (data.get("newName") or "").strip()

    if not key:
        return jsonify({"ok": False, "error": "No model specified."}), 400
    if key == BASE_MODEL_KEY:
        return jsonify({"ok": False, "error": "The base model cannot be renamed."}), 400
    if not raw_new:
        return jsonify({"ok": False, "error": "Please provide a new name."}), 400

    # Sanitize the new name the same way as add_model.
    new_name = re.sub(r"[^A-Za-z0-9._-]+", "-", raw_new).strip("-._")
    if not new_name or new_name == BASE_MODEL_KEY:
        return jsonify({"ok": False, "error": "That model name is not allowed."}), 400

    discovered = _discover_models()
    if key not in discovered:
        return jsonify({"ok": False, "error": f"Model '{key}' was not found."}), 404
    if new_name == key:
        return jsonify({"ok": True, "name": new_name})  # nothing to do

    models_root = os.path.realpath(MODELS_DIR)
    src = os.path.realpath(discovered[key])
    dst = os.path.join(models_root, new_name)

    # Path-safety: source must be directly inside Models/, and so must the target.
    if os.path.dirname(src) != models_root:
        return jsonify({"ok": False, "error": "Refusing to rename outside the Models folder."}), 400
    if os.path.exists(dst):
        return jsonify({"ok": False, "error": f"A model named '{new_name}' already exists."}), 409

    # Drop any cached copy of the old key so it reloads fresh under the new name.
    _models.pop(key, None)

    try:
        os.rename(src, dst)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Could not rename: {e}"}), 500

    print(f"[ocr-api] Renamed model '{key}' -> '{new_name}'.")
    return jsonify({"ok": True, "name": new_name})


if __name__ == "__main__":
    # threaded=True so a large model upload doesn't block health checks.
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
