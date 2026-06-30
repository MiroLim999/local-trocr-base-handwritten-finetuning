"""
train_trocr.py
Fine-tunes TrOCR (microsoft/trocr-base-handwritten) on your handwritten names dataset.

Dataset structure expected:
  dataset/
    manifest.csv                     (filename,label,split,...)
    train/syn_000001.png ...
    val/syn_000002.png ...

Usage:
  python train_trocr.py

Adjust hyperparameters in the CONFIG section below.
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

import time
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    default_data_collator,
)
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

# ============================================================
# CONFIG - Adjust these values as needed
# ============================================================
MODEL_NAME = "microsoft/trocr-base-handwritten"

# Paths
MANIFEST_CSV = os.path.join("dataset", "manifest.csv")
TRAIN_IMG_DIR = os.path.join("dataset", "train")
VAL_IMG_DIR = os.path.join("dataset", "val")

# Training hyperparameters
EPOCHS = 5
BATCH_SIZE = 8                # Adjust based on your GPU VRAM (RTX 4050 = 6GB)
LEARNING_RATE = 5e-5
MAX_LABEL_LENGTH = 32         # Max characters in a label
NUM_WORKERS = 2               # DataLoader workers

# Dataset subset (set to None to use ALL data)
# Your dataset is small (450 train / 50 val), so use all of it.
TRAIN_SUBSET = None          # Use first N training samples (None = all)
VAL_SUBSET = None            # Use first N validation samples (None = all)

# Output
SAVE_DIR = "trocr-finetuned"  # Where to save the fine-tuned model
# ============================================================


class HandwrittenDataset(Dataset):
    """Custom dataset for handwritten text images with labels."""

    def __init__(self, csv_path, img_dir, processor, max_label_length, split, subset=None):
        self.processor = processor
        self.max_label_length = max_label_length
        self.img_dir = img_dir

        # Load CSV. keep_default_na=False prevents the literal label "None"
        # from being parsed as a missing value (NaN).
        df = pd.read_csv(csv_path, keep_default_na=False)

        # Keep only rows for this split (train / val)
        df = df[df["split"] == split]

        # Clean: remove rows with empty or "UNREADABLE" labels
        df = df[df["label"].astype(str).str.strip() != ""]
        df = df[df["label"].astype(str).str.upper() != "UNREADABLE"]
        df = df.reset_index(drop=True)

        # Apply subset limit
        if subset is not None:
            df = df.head(subset)

        self.data = df
        print(f"  Loaded {len(self.data)} '{split}' samples from {csv_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        filename = row["filename"]
        label = str(row["label"]).strip()

        # Load and preprocess image
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path).convert("RGB")

        # Process image -> pixel_values
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        # Tokenize label -> labels (decoder input)
        labels = self.processor.tokenizer(
            label,
            padding="max_length",
            max_length=self.max_label_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        # Replace padding token id with -100 so it's ignored in loss
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }


def evaluate(model, val_loader, device):
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            total_loss += outputs.loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def main():
    print("=" * 60)
    print("TrOCR FINE-TUNING")
    print("=" * 60)

    # --- Device setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # --- Verify paths ---
    for path, desc in [(MANIFEST_CSV, "Manifest CSV"),
                       (TRAIN_IMG_DIR, "Train images"), (VAL_IMG_DIR, "Val images")]:
        if not os.path.exists(path):
            print(f"ERROR: {desc} not found at '{path}'")
            return
    print("All dataset paths verified.\n")

    # --- Load processor and model ---
    print("Loading processor and model...")
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME, local_files_only=True)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME, local_files_only=True)

    # Configure model for fine-tuning.
    # IMPORTANT: TrOCR is pretrained to start decoding from the EOS/SEP token
    # (id 2), NOT the CLS token (id 0). Overriding decoder_start_token_id with
    # cls_token_id trains the decoder under one convention while generate()
    # uses another (the generation_config keeps id 2), which produces garbage
    # output at inference (CER/WER blow up). Keep the pretrained convention and
    # sync model.config and generation_config so training == inference.
    model.config.decoder_start_token_id = processor.tokenizer.sep_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id

    # Keep generation_config consistent with the training-time config.
    model.generation_config.decoder_start_token_id = model.config.decoder_start_token_id
    model.generation_config.eos_token_id = model.config.eos_token_id
    model.generation_config.pad_token_id = model.config.pad_token_id
    model.generation_config.max_length = MAX_LABEL_LENGTH

    model.to(device)
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # --- Create datasets ---
    print("Preparing datasets...")
    train_dataset = HandwrittenDataset(
        MANIFEST_CSV, TRAIN_IMG_DIR, processor, MAX_LABEL_LENGTH, split="train", subset=TRAIN_SUBSET
    )
    val_dataset = HandwrittenDataset(
        MANIFEST_CSV, VAL_IMG_DIR, processor, MAX_LABEL_LENGTH, split="val", subset=VAL_SUBSET
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(f"\n  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print()

    # --- Optimizer and scaler for mixed precision ---
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()
    use_amp = device.type == "cuda"

    # --- Training loop ---
    print("=" * 60)
    print("TRAINING START")
    print("=" * 60)
    print(f"  Epochs:         {EPOCHS}")
    print(f"  Batch size:     {BATCH_SIZE}")
    print(f"  Learning rate:  {LEARNING_RATE}")
    print(f"  Train samples:  {len(train_dataset)}")
    print(f"  Val samples:    {len(val_dataset)}")
    print(f"  Mixed precision: {use_amp}")
    print()

    best_val_loss = float("inf")
    training_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        epoch_start = time.time()

        # Training progress bar
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{EPOCHS} [Train]",
            unit="batch",
            bar_format="{l_bar}{bar:30}{r_bar}",
        )

        for batch in pbar:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            if use_amp:
                with autocast():
                    outputs = model(pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            # Update progress bar with current loss
            avg_loss = epoch_loss / num_batches
            pbar.set_postfix(loss=f"{avg_loss:.4f}")

        pbar.close()

        # --- Validation with progress bar ---
        avg_train_loss = epoch_loss / num_batches

        val_pbar = tqdm(
            val_loader,
            desc=f"Epoch {epoch}/{EPOCHS} [Val]  ",
            unit="batch",
            bar_format="{l_bar}{bar:30}{r_bar}",
        )
        model.eval()
        total_val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_pbar:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(pixel_values=pixel_values, labels=labels)
                total_val_loss += outputs.loss.item()
                val_batches += 1
                val_pbar.set_postfix(loss=f"{total_val_loss / val_batches:.4f}")
        val_pbar.close()

        val_loss = total_val_loss / max(val_batches, 1)

        # Epoch timing
        epoch_time = time.time() - epoch_start
        elapsed_total = time.time() - training_start
        remaining = (elapsed_total / epoch) * (EPOCHS - epoch)

        print(f"\n  Epoch {epoch} Summary:")
        print(f"    Train Loss:     {avg_train_loss:.4f}")
        print(f"    Val Loss:       {val_loss:.4f}")
        print(f"    Epoch Time:     {epoch_time:.0f}s")
        print(f"    Elapsed Total:  {elapsed_total:.0f}s")
        print(f"    ETA Remaining:  {remaining:.0f}s (~{remaining/60:.1f} min)")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(SAVE_DIR)
            processor.save_pretrained(SAVE_DIR)
            print(f"    ✓ Best model saved to '{SAVE_DIR}/'")
        print()

    # --- Done ---
    total_time = time.time() - training_start
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Total training time:  {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"Model saved to: {os.path.abspath(SAVE_DIR)}")
    print("\nTo test the fine-tuned model, run:")
    print(f"  python test_finetuned.py")


if __name__ == "__main__":
    main()
