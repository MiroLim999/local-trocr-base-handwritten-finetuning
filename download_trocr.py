"""
download_trocr.py
Downloads the TrOCR processor and model from Hugging Face.
Model: microsoft/trocr-base-handwritten
"""

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


def main():
    model_name = "microsoft/trocr-base-handwritten"

    # Check GPU availability
    print("=" * 50)
    print("SYSTEM CHECK")
    print("=" * 50)
    cuda_available = torch.cuda.is_available()
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {cuda_available}")
    if cuda_available:
        print(f"GPU device      : {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not available. Model will run on CPU (slower).")
    print()

    # Download the processor (tokenizer + image processor)
    print("=" * 50)
    print("DOWNLOADING PROCESSOR")
    print("=" * 50)
    print(f"Downloading processor from: {model_name}")
    print("This handles image preprocessing and text decoding.")
    processor = TrOCRProcessor.from_pretrained(model_name)
    print("Processor downloaded successfully!\n")

    # Download the model (encoder-decoder architecture)
    print("=" * 50)
    print("DOWNLOADING MODEL")
    print("=" * 50)
    print(f"Downloading model from: {model_name}")
    print("This may take several minutes on first run (~1.2 GB)...")
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    print("Model downloaded successfully!\n")

    # Summary
    print("=" * 50)
    print("DOWNLOAD COMPLETE")
    print("=" * 50)
    print(f"Model       : {model_name}")
    print(f"Parameters  : {sum(p.numel() for p in model.parameters()):,}")
    print(f"CUDA ready  : {cuda_available}")
    print("\nYou can now run test_trocr.py to test the model.")


if __name__ == "__main__":
    main()
