"""
metrics.py
Lightweight, dependency-free evaluation metrics for OCR.

Provides:
  - CER  (Character Error Rate)
  - WER  (Word Error Rate)
  - Exact-match accuracy (whole-string)

All rates are computed in aggregate (total edits / total reference units)
which is the standard way to report corpus-level CER/WER.
"""

import os
from datetime import datetime


def _levenshtein(ref, hyp):
    """Edit distance between two sequences (lists of chars or words)."""
    m, n = len(ref), len(hyp)
    if m == 0:
        return n
    if n == 0:
        return m

    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + cost,  # substitution / match
            )
        prev = curr
    return prev[n]


def compute_metrics(references, hypotheses):
    """
    Compute corpus-level CER, WER, and exact-match accuracy.

    Args:
        references: list of ground-truth strings
        hypotheses: list of predicted strings (same length / order)

    Returns:
        dict with keys: cer, wer, accuracy, exact, total
    """
    total_char_err = 0
    total_chars = 0
    total_word_err = 0
    total_words = 0
    exact = 0

    for ref, hyp in zip(references, hypotheses):
        ref = str(ref)
        hyp = str(hyp)

        # Character level
        total_char_err += _levenshtein(list(ref), list(hyp))
        total_chars += len(ref)

        # Word level
        ref_words = ref.split()
        hyp_words = hyp.split()
        total_word_err += _levenshtein(ref_words, hyp_words)
        total_words += len(ref_words)

        # Exact whole-string match
        if ref == hyp:
            exact += 1

    n = len(references)
    return {
        "cer": (total_char_err / total_chars) if total_chars else 0.0,
        "wer": (total_word_err / total_words) if total_words else 0.0,
        "accuracy": (exact / n) if n else 0.0,
        "exact": exact,
        "total": n,
    }


def print_metrics(metrics, title="EVALUATION METRICS"):
    """Pretty-print a metrics dict produced by compute_metrics()."""
    print("=" * 60)
    print(title)
    print("=" * 60)
    print(f"  Samples evaluated : {metrics['total']}")
    print(f"  Exact match       : {metrics['exact']}/{metrics['total']} "
          f"({metrics['accuracy'] * 100:.2f}%)")
    print(f"  CER               : {metrics['cer'] * 100:.2f}%")
    print(f"  WER               : {metrics['wer'] * 100:.2f}%")
    print("=" * 60)


def save_metrics_png(metrics, subfolder, model_label, base_dir="Evaluation Metrics"):
    """
    Save a metrics dict as a PNG image under:
        <base_dir>/<subfolder>/metrics_<timestamp>.png

    The image contains a bar chart of CER / WER / Accuracy plus a small
    summary table (model, samples, exact matches).

    Args:
        metrics: dict from compute_metrics()
        subfolder: e.g. "base" or "finetuned"
        model_label: human-readable model name shown on the chart
        base_dir: top-level folder (default "Evaluation Metrics")

    Returns:
        The path to the written PNG file.
    """
    # Import here so the rest of metrics.py stays dependency-free.
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend (no display needed)
    import matplotlib.pyplot as plt

    out_dir = os.path.join(base_dir, subfolder)
    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"metrics_{timestamp}.png")

    labels = ["CER", "WER", "Accuracy"]
    values = [
        metrics["cer"] * 100,
        metrics["wer"] * 100,
        metrics["accuracy"] * 100,
    ]
    colors = ["#e76f51", "#f4a261", "#2a9d8f"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=colors, width=0.6)

    ax.set_ylabel("Percent (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Evaluation Metrics — {model_label}", fontsize=12, fontweight="bold")

    # Value labels on top of each bar
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=10)

    # Summary line under the chart
    summary = (f"Samples: {metrics['total']}    "
               f"Exact matches: {metrics['exact']}/{metrics['total']}    "
               f"{timestamp}")
    ax.text(0.5, -0.12, summary, transform=ax.transAxes,
            ha="center", va="top", fontsize=9, color="#555555")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Metrics saved to: {os.path.abspath(out_path)}")
    return out_path
