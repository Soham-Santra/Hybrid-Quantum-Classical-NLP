"""Shared utility functions for the framework."""

import os
import json
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from functools import wraps

import numpy as np


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
class Timer:
    """Context manager for timing code blocks.

    Usage::

        with Timer("Training") as t:
            train(model)
        print(t.elapsed)  # seconds
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.elapsed: float = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        if self.name:
            logger.info(f"[Timer] {self.name}: {self.elapsed:.2f}s")


def timed(func):
    """Decorator that logs execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with Timer(func.__name__):
            return func(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------
def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist, return the path."""
    os.makedirs(path, exist_ok=True)
    return path


def save_json(data: Any, path: str) -> None:
    """Save data to a JSON file."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    logger.debug(f"Saved JSON: {path}")


def load_json(path: str) -> Any:
    """Load data from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results(results: Dict, experiment_name: str, output_dir: str = "results") -> str:
    """Save experiment results to a JSON file with timestamp.

    Parameters
    ----------
    results : dict
        Experiment results including metrics, config, etc.
    experiment_name : str
        Name of the experiment.
    output_dir : str
        Base output directory.

    Returns
    -------
    str
        Path to the saved results file.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{timestamp}.json"
    path = os.path.join(output_dir, "logs", filename)
    save_json(results, path)
    logger.info(f"Results saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Metrics formatting
# ---------------------------------------------------------------------------
def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """Format a metrics dict as a readable string.

    Example output: "Acc=0.8542 | P=0.8510 | R=0.8490 | F1=0.8500"
    """
    parts = []
    short_names = {
        "accuracy": "Acc",
        "precision": "P",
        "recall": "R",
        "f1": "F1",
        "loss": "Loss",
    }
    for key, value in metrics.items():
        name = short_names.get(key, key)
        parts.append(f"{name}={value:.{precision}f}")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------
def get_device(preference: str = "cpu"):
    """Get the torch device, falling back to CPU if CUDA unavailable."""
    try:
        import torch
        if preference == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
        else:
            device = torch.device("cpu")
            if preference == "cuda":
                logger.warning("CUDA requested but not available, falling back to CPU.")
            logger.info(f"Using device: {device}")
        return device
    except ImportError:
        logger.warning("PyTorch not installed, returning 'cpu' string.")
        return "cpu"


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------
def count_parameters(model) -> Dict[str, int]:
    """Count trainable and total parameters in a PyTorch model.

    Returns
    -------
    dict
        {"total": int, "trainable": int, "frozen": int}
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


# ---------------------------------------------------------------------------
# Progress display
# ---------------------------------------------------------------------------
def print_banner(text: str, width: int = 70) -> None:
    """Print a formatted banner to the console."""
    border = "=" * width
    padding = (width - len(text) - 2) // 2
    logger.info(border)
    logger.info(f"{'':>{padding}} {text}")
    logger.info(border)


def print_table(headers: List[str], rows: List[List[Any]], title: str = "") -> str:
    """Format a table as a string using tabulate.

    Returns the formatted string and also logs it.
    """
    try:
        from tabulate import tabulate
        table_str = tabulate(rows, headers=headers, tablefmt="grid")
    except ImportError:
        # Fallback: simple formatting
        col_widths = [max(len(str(h)), *(len(str(r[i])) for r in rows))
                      for i, h in enumerate(headers)]
        lines = []
        header_line = " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
        lines.append(header_line)
        lines.append("-+-".join("-" * w for w in col_widths))
        for row in rows:
            lines.append(" | ".join(f"{str(v):<{w}}" for v, w in zip(row, col_widths)))
        table_str = "\n".join(lines)

    if title:
        logger.info(f"\n{title}\n{table_str}")
    return table_str
