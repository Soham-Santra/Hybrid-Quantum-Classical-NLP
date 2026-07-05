"""Visualization utilities for training curves, confusion matrices, and comparisons."""

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def plot_training_curves(history: dict, title: str = "Training Curves",
                         save_path: Optional[str] = None) -> None:
    """Plot training and validation loss/accuracy curves.

    Parameters
    ----------
    history : dict
        Must contain 'train_loss' and optionally 'val_loss', 'train_acc', 'val_acc'.
    title : str
        Plot title.
    save_path : str, optional
        If provided, save figure to this path.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    if "train_loss" in history:
        axes[0].plot(history["train_loss"], label="Train Loss", linewidth=2)
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} — Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    if "train_acc" in history:
        axes[1].plot(history["train_acc"], label="Train Acc", linewidth=2)
    if "val_acc" in history:
        axes[1].plot(history["val_acc"], label="Val Acc", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{title} — Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")

    plt.show()


def plot_confusion_matrix(y_true, y_pred, label_names: List[str],
                           title: str = "Confusion Matrix",
                           save_path: Optional[str] = None) -> None:
    """Plot confusion matrix as a heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(8, len(label_names)), max(6, len(label_names) * 0.8)))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names,
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")

    plt.show()


def plot_model_comparison(results: Dict[str, Dict], metric: str = "f1",
                           title: str = "Model Comparison",
                           save_path: Optional[str] = None) -> None:
    """Bar chart comparing models on a given metric.

    Parameters
    ----------
    results : dict
        {model_name: {"test_metrics": {"accuracy": ..., "f1": ...}}}
    metric : str
        Which metric to plot.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    models = []
    scores = []

    for model_name, model_results in results.items():
        if isinstance(model_results, dict) and "test_metrics" in model_results:
            models.append(model_name)
            scores.append(model_results["test_metrics"].get(metric, 0))

    if not models:
        logger.warning("No results to plot.")
        return

    fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.5), 6))

    colors = sns.color_palette("Set2", len(models))
    bars = ax.bar(models, scores, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels on bars
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{score:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.set_ylim(0, max(scores) * 1.15 if scores else 1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")

    plt.show()


def plot_all_results(all_results: Dict[str, Any],
                      save_dir: str = "results/figures") -> None:
    """Generate all comparison plots from experiment results."""
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    # Flatten results for comparison
    flat_results = {}

    for exp_type, exp_data in all_results.items():
        if isinstance(exp_data, dict) and "error" not in exp_data:
            for lang, lang_data in exp_data.items():
                if isinstance(lang_data, dict):
                    for model, model_data in lang_data.items():
                        if isinstance(model_data, dict) and "test_metrics" in model_data:
                            key = f"{exp_type}/{model}"
                            flat_results[key] = model_data

    if flat_results:
        # F1 comparison
        plot_model_comparison(
            flat_results, metric="f1",
            title="F1 Score Comparison — All Models",
            save_path=os.path.join(save_dir, "f1_comparison.png"),
        )

        # Accuracy comparison
        plot_model_comparison(
            flat_results, metric="accuracy",
            title="Accuracy Comparison — All Models",
            save_path=os.path.join(save_dir, "accuracy_comparison.png"),
        )

    logger.info(f"All plots saved to: {save_dir}")


def print_results_table(results: Dict[str, Any], title: str = "Results") -> str:
    """Print a formatted results table.

    Returns the table as a string.
    """
    try:
        from tabulate import tabulate
    except ImportError:
        logger.warning("tabulate not installed, using basic formatting.")
        return str(results)

    headers = ["Model", "Accuracy", "Precision", "Recall", "F1", "Params", "Time (s)"]
    rows = []

    for model_name, model_data in results.items():
        if isinstance(model_data, dict) and "test_metrics" in model_data:
            m = model_data["test_metrics"]
            rows.append([
                model_name,
                f"{m.get('accuracy', 0):.4f}",
                f"{m.get('precision', 0):.4f}",
                f"{m.get('recall', 0):.4f}",
                f"{m.get('f1', 0):.4f}",
                model_data.get("params", "?"),
                f"{model_data.get('history', {}).get('training_time', 0):.1f}",
            ])

    table = tabulate(rows, headers=headers, tablefmt="grid")
    print(f"\n{title}")
    print(table)
    return table
