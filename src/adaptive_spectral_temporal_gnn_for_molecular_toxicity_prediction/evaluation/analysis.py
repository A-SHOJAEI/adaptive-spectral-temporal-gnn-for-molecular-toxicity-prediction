"""Results analysis and visualization."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> None:
    """Plot training and validation loss curves.

    Args:
        history: Dictionary containing training history.
        save_path: Path to save the plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Loss curves
    axes[0].plot(history['train_losses'], label='Training Loss', linewidth=2)
    axes[0].plot(history['val_losses'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # AUC curves
    if 'val_aucs' in history:
        axes[1].plot(history['val_aucs'], label='Validation AUC-ROC',
                    linewidth=2, color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('AUC-ROC')
        axes[1].set_title('Validation AUC-ROC')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training curves to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_per_task_performance(
    metrics: Dict[str, any],
    task_names: List[str],
    save_path: Optional[str] = None
) -> None:
    """Plot per-task performance metrics.

    Args:
        metrics: Dictionary of computed metrics.
        task_names: List of task names.
        save_path: Path to save the plot.
    """
    per_task_auc = metrics.get('per_task_auc_roc', [])
    per_task_acc = metrics.get('per_task_accuracy', [])
    per_task_f1 = metrics.get('per_task_f1_score', [])

    if not per_task_auc:
        logger.warning("No per-task metrics to plot")
        return

    # Ensure task_names matches length
    num_tasks = len(per_task_auc)
    if len(task_names) != num_tasks:
        task_names = [f"Task {i}" for i in range(num_tasks)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # AUC-ROC
    axes[0].bar(range(num_tasks), per_task_auc, color='steelblue', alpha=0.8)
    axes[0].axhline(y=np.mean(per_task_auc), color='red', linestyle='--',
                    label=f'Mean: {np.mean(per_task_auc):.3f}')
    axes[0].set_xlabel('Task')
    axes[0].set_ylabel('AUC-ROC')
    axes[0].set_title('Per-Task AUC-ROC')
    axes[0].set_xticks(range(num_tasks))
    axes[0].set_xticklabels(task_names, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    if per_task_acc:
        axes[1].bar(range(num_tasks), per_task_acc, color='forestgreen', alpha=0.8)
        axes[1].axhline(y=np.mean(per_task_acc), color='red', linestyle='--',
                       label=f'Mean: {np.mean(per_task_acc):.3f}')
        axes[1].set_xlabel('Task')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Per-Task Accuracy')
        axes[1].set_xticks(range(num_tasks))
        axes[1].set_xticklabels(task_names, rotation=45, ha='right')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    # F1 Score
    if per_task_f1:
        axes[2].bar(range(num_tasks), per_task_f1, color='coral', alpha=0.8)
        axes[2].axhline(y=np.mean(per_task_f1), color='red', linestyle='--',
                       label=f'Mean: {np.mean(per_task_f1):.3f}')
        axes[2].set_xlabel('Task')
        axes[2].set_ylabel('F1 Score')
        axes[2].set_title('Per-Task F1 Score')
        axes[2].set_xticks(range(num_tasks))
        axes[2].set_xticklabels(task_names, rotation=45, ha='right')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved per-task performance plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_calibration_curve(
    predictions: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
    num_bins: int = 10,
    save_path: Optional[str] = None
) -> None:
    """Plot calibration curve for model predictions.

    Args:
        predictions: Predicted probabilities.
        labels: Ground truth labels.
        masks: Binary mask for valid labels.
        num_bins: Number of bins.
        save_path: Path to save the plot.
    """
    # Flatten and filter
    valid_mask = masks.flatten().astype(bool)
    preds_flat = predictions.flatten()[valid_mask]
    labels_flat = labels.flatten()[valid_mask]

    # Create bins
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    bin_confidences = []
    bin_accuracies = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (preds_flat > bin_lower) & (preds_flat <= bin_upper)

        if in_bin.sum() > 0:
            bin_confidences.append(preds_flat[in_bin].mean())
            bin_accuracies.append(labels_flat[in_bin].mean())

    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.plot(bin_confidences, bin_accuracies, 'o-', linewidth=2,
             markersize=8, label='Model')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved calibration curve to {save_path}")
    else:
        plt.show()

    plt.close()
