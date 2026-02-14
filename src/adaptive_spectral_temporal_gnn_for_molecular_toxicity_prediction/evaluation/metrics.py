"""Evaluation metrics for molecular toxicity prediction."""

import logging
from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

logger = logging.getLogger(__name__)


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Compute evaluation metrics for multi-task prediction.

    Args:
        predictions: Predicted probabilities of shape (num_samples, num_tasks).
        labels: Ground truth labels of shape (num_samples, num_tasks).
        masks: Binary mask for valid labels of shape (num_samples, num_tasks).
        threshold: Classification threshold.

    Returns:
        Dictionary of computed metrics.
    """
    num_tasks = predictions.shape[1]
    metrics = {}

    # Per-task metrics
    auc_rocs = []
    auc_prs = []
    accuracies = []
    f1_scores = []
    precisions = []
    recalls = []

    for task_idx in range(num_tasks):
        # Get valid samples for this task
        valid_mask = masks[:, task_idx].astype(bool)

        if valid_mask.sum() < 10:  # Skip if too few samples
            continue

        task_preds = predictions[valid_mask, task_idx]
        task_labels = labels[valid_mask, task_idx]

        # Check if we have both classes
        if len(np.unique(task_labels)) < 2:
            continue

        # AUC-ROC
        try:
            auc_roc = roc_auc_score(task_labels, task_preds)
            auc_rocs.append(auc_roc)
        except Exception as e:
            logger.warning(f"Failed to compute AUC-ROC for task {task_idx}: {e}")

        # AUC-PR
        try:
            auc_pr = average_precision_score(task_labels, task_preds)
            auc_prs.append(auc_pr)
        except Exception as e:
            logger.warning(f"Failed to compute AUC-PR for task {task_idx}: {e}")

        # Classification metrics
        task_preds_binary = (task_preds >= threshold).astype(int)

        try:
            acc = accuracy_score(task_labels, task_preds_binary)
            accuracies.append(acc)
        except Exception as e:
            logger.warning(f"Failed to compute accuracy for task {task_idx}: {e}")

        try:
            f1 = f1_score(task_labels, task_preds_binary, zero_division=0)
            f1_scores.append(f1)
        except Exception as e:
            logger.warning(f"Failed to compute F1 for task {task_idx}: {e}")

        try:
            prec = precision_score(task_labels, task_preds_binary, zero_division=0)
            precisions.append(prec)
        except Exception as e:
            logger.warning(f"Failed to compute precision for task {task_idx}: {e}")

        try:
            rec = recall_score(task_labels, task_preds_binary, zero_division=0)
            recalls.append(rec)
        except Exception as e:
            logger.warning(f"Failed to compute recall for task {task_idx}: {e}")

    # Aggregate metrics
    if auc_rocs:
        metrics['mean_auc_roc'] = np.mean(auc_rocs)
        metrics['per_task_auc_roc'] = auc_rocs
    else:
        metrics['mean_auc_roc'] = 0.0
        metrics['per_task_auc_roc'] = []

    if auc_prs:
        metrics['mean_auc_pr'] = np.mean(auc_prs)
        metrics['per_task_auc_pr'] = auc_prs

    if accuracies:
        metrics['mean_accuracy'] = np.mean(accuracies)
        metrics['per_task_accuracy'] = accuracies

    if f1_scores:
        metrics['mean_f1_score'] = np.mean(f1_scores)
        metrics['per_task_f1_score'] = f1_scores

    if precisions:
        metrics['mean_precision'] = np.mean(precisions)

    if recalls:
        metrics['mean_recall'] = np.mean(recalls)

    return metrics


def compute_calibration_error(
    predictions: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
    num_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE).

    Args:
        predictions: Predicted probabilities.
        labels: Ground truth labels.
        masks: Binary mask for valid labels.
        num_bins: Number of bins for calibration.

    Returns:
        Expected Calibration Error.
    """
    # Flatten arrays and filter by mask
    valid_mask = masks.flatten().astype(bool)
    preds_flat = predictions.flatten()[valid_mask]
    labels_flat = labels.flatten()[valid_mask]

    if len(preds_flat) == 0:
        return 0.0

    # Create bins
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    total_samples = len(preds_flat)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Samples in this bin
        in_bin = (preds_flat > bin_lower) & (preds_flat <= bin_upper)
        prop_in_bin = in_bin.sum() / total_samples

        if prop_in_bin > 0:
            accuracy_in_bin = labels_flat[in_bin].mean()
            avg_confidence_in_bin = preds_flat[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece
