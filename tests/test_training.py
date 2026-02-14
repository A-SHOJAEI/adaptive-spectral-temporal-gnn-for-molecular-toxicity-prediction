"""Tests for training utilities."""

import pytest
import torch
from torch.optim import AdamW

from adaptive_spectral_temporal_gnn_for_molecular_toxicity_prediction.training.trainer import Trainer
from adaptive_spectral_temporal_gnn_for_molecular_toxicity_prediction.evaluation.metrics import (
    compute_metrics,
    compute_calibration_error
)


def test_compute_metrics():
    """Test metrics computation."""
    predictions = torch.sigmoid(torch.randn(100, 12)).numpy()
    labels = torch.randint(0, 2, (100, 12)).float().numpy()
    masks = torch.ones(100, 12).numpy()

    metrics = compute_metrics(predictions, labels, masks)

    assert 'mean_auc_roc' in metrics
    assert 'mean_accuracy' in metrics
    assert metrics['mean_auc_roc'] >= 0 and metrics['mean_auc_roc'] <= 1


def test_compute_calibration_error():
    """Test calibration error computation."""
    predictions = torch.sigmoid(torch.randn(100, 12)).numpy()
    labels = torch.randint(0, 2, (100, 12)).float().numpy()
    masks = torch.ones(100, 12).numpy()

    ece = compute_calibration_error(predictions, labels, masks, num_bins=10)

    assert isinstance(ece, float)
    assert ece >= 0 and ece <= 1


def test_trainer_initialization(sample_model, sample_loss, device):
    """Test trainer initialization."""
    optimizer = AdamW(sample_model.parameters(), lr=0.001)

    config = {
        'training': {
            'use_curriculum': False,
            'gradient_clip': 1.0
        },
        'model': {
            'num_eigenvectors': 10
        }
    }

    trainer = Trainer(
        model=sample_model,
        criterion=sample_loss,
        optimizer=optimizer,
        scheduler=None,
        device=device,
        config=config
    )

    assert trainer.current_epoch == 0
    assert trainer.best_val_loss == float('inf')
