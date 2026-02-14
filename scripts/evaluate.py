#!/usr/bin/env python
"""Evaluation script for Adaptive Spectral-Temporal GNN."""

import sys
import argparse
import json
import logging
from pathlib import Path

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from tqdm import tqdm

from adaptive_spectral_temporal_gnn_for_molecular_toxicity_prediction.data.loader import Tox21DataLoader
from adaptive_spectral_temporal_gnn_for_molecular_toxicity_prediction.models.model import AdaptiveSpectralTemporalGNN
from adaptive_spectral_temporal_gnn_for_molecular_toxicity_prediction.data.preprocessing import MolecularFeaturizer
from adaptive_spectral_temporal_gnn_for_molecular_toxicity_prediction.evaluation.metrics import (
    compute_metrics,
    compute_calibration_error
)
from adaptive_spectral_temporal_gnn_for_molecular_toxicity_prediction.evaluation.analysis import (
    plot_per_task_performance,
    plot_calibration_curve
)
from adaptive_spectral_temporal_gnn_for_molecular_toxicity_prediction.utils.config import load_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate Adaptive Spectral-Temporal GNN')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best_model.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Dataset split to evaluate'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU device ID'
    )
    return parser.parse_args()


def evaluate_model(model, data_loader, device, config):
    """Evaluate model on dataset.

    Args:
        model: Trained model.
        data_loader: Data loader.
        device: Device to use.
        config: Configuration dict.

    Returns:
        Predictions, labels, and masks arrays.
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_masks = []

    with torch.no_grad():
        for batch_graphs, batch_labels, batch_masks in tqdm(data_loader, desc="Evaluating"):
            batch_graphs = batch_graphs.to(device)

            # Add Laplacian eigenvectors if needed
            if not hasattr(batch_graphs, 'laplacian_eigenvectors'):
                num_eigenvectors = config['model'].get('num_eigenvectors', 20)
                eigenvectors_list = []
                ptr = batch_graphs.ptr if hasattr(batch_graphs, 'ptr') else None

                if ptr is not None:
                    for i in range(len(ptr) - 1):
                        start_idx = ptr[i]
                        end_idx = ptr[i + 1]
                        num_nodes = end_idx - start_idx

                        mask = (batch_graphs.edge_index[0] >= start_idx) & \
                               (batch_graphs.edge_index[0] < end_idx)
                        edge_index = batch_graphs.edge_index[:, mask] - start_idx

                        eigvecs = MolecularFeaturizer.compute_laplacian_eigenvectors(
                            edge_index, num_nodes.item(), k=num_eigenvectors
                        )
                        eigenvectors_list.append(eigvecs)

                    batch_graphs.laplacian_eigenvectors = torch.cat(
                        eigenvectors_list, dim=0
                    ).to(device)

            predictions, _ = model(batch_graphs)
            predictions = torch.sigmoid(predictions)

            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(batch_labels.numpy())
            all_masks.append(batch_masks.numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    masks = np.concatenate(all_masks, axis=0)

    return predictions, labels, masks


def main():
    """Main evaluation function."""
    try:
        # Parse arguments
        args = parse_args()

        # Check if checkpoint exists
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {args.checkpoint}")
            logger.info("Please train the model first using: python scripts/train.py")
            sys.exit(1)

        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)

        # Setup device
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        # Load checkpoint
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load data
        logger.info("Loading Tox21 dataset...")
        data_loader = Tox21DataLoader(
            batch_size=config['data']['batch_size'],
            num_workers=config['data']['num_workers'],
            split_ratio=config['data']['split_ratio'],
            featurizer=config['data']['featurizer']
        )

        train_loader, val_loader, test_loader, task_names = data_loader.load_data()

        # Select appropriate loader
        if args.split == 'train':
            eval_loader = train_loader
        elif args.split == 'val':
            eval_loader = val_loader
        else:
            eval_loader = test_loader

        logger.info(f"Evaluating on {args.split} split")

        # Get input dimension
        sample_batch, _, _ = next(iter(eval_loader))
        input_dim = sample_batch.x.shape[1]

        # Initialize model
        logger.info("Initializing model...")
        model = AdaptiveSpectralTemporalGNN(
            input_dim=input_dim,
            hidden_dim=config['model']['hidden_dim'],
            num_layers=config['model']['num_layers'],
            num_tasks=len(task_names),
            num_eigenvectors=config['model']['num_eigenvectors'],
            num_spectral_filters=config['model']['num_spectral_filters'],
            dropout=config['model']['dropout'],
            pooling=config['model']['pooling'],
            use_adaptive_filters=config['model']['use_adaptive_filters'],
            use_attention=config['model']['use_attention']
        ).to(device)

        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Model loaded successfully")

        # Evaluate
        predictions, labels, masks = evaluate_model(model, eval_loader, device, config)

        # Compute metrics
        logger.info("Computing metrics...")
        metrics = compute_metrics(predictions, labels, masks)

        # Compute calibration error
        calibration_error = compute_calibration_error(
            predictions,
            labels,
            masks,
            num_bins=config['evaluation'].get('calibration_bins', 10)
        )
        metrics['calibration_error'] = calibration_error

        # Print results
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Mean AUC-ROC: {metrics.get('mean_auc_roc', 0.0):.4f}")
        logger.info(f"Mean AUC-PR: {metrics.get('mean_auc_pr', 0.0):.4f}")
        logger.info(f"Mean Accuracy: {metrics.get('mean_accuracy', 0.0):.4f}")
        logger.info(f"Mean F1 Score: {metrics.get('mean_f1_score', 0.0):.4f}")
        logger.info(f"Mean Precision: {metrics.get('mean_precision', 0.0):.4f}")
        logger.info(f"Mean Recall: {metrics.get('mean_recall', 0.0):.4f}")
        logger.info(f"Calibration Error: {calibration_error:.4f}")
        logger.info("=" * 60)

        # Per-task results
        if 'per_task_auc_roc' in metrics:
            logger.info("\nPer-Task AUC-ROC:")
            for i, (task_name, auc) in enumerate(zip(task_names, metrics['per_task_auc_roc'])):
                logger.info(f"  {task_name}: {auc:.4f}")

        # Save results
        results_dir = Path('results')
        results_dir.mkdir(parents=True, exist_ok=True)

        # Convert numpy arrays to lists for JSON serialization
        metrics_serializable = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                metrics_serializable[key] = value.tolist()
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.float64):
                metrics_serializable[key] = [float(v) for v in value]
            else:
                metrics_serializable[key] = value

        results_path = results_dir / f'{args.split}_results.json'
        with open(results_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        logger.info(f"\nSaved results to {results_path}")

        # Generate visualizations
        logger.info("Generating visualizations...")

        # Per-task performance plot
        plot_path = results_dir / f'{args.split}_per_task_performance.png'
        plot_per_task_performance(metrics, task_names, save_path=str(plot_path))

        # Calibration curve
        calibration_path = results_dir / f'{args.split}_calibration_curve.png'
        plot_calibration_curve(predictions, labels, masks, save_path=str(calibration_path))

        logger.info("Evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
