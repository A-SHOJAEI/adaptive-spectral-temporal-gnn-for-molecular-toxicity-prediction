#!/usr/bin/env python
"""Training script for Adaptive Spectral-Temporal GNN."""

import sys
import argparse
import json
import logging
from pathlib import Path

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from adaptive_spectral_temporal_gnn_for_molecular_toxicity_prediction.data.loader import Tox21DataLoader
from adaptive_spectral_temporal_gnn_for_molecular_toxicity_prediction.models.model import AdaptiveSpectralTemporalGNN
from adaptive_spectral_temporal_gnn_for_molecular_toxicity_prediction.models.components import UncertaintyWeightedMultiTaskLoss
from adaptive_spectral_temporal_gnn_for_molecular_toxicity_prediction.training.trainer import Trainer
from adaptive_spectral_temporal_gnn_for_molecular_toxicity_prediction.utils.config import load_config, set_random_seeds
from adaptive_spectral_temporal_gnn_for_molecular_toxicity_prediction.evaluation.analysis import plot_training_curves

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Adaptive Spectral-Temporal GNN')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU device ID'
    )
    return parser.parse_args()


def main():
    """Main training function."""
    try:
        # Parse arguments
        args = parse_args()

        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)

        # Set random seeds for reproducibility
        seed = config.get('seed', 42)
        set_random_seeds(seed)
        logger.info(f"Set random seed to {seed}")

        # Setup device
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        # Initialize MLflow (wrapped in try-except)
        use_mlflow = config.get('logging', {}).get('use_mlflow', False)
        if use_mlflow:
            try:
                import mlflow
                experiment_name = config['logging'].get('experiment_name', 'adaptive_spectral_gnn')
                mlflow.set_experiment(experiment_name)
                mlflow.start_run()
                mlflow.log_params({
                    'learning_rate': config['training']['learning_rate'],
                    'batch_size': config['data']['batch_size'],
                    'num_layers': config['model']['num_layers'],
                    'hidden_dim': config['model']['hidden_dim'],
                    'use_adaptive_filters': config['model']['use_adaptive_filters'],
                    'use_curriculum': config['training']['use_curriculum']
                })
                logger.info("MLflow logging enabled")
            except Exception as e:
                logger.warning(f"MLflow initialization failed: {e}. Continuing without MLflow.")
                use_mlflow = False
        else:
            logger.info("MLflow logging disabled")

        # Load data
        logger.info("Loading Tox21 dataset...")
        data_loader = Tox21DataLoader(
            batch_size=config['data']['batch_size'],
            num_workers=config['data']['num_workers'],
            split_ratio=config['data']['split_ratio'],
            featurizer=config['data']['featurizer']
        )

        train_loader, val_loader, test_loader, task_names = data_loader.load_data()
        logger.info(f"Dataset loaded. Tasks: {task_names}")

        # Get input dimension from first batch
        sample_batch, _, _ = next(iter(train_loader))
        input_dim = sample_batch.x.shape[1]
        logger.info(f"Input feature dimension: {input_dim}")

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

        logger.info(f"Model initialized with {model.get_num_params():,} parameters")

        # Initialize loss function
        criterion = UncertaintyWeightedMultiTaskLoss(
            num_tasks=len(task_names),
            learn_weights=config['loss']['learn_task_weights'],
            regularization=config['loss']['uncertainty_regularization']
        ).to(device)

        # Initialize optimizer
        optimizer = AdamW(
            list(model.parameters()) + list(criterion.parameters()),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            betas=config['optimizer']['betas'],
            eps=config['optimizer']['eps']
        )

        # Initialize scheduler
        scheduler_type = config['scheduler']['type']
        if scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=config['scheduler']['t_max'],
                eta_min=config['scheduler']['eta_min']
            )
        elif scheduler_type == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            scheduler = None

        # Initialize trainer
        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config
        )

        # Create checkpoint directory
        checkpoint_dir = Path(config['training']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Train model
        logger.info("Starting training...")
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config['training']['num_epochs'],
            checkpoint_dir=str(checkpoint_dir),
            early_stopping_patience=config['training']['early_stopping_patience']
        )

        # Save training history
        results_dir = Path('results')
        results_dir.mkdir(parents=True, exist_ok=True)

        history_path = results_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"Saved training history to {history_path}")

        # Plot training curves
        plot_path = results_dir / 'training_curves.png'
        plot_training_curves(history, save_path=str(plot_path))

        # Log final metrics to MLflow
        if use_mlflow:
            try:
                mlflow.log_metric('best_val_auc', trainer.best_val_auc)
                mlflow.log_metric('best_val_loss', trainer.best_val_loss)
                mlflow.log_artifact(str(plot_path))
                mlflow.end_run()
            except Exception as e:
                logger.warning(f"MLflow logging failed: {e}")

        logger.info("Training completed successfully!")
        logger.info(f"Best validation AUC: {trainer.best_val_auc:.4f}")
        logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")

        # Save task names for evaluation
        task_names_path = results_dir / 'task_names.json'
        with open(task_names_path, 'w') as f:
            json.dump(task_names, f)

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
