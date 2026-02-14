"""Training loop with curriculum learning and early stopping."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from ..data.preprocessing import MolecularFeaturizer
from ..evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for Adaptive Spectral-Temporal GNN with curriculum learning."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        device: torch.device,
        config: Dict[str, Any]
    ):
        """Initialize trainer.

        Args:
            model: Neural network model.
            criterion: Loss function.
            optimizer: Optimizer.
            scheduler: Learning rate scheduler.
            device: Device to train on.
            config: Configuration dictionary.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config

        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_auc = 0.0
        self.patience_counter = 0

        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []

        # Curriculum learning settings
        self.use_curriculum = config['training'].get('use_curriculum', False)
        if self.use_curriculum:
            self.curriculum_epochs = config['training'].get('curriculum_epochs', 30)
            self.start_ratio = config['curriculum'].get('start_ratio', 0.3)
            self.end_ratio = config['curriculum'].get('end_ratio', 1.0)

    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        """Train for one epoch.

        Args:
            train_loader: Training data loader.

        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Compute curriculum sampling ratio
        if self.use_curriculum and self.current_epoch < self.curriculum_epochs:
            curriculum_ratio = self.start_ratio + (
                (self.end_ratio - self.start_ratio) *
                (self.current_epoch / self.curriculum_epochs)
            )
        else:
            curriculum_ratio = 1.0

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        for batch_graphs, batch_labels, batch_masks in pbar:
            # Move to device
            batch_graphs = batch_graphs.to(self.device)
            batch_labels = batch_labels.to(self.device)
            batch_masks = batch_masks.to(self.device)

            # Add Laplacian eigenvectors if not present
            if not hasattr(batch_graphs, 'laplacian_eigenvectors'):
                num_eigenvectors = self.config['model'].get('num_eigenvectors', 20)
                # Process each graph in batch
                eigenvectors_list = []
                ptr = batch_graphs.ptr if hasattr(batch_graphs, 'ptr') else None

                if ptr is not None:
                    for i in range(len(ptr) - 1):
                        start_idx = ptr[i]
                        end_idx = ptr[i + 1]
                        num_nodes = end_idx - start_idx

                        # Get edge indices for this graph
                        mask = (batch_graphs.edge_index[0] >= start_idx) & \
                               (batch_graphs.edge_index[0] < end_idx)
                        edge_index = batch_graphs.edge_index[:, mask] - start_idx

                        eigvecs = MolecularFeaturizer.compute_laplacian_eigenvectors(
                            edge_index, num_nodes.item(), k=num_eigenvectors
                        )
                        eigenvectors_list.append(eigvecs)

                    batch_graphs.laplacian_eigenvectors = torch.cat(
                        eigenvectors_list, dim=0
                    ).to(self.device)

            # Forward pass
            predictions, _ = self.model(batch_graphs)

            # Compute loss
            loss = self.criterion(predictions, batch_labels, batch_masks)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            grad_clip = self.config['training'].get('gradient_clip', 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def validate(self, val_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Validate the model.

        Args:
            val_loader: Validation data loader.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_masks = []

        with torch.no_grad():
            for batch_graphs, batch_labels, batch_masks in val_loader:
                batch_graphs = batch_graphs.to(self.device)
                batch_labels = batch_labels.to(self.device)
                batch_masks = batch_masks.to(self.device)

                # Add Laplacian eigenvectors
                if not hasattr(batch_graphs, 'laplacian_eigenvectors'):
                    num_eigenvectors = self.config['model'].get('num_eigenvectors', 20)
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
                        ).to(self.device)

                predictions, _ = self.model(batch_graphs)
                loss = self.criterion(predictions, batch_labels, batch_masks)

                total_loss += loss.item()

                all_predictions.append(torch.sigmoid(predictions).cpu().numpy())
                all_labels.append(batch_labels.cpu().numpy())
                all_masks.append(batch_masks.cpu().numpy())

        # Compute metrics
        predictions = np.concatenate(all_predictions, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        masks = np.concatenate(all_masks, axis=0)

        metrics = compute_metrics(predictions, labels, masks)
        metrics['loss'] = total_loss / len(val_loader)

        return metrics

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int,
        checkpoint_dir: str,
        early_stopping_patience: int = 15
    ) -> Dict[str, List[float]]:
        """Full training loop with early stopping.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            num_epochs: Number of training epochs.
            checkpoint_dir: Directory to save checkpoints.
            early_stopping_patience: Patience for early stopping.

        Returns:
            Dictionary of training history.
        """
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting training for {num_epochs} epochs...")
        logger.info(f"Model has {self.model.get_num_params():,} trainable parameters")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Training
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validation
            val_metrics = self.validate(val_loader)
            val_loss = val_metrics['loss']
            val_auc = val_metrics.get('mean_auc_roc', 0.0)

            self.val_losses.append(val_loss)
            self.val_aucs.append(val_auc)

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()

            # Logging
            logger.info(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, val_auc={val_auc:.4f}"
            )

            # Save best model
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.best_val_loss = val_loss
                self.patience_counter = 0

                best_model_path = checkpoint_path / "best_model.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_auc': val_auc,
                    'val_loss': val_loss,
                    'config': self.config
                }, best_model_path)
                logger.info(f"Saved best model with val_auc={val_auc:.4f}")
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Save final model
        final_model_path = checkpoint_path / "final_model.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, final_model_path)

        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_aucs': self.val_aucs
        }

        return history
