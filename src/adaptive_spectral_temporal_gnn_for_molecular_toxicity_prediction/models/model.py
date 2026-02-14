"""Main model architecture for Adaptive Spectral-Temporal GNN."""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool, global_max_pool

from .components import SpectralGraphConvLayer

logger = logging.getLogger(__name__)


class AdaptiveSpectralTemporalGNN(nn.Module):
    """Adaptive Spectral-Temporal Graph Neural Network for molecular toxicity prediction.

    Combines spectral graph convolution with adaptive filtering based on molecular
    substructure complexity. Designed for multi-task toxicity prediction.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_tasks: int = 12,
        num_eigenvectors: int = 20,
        num_spectral_filters: int = 16,
        dropout: float = 0.3,
        pooling: str = "mean",
        use_adaptive_filters: bool = True,
        use_attention: bool = True
    ):
        """Initialize Adaptive Spectral-Temporal GNN.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden dimension size.
            num_layers: Number of graph convolution layers.
            num_tasks: Number of prediction tasks.
            num_eigenvectors: Number of Laplacian eigenvectors.
            num_spectral_filters: Number of spectral filters.
            dropout: Dropout probability.
            pooling: Graph pooling method ('mean', 'max', or 'mean_max').
            use_adaptive_filters: Whether to use adaptive spectral filtering.
            use_attention: Whether to use attention mechanism.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_tasks = num_tasks
        self.dropout = dropout
        self.pooling = pooling
        self.use_adaptive_filters = use_adaptive_filters
        self.use_attention = use_attention

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Spectral graph convolution layers
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            self.conv_layers.append(
                SpectralGraphConvLayer(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    num_eigenvectors=num_eigenvectors,
                    num_filters=num_spectral_filters,
                    use_adaptive=use_adaptive_filters
                )
            )

        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])

        # Attention pooling
        if use_attention:
            self.attention_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1)
            )

        # Determine output dimension after pooling
        if pooling == "mean_max":
            graph_dim = hidden_dim * 2
        else:
            graph_dim = hidden_dim

        # Output layers for multi-task prediction
        self.output_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(graph_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(num_tasks)
        ])

    def forward(
        self,
        batch: Batch,
        return_embeddings: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            batch: Batched graph data.
            return_embeddings: Whether to return graph embeddings.

        Returns:
            Predictions tensor of shape (batch_size, num_tasks).
            Optionally graph embeddings.
        """
        x = batch.x
        edge_index = batch.edge_index
        batch_idx = batch.batch

        # Get Laplacian eigenvectors if available
        eigenvectors = getattr(batch, 'laplacian_eigenvectors', None)

        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)

        # Apply spectral graph convolution layers
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.batch_norms)):
            x_new = conv(x, edge_index, eigenvectors)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)

            # Residual connection
            if i > 0:
                x = x + x_new
            else:
                x = x_new

        # Graph-level pooling
        if self.pooling == "mean":
            graph_emb = global_mean_pool(x, batch_idx)
        elif self.pooling == "max":
            graph_emb = global_max_pool(x, batch_idx)
        elif self.pooling == "mean_max":
            mean_pool = global_mean_pool(x, batch_idx)
            max_pool = global_max_pool(x, batch_idx)
            graph_emb = torch.cat([mean_pool, max_pool], dim=1)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        # Attention-based pooling (if enabled)
        if self.use_attention:
            attention_scores = self.attention_layer(x)
            attention_weights = torch.softmax(attention_scores, dim=0)
            attention_pool = global_mean_pool(x * attention_weights, batch_idx)
            graph_emb = graph_emb + attention_pool

        # Multi-task predictions
        predictions = []
        for output_layer in self.output_layers:
            task_pred = output_layer(graph_emb)
            predictions.append(task_pred)

        predictions = torch.cat(predictions, dim=1)  # (batch_size, num_tasks)

        if return_embeddings:
            return predictions, graph_emb
        return predictions, None

    def get_num_params(self) -> int:
        """Get total number of trainable parameters.

        Returns:
            Number of parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
