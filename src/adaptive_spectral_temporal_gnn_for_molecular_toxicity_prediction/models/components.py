"""Custom model components including loss functions and layers."""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

logger = logging.getLogger(__name__)


class UncertaintyWeightedMultiTaskLoss(nn.Module):
    """Custom multi-task loss with learned uncertainty weighting.

    Based on "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
    (Kendall et al., 2018). Learns task-specific uncertainty parameters to automatically
    balance multiple toxicity prediction tasks.
    """

    def __init__(
        self,
        num_tasks: int,
        initial_log_vars: float = 0.0,
        learn_weights: bool = True,
        regularization: float = 0.01
    ):
        """Initialize uncertainty-weighted multi-task loss.

        Args:
            num_tasks: Number of prediction tasks.
            initial_log_vars: Initial value for log variance parameters.
            learn_weights: Whether to learn task weights.
            regularization: Regularization strength for log variances.
        """
        super().__init__()
        self.num_tasks = num_tasks
        self.learn_weights = learn_weights
        self.regularization = regularization

        if learn_weights:
            # Log variance parameters for each task
            self.log_vars = nn.Parameter(
                torch.ones(num_tasks) * initial_log_vars
            )
        else:
            self.register_buffer(
                'log_vars',
                torch.zeros(num_tasks)
            )

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor
    ) -> torch.Tensor:
        """Compute uncertainty-weighted multi-task loss.

        Args:
            predictions: Model predictions of shape (batch_size, num_tasks).
            targets: Ground truth labels of shape (batch_size, num_tasks).
            masks: Binary mask for valid labels of shape (batch_size, num_tasks).

        Returns:
            Weighted loss value.
        """
        losses = []

        for i in range(self.num_tasks):
            # Get valid samples for this task
            valid_mask = masks[:, i].bool()

            if valid_mask.sum() == 0:
                continue

            # Compute binary cross-entropy loss for this task
            pred_task = predictions[valid_mask, i]
            target_task = targets[valid_mask, i]

            task_loss = F.binary_cross_entropy_with_logits(
                pred_task,
                target_task,
                reduction='mean'
            )

            if self.learn_weights:
                # Weight by learned uncertainty
                precision = torch.exp(-self.log_vars[i])
                weighted_loss = precision * task_loss + self.log_vars[i]
            else:
                weighted_loss = task_loss

            losses.append(weighted_loss)

        if not losses:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        total_loss = torch.stack(losses).mean()

        # Add regularization on log variances to prevent collapse
        if self.learn_weights and self.regularization > 0:
            reg_loss = self.regularization * torch.mean(self.log_vars ** 2)
            total_loss = total_loss + reg_loss

        return total_loss

    def get_task_weights(self) -> torch.Tensor:
        """Get current task weights (precision values).

        Returns:
            Task weight tensor.
        """
        return torch.exp(-self.log_vars)


class AdaptiveSpectralFilter(nn.Module):
    """Adaptive spectral filter with learned attention over Laplacian eigenvectors.

    This module learns to adaptively weight different frequency components
    based on molecular substructure complexity.
    """

    def __init__(self, num_eigenvectors: int, hidden_dim: int, num_filters: int):
        """Initialize adaptive spectral filter.

        Args:
            num_eigenvectors: Number of Laplacian eigenvectors.
            hidden_dim: Hidden dimension size.
            num_filters: Number of spectral filters.
        """
        super().__init__()
        self.num_eigenvectors = num_eigenvectors
        self.num_filters = num_filters

        # Attention mechanism for eigenvector weighting
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_eigenvectors),
            nn.Softmax(dim=-1)
        )

        # Filter coefficients
        self.filter_weights = nn.Parameter(
            torch.randn(num_filters, num_eigenvectors)
        )

    def forward(
        self,
        node_features: torch.Tensor,
        eigenvectors: torch.Tensor
    ) -> torch.Tensor:
        """Apply adaptive spectral filtering.

        Args:
            node_features: Node feature matrix (num_nodes, hidden_dim).
            eigenvectors: Laplacian eigenvectors (num_nodes, num_eigenvectors).

        Returns:
            Filtered node features.
        """
        # Compute attention weights based on node features
        graph_repr = node_features.mean(dim=0, keepdim=True)  # (1, hidden_dim)
        attention_weights = self.attention(graph_repr)  # (1, num_eigenvectors)

        # Adaptive filter coefficients
        adaptive_filters = self.filter_weights * attention_weights  # (num_filters, num_eigenvectors)

        # Apply spectral filtering
        spectral_features = eigenvectors @ adaptive_filters.t()  # (num_nodes, num_filters)

        return spectral_features


class SpectralGraphConvLayer(MessagePassing):
    """Spectral graph convolution layer with adaptive filtering."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_eigenvectors: int = 20,
        num_filters: int = 16,
        use_adaptive: bool = True
    ):
        """Initialize spectral graph convolution layer.

        Args:
            in_channels: Input feature dimension.
            out_channels: Output feature dimension.
            num_eigenvectors: Number of Laplacian eigenvectors.
            num_filters: Number of spectral filters.
            use_adaptive: Whether to use adaptive filtering.
        """
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_adaptive = use_adaptive

        # Feature transformation
        self.lin = nn.Linear(in_channels, out_channels)

        if use_adaptive:
            self.adaptive_filter = AdaptiveSpectralFilter(
                num_eigenvectors,
                in_channels,
                num_filters
            )
            self.spectral_transform = nn.Linear(num_filters, out_channels)
        else:
            # Fixed spectral transformation
            self.spectral_transform = nn.Linear(num_eigenvectors, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        """Reset layer parameters."""
        self.lin.reset_parameters()
        if hasattr(self, 'spectral_transform'):
            self.spectral_transform.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        eigenvectors: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features (num_nodes, in_channels).
            edge_index: Edge indices.
            eigenvectors: Laplacian eigenvectors (num_nodes, num_eigenvectors).

        Returns:
            Updated node features.
        """
        # Standard graph convolution
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Compute normalization
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Message passing
        out = self.propagate(edge_index, x=x, norm=norm)
        out = self.lin(out)

        # Add spectral component if eigenvectors provided
        if eigenvectors is not None:
            if self.use_adaptive:
                spectral_features = self.adaptive_filter(x, eigenvectors)
            else:
                spectral_features = eigenvectors

            spectral_out = self.spectral_transform(spectral_features)
            out = out + spectral_out

        return out

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        """Construct messages.

        Args:
            x_j: Source node features.
            norm: Normalization coefficients.

        Returns:
            Messages.
        """
        return norm.view(-1, 1) * x_j
