"""Pytest fixtures for testing."""

import pytest
import torch
import numpy as np
from torch_geometric.data import Data

from adaptive_spectral_temporal_gnn_for_molecular_toxicity_prediction.models.model import (
    AdaptiveSpectralTemporalGNN
)
from adaptive_spectral_temporal_gnn_for_molecular_toxicity_prediction.models.components import (
    UncertaintyWeightedMultiTaskLoss
)


@pytest.fixture
def sample_graph():
    """Create a sample molecular graph for testing."""
    # Simple graph with 10 nodes
    num_nodes = 10
    x = torch.randn(num_nodes, 30)  # 30 node features

    # Create edges (ring structure)
    edge_list = []
    for i in range(num_nodes):
        edge_list.append([i, (i + 1) % num_nodes])
        edge_list.append([(i + 1) % num_nodes, i])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t()

    data = Data(x=x, edge_index=edge_index)
    return data


@pytest.fixture
def sample_batch(sample_graph):
    """Create a batch of sample graphs."""
    from torch_geometric.data import Batch
    graphs = [sample_graph for _ in range(4)]
    return Batch.from_data_list(graphs)


@pytest.fixture
def sample_model():
    """Create a sample model for testing."""
    model = AdaptiveSpectralTemporalGNN(
        input_dim=30,
        hidden_dim=64,
        num_layers=2,
        num_tasks=12,
        num_eigenvectors=10,
        num_spectral_filters=8,
        dropout=0.3,
        use_adaptive_filters=True,
        use_attention=True
    )
    return model


@pytest.fixture
def sample_loss():
    """Create a sample loss function."""
    return UncertaintyWeightedMultiTaskLoss(
        num_tasks=12,
        learn_weights=True,
        regularization=0.01
    )


@pytest.fixture
def sample_labels():
    """Create sample labels for multi-task learning."""
    batch_size = 4
    num_tasks = 12
    labels = torch.randint(0, 2, (batch_size, num_tasks)).float()
    masks = torch.ones(batch_size, num_tasks)
    return labels, masks


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
