"""Tests for data loading and preprocessing."""

import pytest
import numpy as np
import torch
from torch_geometric.data import Data

from adaptive_spectral_temporal_gnn_for_molecular_toxicity_prediction.data.preprocessing import (
    MolecularFeaturizer,
    compute_graph_edit_distance,
    compute_curriculum_difficulty,
    select_centroids
)


def test_compute_laplacian_eigenvectors():
    """Test Laplacian eigenvector computation."""
    # Create simple graph
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    num_nodes = 3
    k = 2

    eigenvectors = MolecularFeaturizer.compute_laplacian_eigenvectors(
        edge_index, num_nodes, k
    )

    assert eigenvectors.shape == (num_nodes, k)
    assert not torch.isnan(eigenvectors).any()


def test_add_laplacian_features(sample_graph):
    """Test adding Laplacian features to graph."""
    data_with_features = MolecularFeaturizer.add_laplacian_features(
        sample_graph, num_eigenvectors=10
    )

    assert hasattr(data_with_features, 'laplacian_eigenvectors')
    assert data_with_features.laplacian_eigenvectors.shape[0] == sample_graph.num_nodes
    assert data_with_features.laplacian_eigenvectors.shape[1] == 10


def test_compute_graph_edit_distance(sample_graph):
    """Test graph edit distance computation."""
    # Create another graph
    graph2 = Data(
        x=torch.randn(8, 30),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    )

    distance = compute_graph_edit_distance(sample_graph, graph2)

    assert isinstance(distance, float)
    assert distance >= 0


def test_select_centroids(sample_graph):
    """Test centroid selection."""
    graphs = [sample_graph for _ in range(20)]
    num_centroids = 5

    centroids = select_centroids(graphs, num_centroids)

    assert len(centroids) == num_centroids
    assert all(isinstance(c, Data) for c in centroids)


def test_compute_curriculum_difficulty(sample_graph):
    """Test curriculum difficulty computation."""
    graphs = [sample_graph for _ in range(10)]
    centroids = [sample_graph for _ in range(3)]

    difficulties = compute_curriculum_difficulty(graphs, centroids)

    assert len(difficulties) == len(graphs)
    assert all(d >= 0 for d in difficulties)
