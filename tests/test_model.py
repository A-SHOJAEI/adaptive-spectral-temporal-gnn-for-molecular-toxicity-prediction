"""Tests for model components."""

import pytest
import torch

from adaptive_spectral_temporal_gnn_for_molecular_toxicity_prediction.models.components import (
    UncertaintyWeightedMultiTaskLoss,
    AdaptiveSpectralFilter,
    SpectralGraphConvLayer
)


def test_uncertainty_weighted_loss_forward(sample_labels):
    """Test forward pass of uncertainty-weighted loss."""
    labels, masks = sample_labels
    batch_size = labels.shape[0]
    num_tasks = labels.shape[1]

    loss_fn = UncertaintyWeightedMultiTaskLoss(num_tasks, learn_weights=True)

    predictions = torch.randn(batch_size, num_tasks)
    loss = loss_fn(predictions, labels, masks)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar
    assert not torch.isnan(loss)


def test_uncertainty_weighted_loss_no_learning(sample_labels):
    """Test loss without learned weights."""
    labels, masks = sample_labels
    batch_size = labels.shape[0]
    num_tasks = labels.shape[1]

    loss_fn = UncertaintyWeightedMultiTaskLoss(num_tasks, learn_weights=False)

    predictions = torch.randn(batch_size, num_tasks)
    loss = loss_fn(predictions, labels, masks)

    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)


def test_adaptive_spectral_filter():
    """Test adaptive spectral filter."""
    num_eigenvectors = 10
    hidden_dim = 64
    num_filters = 8
    num_nodes = 15

    filter_module = AdaptiveSpectralFilter(num_eigenvectors, hidden_dim, num_filters)

    node_features = torch.randn(num_nodes, hidden_dim)
    eigenvectors = torch.randn(num_nodes, num_eigenvectors)

    output = filter_module(node_features, eigenvectors)

    assert output.shape == (num_nodes, num_filters)
    assert not torch.isnan(output).any()


def test_spectral_graph_conv_layer():
    """Test spectral graph convolution layer."""
    in_channels = 64
    out_channels = 64
    num_nodes = 15
    num_eigenvectors = 10

    layer = SpectralGraphConvLayer(
        in_channels,
        out_channels,
        num_eigenvectors=num_eigenvectors,
        use_adaptive=True
    )

    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    eigenvectors = torch.randn(num_nodes, num_eigenvectors)

    output = layer(x, edge_index, eigenvectors)

    assert output.shape == (num_nodes, out_channels)
    assert not torch.isnan(output).any()


def test_model_forward(sample_model, sample_batch):
    """Test model forward pass."""
    predictions, embeddings = sample_model(sample_batch, return_embeddings=True)

    assert predictions.shape == (sample_batch.num_graphs, 12)
    assert embeddings.shape[0] == sample_batch.num_graphs
    assert not torch.isnan(predictions).any()


def test_model_num_params(sample_model):
    """Test parameter counting."""
    num_params = sample_model.get_num_params()

    assert num_params > 0
    assert isinstance(num_params, int)
