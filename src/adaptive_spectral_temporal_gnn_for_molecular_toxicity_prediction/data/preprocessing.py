"""Molecular preprocessing and featurization utilities."""

import logging
from typing import List, Tuple

import networkx as nx
import numpy as np
import torch
from rdkit import Chem
from scipy.sparse.linalg import eigsh
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

logger = logging.getLogger(__name__)


class MolecularFeaturizer:
    """Featurizer for molecular graphs."""

    @staticmethod
    def compute_laplacian_eigenvectors(
        edge_index: torch.Tensor,
        num_nodes: int,
        k: int = 20
    ) -> torch.Tensor:
        """Compute Laplacian eigenvectors for spectral convolution.

        Args:
            edge_index: Edge indices of graph.
            num_nodes: Number of nodes in graph.
            k: Number of eigenvectors to compute.

        Returns:
            Eigenvectors of shape (num_nodes, k).
        """
        # Build adjacency matrix
        adj = torch.zeros((num_nodes, num_nodes))
        adj[edge_index[0], edge_index[1]] = 1

        # Compute degree matrix
        degree = adj.sum(dim=1)
        deg_inv_sqrt = torch.pow(degree, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0

        # Compute normalized Laplacian: I - D^(-1/2) A D^(-1/2)
        deg_matrix = torch.diag(deg_inv_sqrt)
        normalized_adj = deg_matrix @ adj @ deg_matrix
        laplacian = torch.eye(num_nodes) - normalized_adj

        # Compute eigenvectors
        try:
            k_actual = min(k, num_nodes - 1)
            eigenvalues, eigenvectors = eigsh(
                laplacian.numpy(),
                k=k_actual,
                which='SM'
            )

            # Pad if necessary
            if k_actual < k:
                padding = torch.zeros((num_nodes, k - k_actual))
                eigenvectors = torch.FloatTensor(eigenvectors)
                eigenvectors = torch.cat([eigenvectors, padding], dim=1)
            else:
                eigenvectors = torch.FloatTensor(eigenvectors)

        except Exception as e:
            logger.warning(f"Failed to compute eigenvectors: {e}. Using random.")
            eigenvectors = torch.randn(num_nodes, k)

        return eigenvectors

    @staticmethod
    def add_laplacian_features(data: Data, num_eigenvectors: int = 20) -> Data:
        """Add Laplacian eigenvector features to graph data.

        Args:
            data: PyTorch Geometric Data object.
            num_eigenvectors: Number of eigenvectors to compute.

        Returns:
            Data object with added eigenvector features.
        """
        eigenvectors = MolecularFeaturizer.compute_laplacian_eigenvectors(
            data.edge_index,
            data.num_nodes,
            k=num_eigenvectors
        )
        data.laplacian_eigenvectors = eigenvectors
        return data


def compute_graph_edit_distance(graph1: Data, graph2: Data) -> float:
    """Compute approximate graph edit distance between two molecular graphs.

    Args:
        graph1: First molecular graph.
        graph2: Second molecular graph.

    Returns:
        Approximate graph edit distance.
    """
    # Convert to NetworkX for GED computation
    g1 = to_networkx(graph1, to_undirected=True)
    g2 = to_networkx(graph2, to_undirected=True)

    # Use simple heuristic: difference in nodes + difference in edges
    node_diff = abs(g1.number_of_nodes() - g2.number_of_nodes())
    edge_diff = abs(g1.number_of_edges() - g2.number_of_edges())

    # Normalized by average graph size
    avg_size = (g1.number_of_nodes() + g2.number_of_nodes()) / 2
    if avg_size == 0:
        return 0.0

    ged = (node_diff + edge_diff) / avg_size
    return ged


def compute_curriculum_difficulty(
    graphs: List[Data],
    centroids: List[Data],
    metric: str = "graph_edit_distance"
) -> np.ndarray:
    """Compute difficulty scores for curriculum learning.

    Args:
        graphs: List of molecular graphs.
        centroids: Centroid graphs for distance computation.
        metric: Difficulty metric to use.

    Returns:
        Array of difficulty scores.
    """
    difficulties = []

    for graph in graphs:
        # Compute minimum distance to any centroid
        min_distance = float('inf')

        for centroid in centroids:
            if metric == "graph_edit_distance":
                distance = compute_graph_edit_distance(graph, centroid)
            else:
                # Default: use number of nodes as proxy
                distance = abs(graph.num_nodes - centroid.num_nodes)

            min_distance = min(min_distance, distance)

        difficulties.append(min_distance)

    return np.array(difficulties)


def select_centroids(graphs: List[Data], num_centroids: int = 10) -> List[Data]:
    """Select centroid graphs using k-means++ style selection.

    Args:
        graphs: List of molecular graphs.
        num_centroids: Number of centroids to select.

    Returns:
        List of centroid graphs.
    """
    if len(graphs) <= num_centroids:
        return graphs

    centroids = []

    # Select first centroid randomly
    idx = np.random.randint(0, len(graphs))
    centroids.append(graphs[idx])

    # Select remaining centroids
    for _ in range(num_centroids - 1):
        # Compute distances to nearest centroid
        distances = []
        for graph in graphs:
            min_dist = min(
                compute_graph_edit_distance(graph, centroid)
                for centroid in centroids
            )
            distances.append(min_dist)

        # Select graph with maximum distance
        max_idx = np.argmax(distances)
        centroids.append(graphs[max_idx])

    return centroids
