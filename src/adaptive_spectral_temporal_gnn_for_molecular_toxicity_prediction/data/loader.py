"""Data loading utilities for Tox21 dataset."""

import logging
from typing import Dict, List, Tuple, Optional

import deepchem as dc
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch

logger = logging.getLogger(__name__)


class Tox21Dataset(Dataset):
    """PyTorch Dataset wrapper for Tox21 molecular data."""

    def __init__(self, graphs: List[Data], labels: np.ndarray, masks: np.ndarray):
        """Initialize dataset.

        Args:
            graphs: List of PyTorch Geometric graph data objects.
            labels: Multi-task labels of shape (num_samples, num_tasks).
            masks: Binary mask indicating valid labels (num_samples, num_tasks).
        """
        self.graphs = graphs
        self.labels = torch.FloatTensor(labels)
        self.masks = torch.FloatTensor(masks)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Tuple[Data, torch.Tensor, torch.Tensor]:
        return self.graphs[idx], self.labels[idx], self.masks[idx]


def collate_fn(batch: List[Tuple[Data, torch.Tensor, torch.Tensor]]) -> Tuple[Batch, torch.Tensor, torch.Tensor]:
    """Collate function for batching graph data.

    Args:
        batch: List of (graph, label, mask) tuples.

    Returns:
        Batched graphs, labels, and masks.
    """
    graphs, labels, masks = zip(*batch)
    batched_graphs = Batch.from_data_list(graphs)
    batched_labels = torch.stack(labels)
    batched_masks = torch.stack(masks)
    return batched_graphs, batched_labels, batched_masks


class Tox21DataLoader:
    """Data loader for Tox21 molecular toxicity dataset."""

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        split_ratio: List[float] = [0.8, 0.1, 0.1],
        featurizer: str = "GraphConv"
    ):
        """Initialize Tox21 data loader.

        Args:
            batch_size: Batch size for data loaders.
            num_workers: Number of worker processes for data loading.
            split_ratio: Train/val/test split ratios.
            featurizer: Featurization method for molecules.
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_ratio = split_ratio
        self.featurizer = featurizer
        self.num_tasks = 12  # Tox21 has 12 toxicity tasks

    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
        """Load and prepare Tox21 dataset.

        Returns:
            Train, validation, and test data loaders, and task names.
        """
        logger.info("Loading Tox21 dataset from DeepChem...")

        # Load Tox21 dataset using DeepChem
        tasks, datasets, transformers = dc.molnet.load_tox21(
            featurizer=self.featurizer,
            splitter='random',
            reload=True
        )

        train_dataset, valid_dataset, test_dataset = datasets

        logger.info(f"Dataset sizes - Train: {len(train_dataset)}, "
                   f"Val: {len(valid_dataset)}, Test: {len(test_dataset)}")
        logger.info(f"Number of tasks: {len(tasks)}")

        # Convert to PyTorch Geometric format
        train_graphs, train_labels, train_masks = self._convert_to_pyg(train_dataset)
        val_graphs, val_labels, val_masks = self._convert_to_pyg(valid_dataset)
        test_graphs, test_labels, test_masks = self._convert_to_pyg(test_dataset)

        # Create PyTorch datasets
        train_ds = Tox21Dataset(train_graphs, train_labels, train_masks)
        val_ds = Tox21Dataset(val_graphs, val_labels, val_masks)
        test_ds = Tox21Dataset(test_graphs, test_labels, test_masks)

        # Create data loaders
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available()
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available()
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available()
        )

        return train_loader, val_loader, test_loader, tasks

    def _convert_to_pyg(
        self,
        dataset: dc.data.Dataset
    ) -> Tuple[List[Data], np.ndarray, np.ndarray]:
        """Convert DeepChem dataset to PyTorch Geometric format.

        Args:
            dataset: DeepChem dataset.

        Returns:
            List of PyG Data objects, labels array, and mask array.
        """
        from rdkit import Chem

        graphs = []
        labels = []
        masks = []

        for i in range(len(dataset)):
            try:
                # Get graph features from DeepChem
                graph_feat = dataset.X[i]
                label = dataset.y[i]
                mask = dataset.w[i]

                # Convert to PyTorch Geometric Data object
                if hasattr(graph_feat, 'node_features'):
                    # GraphConv featurizer
                    x = torch.FloatTensor(graph_feat.node_features)
                    edge_index = torch.LongTensor(graph_feat.edge_index)

                    data = Data(x=x, edge_index=edge_index)
                    graphs.append(data)
                    labels.append(label)
                    masks.append(mask)
                else:
                    # Try to manually convert the molecule
                    smiles = dataset.ids[i]
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        # Create simple node features (atom types)
                        num_atoms = mol.GetNumAtoms()
                        if num_atoms > 0:
                            # Simple one-hot encoding of atom types
                            x = torch.zeros(num_atoms, 30)  # Support up to 30 atom types
                            for j, atom in enumerate(mol.GetAtoms()):
                                atomic_num = min(atom.GetAtomicNum(), 29)  # Cap at 29
                                x[j, atomic_num] = 1.0

                            # Create edge index from bonds
                            edge_list = []
                            for bond in mol.GetBonds():
                                i_atom = bond.GetBeginAtomIdx()
                                j_atom = bond.GetEndAtomIdx()
                                edge_list.append([i_atom, j_atom])
                                edge_list.append([j_atom, i_atom])  # Undirected

                            if len(edge_list) > 0:
                                edge_index = torch.LongTensor(edge_list).t()
                            else:
                                # Self-loops for isolated atoms
                                edge_index = torch.LongTensor([list(range(num_atoms)), list(range(num_atoms))])

                            data = Data(x=x, edge_index=edge_index)
                            graphs.append(data)
                            labels.append(label)
                            masks.append(mask)
            except Exception as e:
                # Skip molecules that fail to convert
                logger.debug(f"Skipping molecule {i}: {e}")
                continue

        if len(graphs) == 0:
            raise ValueError("No valid molecules found in dataset. Check featurizer settings.")

        labels_array = np.array(labels)
        masks_array = np.array(masks)

        logger.info(f"Converted {len(graphs)} molecules to PyG format")

        return graphs, labels_array, masks_array
