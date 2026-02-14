#!/usr/bin/env python
"""Prediction script for Adaptive Spectral-Temporal GNN."""

import sys
import argparse
import json
import logging
from pathlib import Path

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

from adaptive_spectral_temporal_gnn_for_molecular_toxicity_prediction.models.model import AdaptiveSpectralTemporalGNN
from adaptive_spectral_temporal_gnn_for_molecular_toxicity_prediction.data.preprocessing import MolecularFeaturizer
from adaptive_spectral_temporal_gnn_for_molecular_toxicity_prediction.utils.config import load_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Make predictions with Adaptive Spectral-Temporal GNN')
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
        '--smiles',
        type=str,
        help='SMILES string of molecule to predict'
    )
    parser.add_argument(
        '--smiles_file',
        type=str,
        help='File containing SMILES strings (one per line)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/predictions.json',
        help='Output file for predictions'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU device ID'
    )
    return parser.parse_args()


def smiles_to_graph(smiles: str, featurizer_type: str = 'deepchem') -> torch.Tensor:
    """Convert SMILES string to graph representation using proper molecular featurization.

    Args:
        smiles: SMILES string.
        featurizer_type: Type of featurizer to use ('deepchem' or 'rdkit').

    Returns:
        Graph data object or None if invalid.
    """
    try:
        from torch_geometric.data import Data

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Use DeepChem-style featurization to match training
        if featurizer_type == 'deepchem':
            try:
                from deepchem.feat import MolGraphConvFeaturizer
                dc_featurizer = MolGraphConvFeaturizer(use_edges=False)
                feat = dc_featurizer.featurize([smiles])[0]

                node_features = torch.FloatTensor(feat.node_features)
                edge_index = torch.LongTensor(feat.edge_index)

                data = Data(x=node_features, edge_index=edge_index)
                return data
            except ImportError:
                logger.warning("DeepChem not available, falling back to RDKit featurization")
                featurizer_type = 'rdkit'

        if featurizer_type == 'rdkit':
            # RDKit-based atom featurization
            num_atoms = mol.GetNumAtoms()
            node_features_list = []

            for atom in mol.GetAtoms():
                # Atom features: atomic number, degree, formal charge, hybridization, aromaticity
                atom_features = [
                    atom.GetAtomicNum() / 100.0,  # Normalized atomic number
                    atom.GetDegree() / 6.0,  # Normalized degree
                    (atom.GetFormalCharge() + 5) / 10.0,  # Normalized formal charge
                    atom.GetTotalNumHs() / 4.0,  # Normalized num hydrogens
                    float(atom.GetIsAromatic()),
                    float(atom.IsInRing()),
                ]

                # One-hot encoding for hybridization
                hybridization = [0.0] * 5
                hybrid_type = str(atom.GetHybridization())
                if 'SP' in hybrid_type:
                    if 'SP3' in hybrid_type:
                        hybridization[3] = 1.0
                    elif 'SP2' in hybrid_type:
                        hybridization[2] = 1.0
                    else:
                        hybridization[1] = 1.0
                else:
                    hybridization[0] = 1.0

                atom_features.extend(hybridization)

                # Pad to 30 features to match DeepChem default
                while len(atom_features) < 30:
                    atom_features.append(0.0)

                node_features_list.append(atom_features[:30])

            node_features = torch.FloatTensor(node_features_list)

            # Edge indices
            edges = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edges.append([i, j])
                edges.append([j, i])

            if len(edges) == 0:
                # Single atom molecule
                edge_index = torch.zeros((2, 0), dtype=torch.long)
            else:
                edge_index = torch.tensor(edges, dtype=torch.long).t()

            data = Data(x=node_features, edge_index=edge_index)
            return data

    except Exception as e:
        logger.error(f"Error converting SMILES to graph: {e}")
        return None


def predict_molecule(model, graph, device, config, task_names):
    """Make prediction for a single molecule.

    Args:
        model: Trained model.
        graph: Graph data.
        device: Device to use.
        config: Configuration dict.
        task_names: List of task names.

    Returns:
        Dictionary of predictions.
    """
    model.eval()

    with torch.no_grad():
        # Move to device
        graph = graph.to(device)

        # Add batch dimension
        from torch_geometric.data import Batch
        batch = Batch.from_data_list([graph])

        # Add Laplacian eigenvectors
        if not hasattr(batch, 'laplacian_eigenvectors'):
            num_eigenvectors = config['model'].get('num_eigenvectors', 20)
            eigvecs = MolecularFeaturizer.compute_laplacian_eigenvectors(
                batch.edge_index,
                batch.num_nodes,
                k=num_eigenvectors
            )
            batch.laplacian_eigenvectors = eigvecs.to(device)

        # Forward pass
        predictions, _ = model(batch)
        predictions = torch.sigmoid(predictions).cpu().numpy()[0]

    # Format results
    results = {}
    for task_name, pred in zip(task_names, predictions):
        results[task_name] = {
            'probability': float(pred),
            'prediction': 'toxic' if pred > 0.5 else 'non-toxic',
            'confidence': float(max(pred, 1 - pred))
        }

    return results


def main():
    """Main prediction function."""
    try:
        # Parse arguments
        args = parse_args()

        # Check inputs
        if not args.smiles and not args.smiles_file:
            logger.error("Please provide either --smiles or --smiles_file")
            sys.exit(1)

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

        # Load task names
        task_names_path = Path('results/task_names.json')
        if task_names_path.exists():
            with open(task_names_path, 'r') as f:
                task_names = json.load(f)
        else:
            # Default Tox21 task names
            task_names = [
                'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
                'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
                'SR-HSE', 'SR-MMP', 'SR-p53'
            ]

        # Get input dimension from checkpoint config
        if 'config' in checkpoint:
            input_dim = 30  # Default from DeepChem GraphConv featurizer
        else:
            input_dim = 30

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

        # Collect SMILES strings
        smiles_list = []
        if args.smiles:
            smiles_list.append(args.smiles)
        elif args.smiles_file:
            with open(args.smiles_file, 'r') as f:
                smiles_list = [line.strip() for line in f if line.strip()]

        logger.info(f"Processing {len(smiles_list)} molecule(s)...")

        # Make predictions
        all_predictions = []
        for smiles in smiles_list:
            logger.info(f"\nPredicting for SMILES: {smiles}")

            # Convert to graph using proper featurization
            graph = smiles_to_graph(smiles, featurizer_type=config['data'].get('featurizer', 'deepchem'))

            if graph is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                all_predictions.append({
                    'smiles': smiles,
                    'error': 'Invalid SMILES string'
                })
                continue

            # Predict
            predictions = predict_molecule(model, graph, device, config, task_names)

            result = {
                'smiles': smiles,
                'predictions': predictions
            }
            all_predictions.append(result)

            # Print predictions
            logger.info("Predictions:")
            for task_name, pred_info in predictions.items():
                logger.info(f"  {task_name}: {pred_info['prediction']} "
                          f"(probability: {pred_info['probability']:.4f}, "
                          f"confidence: {pred_info['confidence']:.4f})")

        # Save predictions
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(all_predictions, f, indent=2)

        logger.info(f"\nSaved predictions to {output_path}")
        logger.info("Prediction completed successfully!")

    except Exception as e:
        logger.error(f"Prediction failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
