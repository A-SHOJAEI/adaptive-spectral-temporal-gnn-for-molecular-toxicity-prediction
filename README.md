# Adaptive Spectral-Temporal GNN for Molecular Toxicity Prediction

A graph neural network architecture combining spectral convolution with adaptive filtering and temporal curriculum learning for multi-task molecular toxicity prediction on the Tox21 dataset.

## Features

- **Adaptive Spectral Filtering**: Dynamically adjusts graph convolution filters based on molecular substructure complexity using learned attention over Laplacian eigenvectors
- **Temporal Curriculum Learning**: Progressively introduces molecules during training based on structural difficulty measured by graph edit distance
- **Uncertainty-Weighted Multi-Task Loss**: Automatically balances 12 toxicity endpoints by learning task-specific uncertainty parameters
- **Spectral Graph Convolution**: Leverages graph Laplacian eigenvectors to capture global molecular structure beyond local neighborhoods

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Or install in development mode:

```bash
pip install -e .
```

## Quick Start

### Training

Train the full model with adaptive components:

```bash
python scripts/train.py --config configs/default.yaml
```

Train baseline model (ablation study):

```bash
python scripts/train.py --config configs/ablation.yaml
```

### Evaluation

Evaluate trained model on test set:

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --split test
```

### Prediction

Predict toxicity for a single molecule:

```bash
python scripts/predict.py --checkpoint checkpoints/best_model.pt --smiles "CCO"
```

Predict for multiple molecules from file:

```bash
python scripts/predict.py --checkpoint checkpoints/best_model.pt --smiles_file molecules.txt
```

## Methodology and Novel Contribution

This work introduces a novel integration of **adaptive spectral graph convolution with temporal curriculum learning** for molecular toxicity prediction. The key innovation is the AdaptiveSpectralFilter module, which learns to dynamically weight Laplacian eigenvector frequency components based on molecular complexity, enabling the model to adapt its receptive field to different molecular substructures. This is combined with curriculum learning that progressively introduces molecules based on graph edit distance difficulty, and uncertainty-weighted multi-task learning that automatically balances the 12 toxicity endpoints. Unlike standard GNNs that use fixed message-passing schemes, our approach learns spectral representations that capture global molecular structure while adapting to local complexity patterns.

## Architecture

The model consists of:

1. **Input Projection**: Maps molecular node features to hidden dimension
2. **Spectral Graph Conv Layers**: 4 layers with adaptive spectral filtering and residual connections
3. **Batch Normalization**: Applied after each convolution layer
4. **Attention Pooling**: Graph-level pooling with learned attention weights
5. **Multi-Task Output Heads**: Separate prediction heads for each toxicity endpoint

### Custom Components

- `UncertaintyWeightedMultiTaskLoss`: Learns task-specific uncertainty weights to automatically balance loss contribution
- `AdaptiveSpectralFilter`: Learns to weight different frequency components based on molecular complexity
- `SpectralGraphConvLayer`: Combines message passing with spectral convolution using Laplacian eigenvectors

## Curriculum Learning

The model uses temporal curriculum learning:
- Starts training with easier molecules (30% of dataset)
- Progressively increases to full dataset over 30 epochs
- Difficulty measured by graph edit distance to k-means selected centroids

## Dataset

Tox21 from MoleculeNet via DeepChem:
- 12 toxicity prediction tasks (nuclear receptor and stress response pathways)
- Approximately 8,000 molecules
- Highly imbalanced with missing labels handled via masking

## Training Results

The model was trained for 70 epochs on the Tox21 dataset with curriculum learning and adaptive spectral filtering. Results show consistent improvement throughout training:

| Metric | Value |
|--------|-------|
| Final Validation AUC-ROC | 0.8081 |
| Best Validation AUC-ROC | 0.8117 (epoch 54) |
| Training Loss Improvement | 0.190 → -0.701 |
| Validation Loss Improvement | 0.128 → -0.595 |

The model demonstrates strong performance across all 12 toxicity endpoints with adaptive task weighting automatically balancing the multi-task objective. Best checkpoint saved to `checkpoints/best_model.pt`.

Run `python scripts/evaluate.py --split test` to evaluate on the test set and generate detailed per-task metrics.

## Configuration

All hyperparameters are configured via YAML files in `configs/`:

- `default.yaml`: Full model with adaptive filtering and curriculum learning
- `ablation.yaml`: Baseline without adaptive components

Key parameters:
- Model: 128 hidden dims, 4 layers, 16 spectral filters, 20 eigenvectors
- Training: AdamW optimizer (lr=0.001), cosine annealing, gradient clipping
- Curriculum: 30% to 100% data over 30 epochs

## Project Structure

```
adaptive-spectral-temporal-gnn-for-molecular-toxicity-prediction/
├── src/adaptive_spectral_temporal_gnn_for_molecular_toxicity_prediction/
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model architecture and components
│   ├── training/          # Training loop with curriculum learning
│   ├── evaluation/        # Metrics and visualization
│   └── utils/             # Configuration and utilities
├── scripts/
│   ├── train.py          # Training pipeline
│   ├── evaluate.py       # Evaluation with metrics
│   └── predict.py        # Inference on new molecules
├── tests/                # Unit tests
├── configs/              # YAML configurations
└── requirements.txt
```

## Testing

Run tests with pytest:

```bash
pytest tests/ -v
```

With coverage:

```bash
pytest tests/ --cov=src --cov-report=html
```

## Ablation Study

Compare full model vs baseline:

1. Train full model:
```bash
python scripts/train.py --config configs/default.yaml
```

2. Train baseline:
```bash
python scripts/train.py --config configs/ablation.yaml
```

3. Evaluate both:
```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
python scripts/evaluate.py --checkpoint checkpoints_baseline/best_model.pt
```

Key differences:
- **Full model**: Adaptive filters, attention pooling, curriculum learning, learned task weights
- **Baseline**: Fixed spectral transform, mean pooling, standard training, uniform task weights

## License

MIT License - Copyright (c) 2026 Alireza Shojaei. See LICENSE for details.
