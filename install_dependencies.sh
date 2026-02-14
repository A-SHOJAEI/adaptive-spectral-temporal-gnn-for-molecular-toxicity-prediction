#!/bin/bash

# Installation script for Adaptive Spectral-Temporal GNN project
# This script installs all required dependencies

set -e  # Exit on error

echo "=========================================="
echo "Installing Dependencies"
echo "=========================================="

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "ERROR: pip3 is not installed."
    echo "Please install pip3 first:"
    echo "  Ubuntu/Debian: sudo apt-get install python3-pip"
    echo "  Or download: curl https://bootstrap.pypa.io/get-pip.py | python3"
    exit 1
fi

echo "✓ pip3 found"

# Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip

# Install PyTorch (adjust for your CUDA version if needed)
echo "Installing PyTorch..."
python3 -m pip install torch>=2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric dependencies
echo "Installing PyTorch Geometric..."
python3 -m pip install torch-geometric>=2.3.0
python3 -m pip install pyg_lib torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# Install other requirements
echo "Installing other dependencies..."
python3 -m pip install \
    dgl>=1.1.0 \
    networkx>=3.0 \
    deepchem>=2.7.0 \
    numpy>=1.24.0 \
    scipy>=1.10.0 \
    scikit-learn>=1.2.0 \
    pandas>=2.0.0 \
    pyyaml>=6.0 \
    mlflow>=2.8.0 \
    matplotlib>=3.7.0 \
    seaborn>=0.12.0 \
    tqdm>=4.65.0 \
    pytest>=7.3.0 \
    rdkit>=2023.3.1

echo ""
echo "=========================================="
echo "Verifying Installation"
echo "=========================================="

# Verify critical packages
python3 << 'EOF'
import sys
packages = [
    'torch',
    'torch_geometric',
    'dgl',
    'networkx',
    'deepchem',
    'numpy',
    'scipy',
    'sklearn',
    'pandas',
    'yaml',
    'mlflow',
    'matplotlib',
    'seaborn',
    'tqdm',
    'pytest',
    'rdkit'
]

print("\nPackage Status:")
all_installed = True
for pkg in packages:
    try:
        __import__(pkg)
        print(f"  ✓ {pkg}")
    except ImportError:
        print(f"  ✗ {pkg} - FAILED")
        all_installed = False

if all_installed:
    print("\n✓ All packages installed successfully!")
    sys.exit(0)
else:
    print("\n✗ Some packages failed to install")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Installation Complete!"
    echo "=========================================="
    echo ""
    echo "You can now run:"
    echo "  python3 scripts/train.py"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "Installation Failed"
    echo "=========================================="
    echo "Please check the error messages above"
    exit 1
fi
