# QM9-GNN: Molecular Property Prediction with Graph Neural Networks

A Graph Neural Network implementation for predicting molecular properties on the QM9 dataset. This project uses PyTorch Geometric and PyTorch Lightning to build and train GNN models for quantum chemistry tasks.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/qm9-gnn.git
cd qm9-gnn
```

2. Install dependencies:
```bash
uv venv
source .venv/bin/activate
uv sync
```

## Project Structure

- hello.py: Main script containing the model, data processing, and training pipeline
- data: Directory containing QM9 dataset
- lightning_logs: Training logs and checkpoints

## Regression Tasks

The model supports prediction of various molecular properties from the QM9 dataset, including:
- Dipole moment
- HOMO-LUMO energy gap
- Atomization energies
- And other quantum chemical properties
