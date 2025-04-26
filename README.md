# QM9-GNN: Molecular Property Prediction with Graph Neural Networks

A concise, modern PyTorch Geometric + Lightning implementation for molecular property prediction on the QM9 dataset.

## Features
- Predicts multiple quantum chemical properties (e.g., dipole moment, HOMO-LUMO gap, atomization energies)
- Modular GNN architecture (customizable message passing layers)
- Fast training with PyTorch Lightning
- Easy experiment tracking (WandB, rich logging)

## Installation
```zsh
git clone https://github.com/yourusername/qm9-gnn.git
cd qm9-gnn
uv venv
source .venv/bin/activate
uv sync
```

## Usage
### Training
```zsh
python scripts/train.py
```
- Configurable via `qm9_gnn/config/config.yaml`

### Inference
```zsh
python scripts/inference.py --checkpoint-path outputs/qm9_experiment/<run_id>/checkpoints/<ckpt_file> --output-path outputs/predictions/outputs.csv
```

## Project Structure
- `qm9_gnn/` – Core package (models, data loaders, utils)
- `scripts/` – Training and inference scripts
- `data/` – QM9 dataset (raw and processed)
- `outputs/` – Logs, checkpoints, predictions

## Requirements
- Python 3.12+
- See `pyproject.toml` for dependencies

## References
- [QM9 Dataset](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [PyTorch Lightning](https://lightning.ai/)
