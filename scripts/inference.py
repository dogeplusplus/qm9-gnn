from pathlib import Path

import click
import pandas as pd
import torch
from lightning import Trainer
from lightning.pytorch.tuner import Tuner

from qm9_gnn.data_loader.qm9 import QM9DataModule
from qm9_gnn.models.lightning_model import QM9GNN
from qm9_gnn.utils.constants import RegressionTask


@click.command()
@click.option(
    "--output-path",
    type=click.File("wb"),
    help="Path to output predictions",
    default="outputs/predictions/outputs.csv",
)
@click.option(
    "--checkpoint-path",
    type=click.File("rb"),
    help="Path to model checkpoint",
)
def molecular_inference(checkpoint_path: Path, output_path: Path):
    model = QM9GNN.load_from_checkpoint(checkpoint_path)
    model.eval()

    data = QM9DataModule()
    trainer = Trainer()
    tuner = Tuner(trainer)
    tuner.scale_batch_size(model, mode="power", datamodule=data)

    data.setup(stage="predict")
    columns = [x.name.lower() for x in list(RegressionTask)]

    ground_truth = torch.vstack([graph.y for graph in data.predict_dataset])
    ground_truth = pd.DataFrame(ground_truth.numpy())
    ground_truth.columns = [c + "_true" for c in columns]

    predictions = trainer.predict(model, data)
    predictions = torch.vstack(predictions)
    predictions = pd.DataFrame(predictions.numpy())
    predictions.columns = [c + "_pred" for c in columns]

    combined = pd.concat([predictions, ground_truth], axis=1)
    combined.to_csv(output_path, index=False)


if __name__ == "__main__":
    molecular_inference()
