import torch
import click
from pathlib import Path
from lightning import Trainer
from lightning.pytorch.tuner import Tuner

from qm9_gnn.data_loader.qm9 import QM9DataModule
from qm9_gnn.models.lightning_model import QM9GNN


@click.command()
@click.option(
    "--checkpoint-path",
    type=click.File("rb"),
    help="Path to model checkpoint",
    default="/home/albert/github/qm9-gnn/qm9_experiment/08qehxy3/checkpoints/epoch=0-step=52.ckpt",
)
@click.option(
    "--output-path",
    type=click.File("wb"),
    help="Path to output predictions",
)
def molecular_inference(checkpoint_path: Path, output_path: Path):
    model = QM9GNN.load_from_checkpoint(checkpoint_path)
    model.eval()

    data = QM9DataModule()
    trainer = Trainer()
    tuner = Tuner(trainer)
    tuner.scale_batch_size(model, mode="power", datamodule=data)

    predictions = trainer.predict(model, data)
    predictions = torch.vstack(predictions)
    torch.save(predictions, output_path)


if __name__ == "__main__":
    molecular_inference()
