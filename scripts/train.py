import importlib
from functools import partial
from typing import Callable

import hydra
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, RichProgressBar
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.tuner import Tuner
from omegaconf import DictConfig

from qm9_gnn.data_loader.qm9 import QM9DataModule
from qm9_gnn.models.lightning_model import QM9GNN


def load_layer(layer_path: str) -> Callable:
    module_name, layer_name = layer_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    layer = getattr(module, layer_name)
    return layer


@hydra.main(version_base=None, config_path="../qm9_gnn/config", config_name="config")
def main(cfg: DictConfig):
    data_module = QM9DataModule()
    logger = WandbLogger(
        project="qm9_experiment",
        log_model=False,
        offline=True,
    )
    data_module = QM9DataModule(batch_size=cfg.batch_size)

    out_channels = data_module.num_classes
    in_channels = data_module.num_features

    layer = load_layer(cfg.layer_name)
    mp_layer = partial(layer, **cfg.layer_kwargs)
    model = QM9GNN(
        in_channels=in_channels,
        hidden_size=cfg.hidden_size,
        out_channels=out_channels,
        num_layers=cfg.num_layers,
        mp_layer=mp_layer,
    )
    trainer = L.Trainer(
        strategy="ddp",
        precision=16,
        max_epochs=cfg.epochs,
        callbacks=[
            RichProgressBar(refresh_rate=10),
            EarlyStopping(monitor="valid_mae", patience=3),
        ],
        accumulate_grad_batches=cfg.accumulation,
        logger=logger,
        profiler=PyTorchProfiler(
            profiler="simple",
            output_filename="profiler_output.txt",
            log_graph=True,
        ),
    )

    tuner = Tuner(trainer)
    tuner.scale_batch_size(model, mode="power", datamodule=data_module)
    trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    main()
