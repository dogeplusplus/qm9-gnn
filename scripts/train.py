import lightning as L
from functools import partial
from torch_geometric.nn import ChebConv
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import EarlyStopping
from qm9_gnn.data_loader.qm9 import QM9DataModule
from qm9_gnn.models.lightning_model import QM9GNN


def main():
    data_module = QM9DataModule()
    logger = MLFlowLogger(
        experiment_name="qm9_experiment",
        tracking_uri="data/mlruns",
        save_dir="data/mlruns",
        log_model=True,
    )
    data_module = QM9DataModule(batch_size=512)

    out_channels = data_module.num_classes
    in_channels = data_module.num_features
    hidden_size = 64
    num_layers = 24
    mp_layer = partial(ChebConv, K=3)
    model = QM9GNN(
        in_channels=in_channels,
        hidden_size=hidden_size,
        out_channels=out_channels,
        num_layers=num_layers,
        mp_layer=mp_layer,
    )
    trainer = L.Trainer(
        max_epochs=10,
        callbacks=[
            RichProgressBar(refresh_rate=10),
            EarlyStopping(monitor="valid_mae", patience=3),
        ],
        accumulate_grad_batches=4,
        logger=logger,
    )
    trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    main()
