import lightning as L
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import EarlyStopping
from qm9_gnn.data_loader.qm9 import QM9DataModule
from qm9_gnn.models.lightning_model import QM9GNN


def main():
    data_module = QM9DataModule()
    model = QM9GNN(num_outputs=19)
    logger = MLFlowLogger(
        experiment_name="qm9_experiment",
        tracking_uri="data",
        save_dir="data",
        log_model=True,
    )
    data_module = QM9DataModule()
    model = QM9GNN(num_outputs=19)
    trainer = L.Trainer(
        max_epochs=10,
        callbacks=[
            RichProgressBar(refresh_rate=10),
            EarlyStopping(monitor='val_loss', patience=3),
        ],
        accumulate_grad_batches=4,
        logger=logger,
    )
    trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    main()
