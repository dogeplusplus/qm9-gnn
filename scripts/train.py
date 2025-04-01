import lightning as L
from lightning.pytorch.callbacks import RichProgressBar

from qm9_gnn.data_loader.qm9 import QM9DataModule
from qm9_gnn.models.lightning_model import QM9GNN


def main():
    data_module = QM9DataModule()
    model = QM9GNN(num_outputs=19)
    trainer = L.Trainer(
        max_epochs=10,
        callbacks=RichProgressBar(),
        accumulate_grad_batches=4,
    )
    trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    main()
