import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics

from qm9_gnn.models.gnn import MolGNN


class QM9GNN(L.LightningModule):
    def __init__(self, num_outputs: int, display_every: int = 100):
        super().__init__()
        self.model = MolGNN(11, 128, num_outputs, 24)

        self.train_metrics = torchmetrics.MetricCollection(
            {
                "rmse": torchmetrics.regression.MeanSquaredError(
                    squared=False,
                ),
                "mae": torchmetrics.regression.MeanAbsoluteError(),
            },
            prefix="train_",
        )
        self.valid_metrics = self.train_metrics.clone(prefix="valid_")
        self.display_every = display_every

    def training_step(self, batch, batch_idx):
        pred = self.model(batch)
        loss = F.mse_loss(pred, batch.y)
        self.log_dict(
            self.train_metrics(pred, batch.y), prog_bar=True, batch_size=len(batch.y),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self.model(batch)
        loss = F.mse_loss(pred, batch.y)
        self.log_dict(
            self.valid_metrics(pred, batch.y), prog_bar=True, batch_size=len(batch.y),
        )
        return loss

    def on_validation_epoch_end(self):
        self.valid_metrics.reset()

    def on_train_epoch_end(self):
        self.train_metrics.reset()

    def predict_step(self, batch, batch_idx):
        pred = self.model(batch)
        return pred

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters())
        return self.optimizer
