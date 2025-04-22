import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics

from torch_geometric.nn import MessagePassing
from qm9_gnn.models.gnn import MolGNN


class QM9GNN(L.LightningModule):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        out_channels: int,
        num_layers: int,
        mp_layer: MessagePassing,
        display_every: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = MolGNN(in_channels, hidden_size, out_channels, num_layers, mp_layer)

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
            self.train_metrics(pred, batch.y),
            prog_bar=True,
            batch_size=len(batch.y),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self.model(batch)
        loss = F.mse_loss(pred, batch.y)
        self.log_dict(
            self.valid_metrics(pred, batch.y),
            prog_bar=True,
            batch_size=len(batch.y),
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
        optimizer = torch.optim.AdamW(self.model.parameters())
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }
