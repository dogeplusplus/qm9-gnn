from pathlib import Path
import os
import torch
import lightning as L
from torch.utils.data import random_split
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader


class QM9DataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: Path = Path("data"),
        batch_size: int = 32,
        test_size: float = 0.2,
        seed: int = 42,
        workers=os.cpu_count() // 2,
    ):
        super().__init__()
        self.dataset = QM9(root=data_dir)
        self.batch_size = batch_size
        self.test_size = test_size
        self.seed = seed
        self.workers = workers

    def setup(self, stage: str):
        dataset_size = len(self.dataset)
        test_size = int(dataset_size * self.test_size)

        if stage == "fit":
            self.train_dataset, self.val_dataset = random_split(
                self.dataset,
                [dataset_size - test_size, test_size],
                generator=torch.Generator().manual_seed(self.seed),
            )
        elif stage == "test":
            self.test_dataset = self.dataset
        elif stage == "predict":
            self.predict_dataset = self.dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
        )
