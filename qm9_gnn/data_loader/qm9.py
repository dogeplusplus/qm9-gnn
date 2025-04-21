import os
from pathlib import Path
from typing import Tuple

import lightning as L
import numpy as np
import torch
from rdkit.Chem import rdFingerprintGenerator
from sklearn.cluster import KMeans
from torch.utils.data import Dataset
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader


def get_train_valid_indicies(
    dataset: QM9,
    test_ratio: float,
    seed: int,
    k: int = 10,
) -> Tuple[Dataset, Dataset]:
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
    )

    fingerprints = np.array([mfpgen.GetFingerprint(mol).GetOnBits() for mol in dataset])

    kmeans = KMeans(n_clusters=k, random_state=seed)
    kmeans.fit(fingerprints)
    labels = kmeans.labels_

    test_indices = []
    test_size = int(len(dataset) * test_ratio)

    while len(test_indices) < test_size:
        cluster_indices = np.where(labels == np.random.randint(k))[0]
        if len(cluster_indices) > 0:
            test_indices.append(np.random.choice(cluster_indices))

    test_indices = np.unique(test_indices)

    train_indices = np.setdiff1d(
        np.arange(len(dataset)),
        test_indices,
    )

    train_dataset = dataset[train_indices]
    test_dataset = dataset[test_indices]

    return train_dataset, test_dataset


class QM9DataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: Path = Path("data"),
        batch_size: int = 32,
        test_ratio: float = 0.2,
        seed: int = 42,
        workers=os.cpu_count() // 2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.dataset = QM9(root=data_dir)
        self.num_classes = self.dataset.num_classes
        self.num_features = self.dataset.num_features
        self.batch_size = batch_size
        self.test_ratio = test_ratio
        self.seed = seed
        self.workers = workers

    def setup(self, stage: str):
        if stage == "fit":
            test_size = int(self.test_ratio * len(self.dataset))
            train_size = len(self.dataset) - test_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.dataset,
                [train_size, test_size],
                generator=torch.Generator().manual_seed(self.seed),
            )
        elif stage == "test":
            self.test_dataset = self.dataset
        elif stage == "predict":
            self.predict_dataset = self.dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size | self.hparams.batch_size,
            num_workers=self.workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size | self.hparams.batch_size,
            shuffle=False,
            num_workers=self.workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size | self.hparams.batch_size,
            shuffle=False,
            num_workers=self.workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size | self.hparams.batch_size,
            shuffle=False,
            num_workers=self.workers,
        )
