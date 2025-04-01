import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, LayerNorm, Linear
from torch_geometric.utils import scatter
from torch_geometric.data import Batch


class MolGNN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        self.layers = nn.ModuleList(
            GCNConv(
                self.in_channels if idx == 0 else self.hidden_channels,
                self.hidden_channels,
            )
            for idx in range(self.num_layers)
        )
        self.norms = nn.ModuleList(
            LayerNorm(self.in_channels if idx == 0 else self.hidden_channels)
            for idx in range(self.num_layers)
        )
        self.linear = Linear(self.hidden_channels, self.out_channels)

    def forward(self, data: Batch):
        x, edge_index = data.x, data.edge_index
        for norm, layer in zip(self.norms, self.layers):
            x = norm(x)
            x = layer(x, edge_index)
            x = F.relu(x)

        x = scatter(x, data.batch, dim=0, reduce="mean")
        x = self.linear(x)

        return x
