import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LayerNorm, Linear, MessagePassing
from torch_geometric.utils import scatter
from torch_geometric.data import Batch


class MolGNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        mp_layer: MessagePassing,
        residual_every: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.residual_every = residual_every

        self.layers = nn.ModuleList(
            mp_layer(
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

        residual = None
        for idx, (norm, layer) in enumerate(zip(self.norms, self.layers)):
            x = norm(x)
            x = layer(x, edge_index)
            x = F.relu(x)
            if idx % self.residual_every == 0:
                if residual is not None:
                    x = x + residual
                if idx > 2:
                    residual = x

        x = scatter(x, data.batch, dim=0, reduce="mean")
        x = self.linear(x)

        return x
