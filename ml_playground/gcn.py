from typing import Optional

import torch
from torch import nn, Tensor, device, dtype

from ml_playground.dense import Dense
from ml_playground.utils import TorchKw

class GCN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        add_self_loops: bool = True,
        bias: bool = True,
        device: Optional[device] = None,
        dtype: Optional[dtype] = None,
    ) -> None:
        kwargs: TorchKw = {"device": device, "dtype": dtype}
        super().__init__()
        self.linear = Dense(in_dim, out_dim, bias, **kwargs)
        self.add_self_loops = add_self_loops

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_weight: Optional[Tensor] = None
    ) -> Tensor:

        num_nodes = x.size(0)
        num_edges = edge_index.size(1)
        in_dim = x.size(1)

        src = edge_index[0]
        tgt = edge_index[1]

        if edge_weight is None:
            edge_weight = torch.ones(
                (num_edges,), dtype=torch.int64, device=edge_index.device
            )

        if self.add_self_loops:
            src = torch.cat((src, torch.tensor(range(num_nodes), device=src.device)))
            tgt = torch.cat((tgt, torch.tensor(range(num_nodes), device=tgt.device)))
            edge_weight = torch.cat(
                (
                    edge_weight,
                    torch.ones(
                        num_nodes, dtype=edge_weight.dtype, device=edge_weight.device
                    ),
                )
            )
            num_edges = src.size(0)

        deg = torch.zeros(
            num_nodes, device=edge_index.device, dtype=edge_weight.dtype
        ).scatter_add_(0, src, edge_weight)
        deg = 1 / torch.sqrt(deg)

        edge_weight_norm = edge_weight * deg[src] * deg[tgt]

        x_weight = x[src] * edge_weight_norm.unsqueeze(-1)

        x = torch.zeros(
            (num_nodes, in_dim), device=edge_index.device, dtype=x_weight.dtype
        ).scatter_add_(0, tgt.unsqueeze(-1).expand((num_edges, in_dim)), x_weight)

        return self.linear(x)
