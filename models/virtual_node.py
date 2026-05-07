import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool


def _make_vn_mlp(hidden_dim: int) -> nn.Sequential:
    """Two-layer MLP with BatchNorm used to update the virtual node state."""
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
    )


class GCN_VN(nn.Module):
    """GCN with a Virtual Node inserted between every message-passing layer."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int):
        """
        Args:
            in_dim:     input feature size from TreeDataset.get_dims()
            hidden_dim: width of every hidden layer and the VN state
            out_dim:    number of output classes
            num_layers: number of GCNConv layers (same count as baseline GCN)
        """
        super(GCN_VN, self).__init__()
        self.hidden_dim = hidden_dim

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.vn_mlps = nn.ModuleList()

        # First conv: in_dim → hidden_dim
        self.convs.append(GCNConv(in_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.vn_mlps.append(_make_vn_mlp(hidden_dim))

        # Remaining convs: hidden_dim → hidden_dim
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.vn_mlps.append(_make_vn_mlp(hidden_dim))

        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, root_mask, batch):
        """
        Args:
            x:          node features  [total_nodes, in_dim]
            edge_index: COO edge list  [2, total_edges]
            root_mask:  bool mask selecting one root per graph [total_nodes]
            batch:      graph assignment vector [total_nodes]
        Returns:
            logits:     [batch_size, out_dim]
        """
        num_graphs = int(batch.max()) + 1
        vn_emb = torch.zeros(num_graphs, self.hidden_dim, device=x.device)

        for i, (conv, bn, vn_mlp) in enumerate(zip(self.convs, self.bns, self.vn_mlps)):
            # Broadcast: add VN state to every node in its graph.
            # Skipped at layer 0
            if i > 0:
                x = x + vn_emb[batch]

            # Local message passing
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)

            # Update VN: pool current node states + residual from previous VN
            vn_emb = vn_mlp(global_mean_pool(x, batch) + vn_emb)

        return self.classifier(x[root_mask])


class GIN_VN(nn.Module):
    """GIN with a Virtual Node inserted between every message-passing layer.

    Identical VN mechanism to GCN_VN; only the local aggregation step
    changes from GCNConv to GINConv (backed by a 2-layer MLP).
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int):
        """
        Args:
            in_dim:     input feature size
            hidden_dim: width of every hidden layer and the VN state
            out_dim:    number of output classes
            num_layers: number of GINConv layers (same count as baseline GIN)
        """
        super(GIN_VN, self).__init__()
        self.hidden_dim = hidden_dim

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.vn_mlps = nn.ModuleList()

        # Input dims per layer: first takes in_dim, the rest take hidden_dim
        layer_in_dims = [in_dim] + [hidden_dim] * (num_layers - 1)
        for d_in in layer_in_dims:
            gin_mlp = nn.Sequential(
                nn.Linear(d_in, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(gin_mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.vn_mlps.append(_make_vn_mlp(hidden_dim))

        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, root_mask, batch):
        """
        Args:
            x:          node features  [total_nodes, in_dim]
            edge_index: COO edge list  [2, total_edges]
            root_mask:  bool mask selecting one root per graph [total_nodes]
            batch:      graph assignment vector [total_nodes]
        Returns:
            logits:     [batch_size, out_dim]
        """
        num_graphs = int(batch.max()) + 1
        vn_emb = torch.zeros(num_graphs, self.hidden_dim, device=x.device)

        for i, (conv, bn, vn_mlp) in enumerate(zip(self.convs, self.bns, self.vn_mlps)):
            # Broadcast VN to all nodes (skip layer 0)
            if i > 0:
                x = x + vn_emb[batch]

            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)

            vn_emb = vn_mlp(global_mean_pool(x, batch) + vn_emb)

        return self.classifier(x[root_mask])
