import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv


class GCN(nn.Module):
    """Standard Graph Convolutional Network (Kipf & Welling, 2017).

    Stack of GCNConv → BatchNorm → ReLU layers followed by a linear
    classifier applied exclusively to the root node of each graph.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int):
        """
        Args:
            in_dim:     input feature size from TreeDataset.get_dims()
            hidden_dim: width of every hidden layer
            out_dim:    number of output classes (num_leaves)
            num_layers: number of GCNConv layers
        """
        super(GCN, self).__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First conv projects from in_dim → hidden_dim
        self.convs.append(GCNConv(in_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Remaining convs stay in hidden_dim space
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Classifier applied to the root node embedding only
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, root_mask, batch=None):
        """
        Args:
            x:          node features  [total_nodes, in_dim]
            edge_index: COO edge list  [2, total_edges]
            root_mask:  bool mask selecting one root per graph [total_nodes]
            batch:      batch assignment vector (unused; kept for API parity)
        Returns:
            logits:     [batch_size, out_dim]
        """

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)

        # root_mask selects exactly one node per graph in the batch (root)
        return self.classifier(x[root_mask])


class GIN(nn.Module):
    """Graph Isomorphism Network (Xu et al., 2019).

    Uses GINConv layers, each backed by a 2-layer MLP (more expressive than
    GCNConv under the 1-WL framework), but still subject to over-squashing on
    deep trees because the bottleneck is structural, not representational.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int):
        """
        Args:
            in_dim:     input feature size
            hidden_dim: width of every hidden layer
            out_dim:    number of output classes
            num_layers: number of GINConv layers
        """
        super(GIN, self).__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Each GINConv wraps a dedicated 2-layer MLP
        layer_in_dims = [in_dim] + [hidden_dim] * (num_layers - 1)
        for d_in in layer_in_dims:
            mlp = nn.Sequential(
                nn.Linear(d_in, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            # train_eps=True lets the model learn its own epsilon parameter
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, root_mask, batch=None):
        """
        Args:
            x:          node features  [total_nodes, in_dim]
            edge_index: COO edge list  [2, total_edges]
            root_mask:  bool mask selecting one root per graph [total_nodes]
            batch:      graph assignment vector (unused; kept for API parity)
        Returns:
            logits:     [batch_size, out_dim]
        """
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)

        return self.classifier(x[root_mask])
