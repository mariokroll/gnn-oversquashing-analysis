"""
Dataset generation for the Tree-NeighborsMatch task.

This code is based on the original Tree-NeightborsMatch implementation from https://github.com/tech-srl/bottleneck
"""

import itertools
import math
import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class TreeDataset(object):
    def __init__(self, depth):
        super(TreeDataset, self).__init__()
        self.depth = depth
        self.num_nodes, self.edges, self.leaf_indices = self._create_blank_tree()
        self.criterion = F.cross_entropy

    def add_child_edges(self, cur_node, max_node):
        edges = []
        leaf_indices = []
        stack = [(cur_node, max_node)]
        while len(stack) > 0:
            cur_node, max_node = stack.pop()
            if cur_node == max_node:
                leaf_indices.append(cur_node)
                continue

            left_child = cur_node + 1
            right_child = cur_node + 1 + ((max_node - cur_node) // 2)
            edges.append([left_child, cur_node])
            edges.append([right_child, cur_node])
            stack.append((right_child, max_node))
            stack.append((left_child, right_child - 1))

        return edges, leaf_indices

    def _create_blank_tree(self):
        max_node_id = 2 ** (self.depth + 1) - 2
        edges, leaf_indices = self.add_child_edges(cur_node=0, max_node=max_node_id)
        return max_node_id + 1, edges, leaf_indices

    def create_blank_tree(self):
        return torch.tensor(self.edges, dtype=torch.long).t().contiguous()

    def encode_node_features(self, nodes_raw):
        """Encode root query and leaf key-value pairs without losing pair identity."""
        num_leaves = len(self.leaf_indices)
        num_symbols = num_leaves + 1

        raw = torch.tensor(nodes_raw, dtype=torch.long)
        query = torch.zeros(self.num_nodes, num_symbols)
        pairs = torch.zeros(self.num_nodes, num_symbols * num_symbols)

        selected_key = raw[0, 0]
        query[0, selected_key] = 1.0

        for leaf_idx in self.leaf_indices:
            key_idx = raw[leaf_idx, 0]
            value_idx = raw[leaf_idx, 1]
            pair_idx = key_idx * num_symbols + value_idx
            pairs[leaf_idx, pair_idx] = 1.0

        return torch.cat([query, pairs], dim=1)

    def generate_data(self, train_fraction, batch_size=32):
        """Generate the full dataset and return DataLoader objects.

        The root stores the queried key as a one-hot vector. Each leaf stores
        its full (key, value) pair as one joint one-hot vector, which preserves
        the association when sibling messages are aggregated.
        """
        data_list = []

        for comb in self.get_combinations():
            edge_index = self.create_blank_tree()
            x = self.encode_node_features(self.get_nodes_features(comb))

            root_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
            root_mask[0] = True

            label = self.label(comb) - 1
            data_list.append(
                Data(x=x, edge_index=edge_index, root_mask=root_mask, y=label)
            )

        in_dim, out_dim = self.get_dims()

        X_train, X_test = train_test_split(
            data_list,
            train_size=train_fraction,
            shuffle=True,
            stratify=[data.y for data in data_list],
        )

        train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(X_test, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader, in_dim, out_dim, self.criterion

    def get_combinations(self):
        # returns: an iterable of [key, permutation(leaves)]
        # number of combinations: (num_leaves!) * num_choices
        num_leaves = len(self.leaf_indices)
        num_permutations = 1000
        max_examples = 32000

        if self.depth > 3:
            per_depth_num_permutations = min(
                num_permutations, math.factorial(num_leaves), max_examples // num_leaves
            )
            permutations = [
                np.random.permutation(range(1, num_leaves + 1))
                for _ in range(per_depth_num_permutations)
            ]
        else:
            permutations = random.sample(
                list(itertools.permutations(range(1, num_leaves + 1))),
                min(num_permutations, math.factorial(num_leaves)),
            )

        return itertools.chain.from_iterable(
            zip(range(1, num_leaves + 1), itertools.repeat(perm))
            for perm in permutations
        )

    def get_nodes_features(self, combination):
        selected_key, values = combination
        nodes = [(selected_key, 0)]

        for i in range(1, self.num_nodes):
            if i in self.leaf_indices:
                leaf_num = self.leaf_indices.index(i)
                node = (leaf_num + 1, values[leaf_num])
            else:
                node = (0, 0)
            nodes.append(node)

        return nodes

    def label(self, combination):
        selected_key, values = combination
        return int(values[selected_key - 1])

    def get_dims(self):
        """Return feature and output dimensions for the encoded task."""
        num_leaves = len(self.leaf_indices)
        num_symbols = num_leaves + 1
        in_dim = num_symbols + num_symbols * num_symbols
        out_dim = num_leaves
        return in_dim, out_dim
