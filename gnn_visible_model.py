import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# Utility: adjacency power computation
# ------------------------------------------------------------
def compute_adj_powers(edge_index, num_nodes, k):
    A = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1)), (num_nodes, num_nodes))
    adj_list = [A]
    cur = A
    for _ in range(1, k):
        cur = torch.sparse.mm(cur, A)
        adj_list.append(cur.coalesce())
    return adj_list

# ------------------------------------------------------------
# Learnable adjacency weight layer (edge_attr REQUIRED)
# ------------------------------------------------------------
class LearnableEdgeWeight(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        diff = x[src] - x[dst]
        mlp_in = torch.cat([diff, edge_attr], dim=-1)
        w = torch.sigmoid(self.mlp(mlp_in)).squeeze(-1)
        return w

# ------------------------------------------------------------
# Multi-hop GNN layer
# ------------------------------------------------------------
class MultiHopGNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, hops=3):
        super().__init__()
        self.hops = hops
        self.linears = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(hops)])

    def forward(self, x, adj_list, edge_weights_list):
        out = 0
        for j in range(self.hops):
            A = adj_list[j]
            w = edge_weights_list[j]
            xw = x * w.unsqueeze(-1)
            Ax = torch.sparse.mm(A, xw)
            out = out + self.linears[j](Ax)
        return torch.relu(out)

# ------------------------------------------------------------
# Residual Block
# ------------------------------------------------------------
class GraphResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, hops=3):
        super().__init__()
        self.layer1 = MultiHopGNNLayer(in_dim, out_dim, hops)
        self.layer2 = MultiHopGNNLayer(out_dim, out_dim, hops)
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else None

    def forward(self, x, adj_list, edge_weights_list):
        res = x if self.proj is None else self.proj(x)
        out = self.layer1(x, adj_list, edge_weights_list)
        out = self.layer2(out, adj_list, edge_weights_list)
        return torch.relu(out + res)

# ------------------------------------------------------------
# Final Model (edge_attr REQUIRED)
# ------------------------------------------------------------
class VisibilityGNNv3(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden_dim=128, hops=3, num_layers=3):
        super().__init__()
        self.hops = hops

        self.edge_weight_layers = nn.ModuleList([
            LearnableEdgeWeight(
                node_dim=(in_dim if i == 0 else hidden_dim),
                edge_dim=edge_dim
            )
            for i in range(num_layers)
        ])

        self.blocks = nn.ModuleList([
            GraphResidualBlock(
                in_dim if i == 0 else hidden_dim,
                hidden_dim,
                hops
            )
            for i in range(num_layers)
        ])

        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_attr):
        num_nodes = x.size(0)
        adj_list = compute_adj_powers(edge_index, num_nodes, self.hops)

        for block, ew_layer in zip(self.blocks, self.edge_weight_layers):
            w = ew_layer(x, edge_index, edge_attr)
            edge_weights_list = [w for _ in range(self.hops)]
            x = block(x, adj_list, edge_weights_list)

        return torch.sigmoid(self.out(x)).squeeze(-1)