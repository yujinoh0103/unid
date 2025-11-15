import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class QueryGNN(nn.Module):
    """
    단일 문서를 Graph로 보고 node-classification or node-scoring 하는 GNN 모델
    node_feature_dim: node_feature.pt에서 불러오는 차원
    """
    def __init__(self, node_feature_dim, hidden_dim=256, num_heads=4):
        super().__init__()

        self.gat1 = GATConv(node_feature_dim, hidden_dim, heads=num_heads, concat=True)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False)

        # 최종적으로 "score"를 뽑아내는 FC layer
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        """
        x: node features (N, F)
        edge_index: (2, E)
        """
        h = self.gat1(x, edge_index)
        h = torch.relu(h)

        h = self.gat2(h, edge_index)
        h = torch.relu(h)

        out = self.classifier(h)  # (N, 1)
        return out.squeeze(-1)    # (N,)
