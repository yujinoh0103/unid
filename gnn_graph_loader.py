# gnn_graph_loader.py
import pandas as pd
import torch
from torch_geometric.data import Data
import re
import config

def clean_id(x):
    """src/dst 컬럼이 tensor(7, device='cuda:0') 같은 문자열이면 숫자만 추출"""
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x)
    if isinstance(x, str):
        # tensor(7, device='cuda:0') → 7
        m = re.search(r"(\d+)", x)
        if m:
            return int(m.group(1))
    # 어떤 경우에도 매칭 안 되면 0
    return 0

def load_graph(doc_name: str):
    node_csv = config.OUTPUT_DIR / "05_gnn_dataset" / "nodes" / f"{doc_name}_nodes.csv"
    edge_csv = config.OUTPUT_DIR / "05_gnn_dataset" / "edges" / f"{doc_name}_edges.csv"
    feat_pt  = config.OUTPUT_DIR / "06_node_features" / f"{doc_name}_nodes.pt"

    df_node = pd.read_csv(node_csv)
    df_edge = pd.read_csv(edge_csv)

    # 🔥 src/dst를 강제로 int로 변환
    df_edge["src"] = df_edge["src"].apply(clean_id)
    df_edge["dst"] = df_edge["dst"].apply(clean_id)

    x = torch.load(feat_pt)

    edge_index = torch.tensor(df_edge[["src","dst"]].values.T, dtype=torch.long)
    edge_attr = torch.tensor(df_edge[["dx","dy","semantic_sim","visual_link"]].values, dtype=torch.float)

    y = torch.tensor([
        1 if isinstance(row["instruction"], str) and row["instruction"].strip() else 0
        for _, row in df_node.iterrows()
    ], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
