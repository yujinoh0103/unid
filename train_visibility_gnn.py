import torch
import pandas as pd
from pathlib import Path
from torch_geometric.data import Data
from gnn_visible_model import VisibilityGNNv3
import config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# Load ONE graph (nodes + edges)
# ------------------------------------------------------------
def load_graph(nodes_csv, edges_csv):
    df_nodes = pd.read_csv(nodes_csv)
    df_edges = pd.read_csv(edges_csv)

    # Node features ------------------------------------------
    # bbox + hist + engineered features + text embedding
    node_cols = [
        "x","y","w","h",
        "num_hist","alpha_hist","symbol_hist",
        "feat_text_length",
        "feat_startsWithListSymbol",
        "feat_containsRefSymbol",
        "feat_isPageNumberFormat",
        "feat_containsTableKeyword",
    ]

    # append all emb_* columns
    emb_cols = [c for c in df_nodes.columns if c.startswith("emb_")]
    node_cols.extend(emb_cols)

    x = torch.tensor(df_nodes[node_cols].values, dtype=torch.float)

    # Labels --------------------------------------------------
    # (미리 포함되어 있을 경우)
    if "is_visual_element" in df_nodes.columns:
        y = torch.tensor(df_nodes["is_visual_element"].values, dtype=torch.float)
    else:
        raise RuntimeError("nodes_csv에 is_visual_element 라벨이 없습니다.")

    # Edge index ---------------------------------------------
    edge_index = torch.tensor(df_edges[["src","dst"]].values.T, dtype=torch.long)

    # Edge feature (edge_attr) -------------------------------
    edge_cols = ["dx","dy","semantic_sim","visual_link"]
    edge_attr = torch.tensor(df_edges[edge_cols].values, dtype=torch.float)

    return Data(
        x=x.to(DEVICE),
        edge_index=edge_index.to(DEVICE),
        edge_attr=edge_attr.to(DEVICE),
        y=y.to(DEVICE)
    )

# ------------------------------------------------------------
# Load ALL graphs in dataset folder
# ------------------------------------------------------------
def load_all_graphs():
    nodes_dir = config.GNN_NODES_DIR
    edges_dir = config.GNN_EDGES_DIR

    data_list = []

    for nodes_file in sorted(nodes_dir.glob("*_nodes.csv")):
        name = nodes_file.name.replace("_nodes.csv","")
        edges_file = edges_dir / f"{name}_edges.csv"

        if edges_file.exists():
            print(f"[Load] {name}")
            data = load_graph(nodes_file, edges_file)
            data_list.append(data)

    return data_list

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------
data_list = load_all_graphs()

# Assume single graph training first
data = data_list[0]

in_dim = data.x.size(1)
edge_dim = data.edge_attr.size(1)

model = VisibilityGNNv3(
    in_dim=in_dim,
    edge_dim=edge_dim,
    hidden_dim=128,
    hops=3,
    num_layers=3
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(30):
    model.train()
    optimizer.zero_grad()

    scores = model(data.x, data.edge_index, data.edge_attr)
    loss = torch.nn.functional.binary_cross_entropy(scores, data.y)

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch:02d} loss: {loss.item():.4f}")
