from gnn_graph_loader import load_graph

data = load_graph("MI3_241101_TY1_1494_2")   # 예시
print(data)
print(data.x.shape)         # node feature 수
print(data.edge_index)
print(data.y)
