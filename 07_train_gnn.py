# train_gnn.py
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sentence_transformers import SentenceTransformer
from gnn_model import QueryGNN
from gnn_graph_loader import load_graph

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_doc(doc_name, num_epochs=30, lr=1e-4):
    print(f"📄 Training on document: {doc_name}")

    # 1. 그래프 로딩
    data = load_graph(doc_name).to(DEVICE)

    # 2. 질의 임베딩 (instruction이 있는 노드를 찾아 임베딩)
    instruction_node = None
    for i, y in enumerate(data.y):
        if y == 1:
            instruction_node = data.node_instruction[i]
            break

    if instruction_node is None:
        print(f"[!] No instruction label found in {doc_name}")
        return

    # Sentence embedding
    query_encoder = SentenceTransformer("jhgan/ko-sroberta-multitask")
    query_vec = torch.tensor(query_encoder.encode(instruction_node), dtype=torch.float).to(DEVICE)

    # 3. GNN 모델 초기화
    model = QueryGNN(in_dim=data.x.shape[1]).to(DEVICE)
    optim = Adam(model.parameters(), lr=lr)

    # 4. 훈련 루프
    for epoch in range(num_epochs):
        model.train()
        optim.zero_grad()

        logits = model(data, query_vec)  # (N,)
        loss = F.binary_cross_entropy_with_logits(logits, data.y.float().to(DEVICE))

        loss.backward()
        optim.step()

        print(f"[{epoch+1:02d}] Loss: {loss.item():.4f}")

    print("🎉 Training complete!")

if __name__ == "__main__":
    train_one_doc("MI3_241101_TY1_1494_2")  # 예시 문서명
