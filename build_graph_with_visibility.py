"""
Visibility Graph + Semantic Edges 통합 버전
"""
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from tqdm.auto import tqdm

# Visibility graph builder import
from visibility_graph_builder import build_visibility_graph

# ============================================================
# CONFIG
# ============================================================
TOP_K_SEMANTIC = 10  # semantic edges per node
EMB_DIM = 384
USE_VISIBILITY = True  # Visibility graph 사용 여부

# ============================================================
# LOAD JSON
# ============================================================
def load_annotations(json_path):
    """JSON에서 annotations 로드"""
    data = json.load(open(json_path, "r", encoding="utf-8"))
    anns = data.get("annotations", [])
    doc_res = data.get("document_resolution", [2480, 3508])
    return anns, doc_res

# ============================================================
# TEXT EMBEDDING
# ============================================================
def embed_texts(bi_model, texts):
    """텍스트 임베딩"""
    return bi_model.encode(texts, convert_to_tensor=True, show_progress_bar=False)

# ============================================================
# EDGE FEATURE CALCULATION
# ============================================================
def compute_edge_features(nodes_df, src, dst):
    """
    엣지 특징 계산:
    - dx, dy: spatial offset (normalized)
    - semantic_sim: 이미 계산됨
    - visual_link: instruction edge 여부
    """
    src_row = nodes_df.iloc[src]
    dst_row = nodes_df.iloc[dst]
    
    # Spatial offset (중심점 기준)
    src_cx = src_row["x"] + src_row["w"] / 2
    src_cy = src_row["y"] + src_row["h"] / 2
    dst_cx = dst_row["x"] + dst_row["w"] / 2
    dst_cy = dst_row["y"] + dst_row["h"] / 2
    
    # Normalize by document size (2480 x 3508)
    dx = (dst_cx - src_cx) / 2480
    dy = (dst_cy - src_cy) / 3508
    
    return dx, dy

# ============================================================
# GRAPH BUILDER (Visibility + Semantic)
# ============================================================
def build_graph_hybrid(json_path: Path, bi_model, output_dir: Path, 
                      top_k=TOP_K_SEMANTIC, use_visibility=USE_VISIBILITY):
    """
    Hybrid graph builder:
    1. Visibility edges (구조적)
    2. Semantic edges (의미적)
    3. Instruction edges (supervised)
    """
    anns, doc_res = load_annotations(json_path)
    if not anns:
        print(f"[!] No annotations in {json_path}")
        return
    
    doc_width, doc_height = doc_res
    
    # ========================================
    # 1. NODE FEATURES
    # ========================================
    texts = []
    nodes = []
    
    for i, ann in enumerate(anns):
        # Bbox
        bbox = ann.get("bounding_box") or ann.get("bbox")
        if not bbox or len(bbox) < 4:
            bbox = [0, 0, 100, 100]
        
        x, y, w, h = bbox
        
        # Histograms
        num_h = ann.get("num_hist", 0)
        alp_h = ann.get("alpha_hist", 0)
        sym_h = ann.get("symbol_hist", 0)
        
        # Text
        t = ann.get("text", "")
        texts.append(t)
        
        # Class
        cls = ann.get("class_name", "기타")
        
        # Instance ID & Instruction
        iid = ann.get("instance_id", "")
        instr = ann.get("visual_instruction", "")
        
        nodes.append({
            "node_id": i,
            "class": cls,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "num_hist": num_h,
            "alpha_hist": alp_h,
            "symbol_hist": sym_h,
            "text": t,
            "instance_id": iid,
            "instruction": instr,
        })
    
    df_nodes = pd.DataFrame(nodes)
    N = len(df_nodes)
    
    # ========================================
    # 2. TEXT EMBEDDING
    # ========================================
    emb = embed_texts(bi_model, texts).cpu().numpy()  # [N, EMB_DIM]
    
    for d in range(EMB_DIM):
        df_nodes[f"emb_{d}"] = emb[:, d]
    
    # ========================================
    # 3. BUILD EDGES
    # ========================================
    all_edges = []
    
    # --- 3.1 Visibility Edges ---
    if use_visibility:
        print(f"  [Visibility] Building graph...")
        _, vis_edges = build_visibility_graph(df_nodes, long_edge_thresh=0.3)
        
        for src, dst in vis_edges:
            dx, dy = compute_edge_features(df_nodes, src, dst)
            
            # Semantic similarity (이미 임베딩 계산됨)
            sem_sim = util.cos_sim(
                torch.tensor(emb[src]), 
                torch.tensor(emb[dst])
            ).item()
            
            all_edges.append({
                "src": src,
                "dst": dst,
                "dx": dx,
                "dy": dy,
                "semantic_sim": sem_sim,
                "visual_link": 0,  # 구조적 엣지
                "edge_type": "visibility"
            })
        
        print(f"  [Visibility] {len(vis_edges)} edges")
    
    # --- 3.2 Semantic Edges (Top-K) ---
    print(f"  [Semantic] Building top-{top_k} edges...")
    semantic_count = 0
    
    for i in range(N):
        cos_scores = util.cos_sim(torch.tensor(emb[i]), torch.tensor(emb))[0]
        top = torch.topk(cos_scores, k=min(top_k + 1, N))
        
        for score, j in zip(top[0], top[1]):
            j = int(j)
            if i == j:
                continue
            
            # Visibility에 이미 있으면 스킵 (중복 방지)
            if use_visibility:
                existing = any(e["src"] == i and e["dst"] == j for e in all_edges)
                if existing:
                    continue
            
            dx, dy = compute_edge_features(df_nodes, i, j)
            
            all_edges.append({
                "src": i,
                "dst": j,
                "dx": dx,
                "dy": dy,
                "semantic_sim": float(score),
                "visual_link": 0,
                "edge_type": "semantic"
            })
            semantic_count += 1
    
    print(f"  [Semantic] {semantic_count} edges")
    
    # --- 3.3 Instruction Edges (Supervised) ---
    print(f"  [Instruction] Building supervised edges...")
    
    # instance_id → visual node mapping
    inst_map = {}
    for i, row in df_nodes.iterrows():
        iid = row["instance_id"]
        if iid and iid != "":
            inst_map.setdefault(iid, []).append(i)
    
    # Instruction nodes → Visual nodes
    instruction_count = 0
    for i, row in df_nodes.iterrows():
        iid = row["instance_id"]
        instr = row["instruction"]
        
        if instr and iid and iid in inst_map:
            for target_node in inst_map[iid]:
                if i == target_node:
                    continue
                
                dx, dy = compute_edge_features(df_nodes, i, target_node)
                sem_sim = util.cos_sim(
                    torch.tensor(emb[i]), 
                    torch.tensor(emb[target_node])
                ).item()
                
                all_edges.append({
                    "src": i,
                    "dst": target_node,
                    "dx": dx,
                    "dy": dy,
                    "semantic_sim": sem_sim,
                    "visual_link": 1,  # Supervised edge
                    "edge_type": "instruction"
                })
                instruction_count += 1
    
    print(f"  [Instruction] {instruction_count} edges")
    
    # ========================================
    # 4. CREATE DATAFRAME
    # ========================================
    df_edges = pd.DataFrame(all_edges)
    
    print(f"  [Total] {len(df_edges)} edges")
    
    # ========================================
    # 5. SAVE
    # ========================================
    (output_dir / "nodes").mkdir(parents=True, exist_ok=True)
    (output_dir / "edges").mkdir(parents=True, exist_ok=True)
    
    name = json_path.stem
    df_nodes.to_csv(
        output_dir / "nodes" / f"{name}_nodes.csv", 
        index=False, 
        encoding="utf-8-sig"
    )
    df_edges.to_csv(
        output_dir / "edges" / f"{name}_edges.csv", 
        index=False, 
        encoding="utf-8-sig"
    )
    
    print(f"✅ Saved: {name}")

# ============================================================
# MAIN
# ============================================================
def main():
    import config
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Bi-Encoder loading on {device}")
    
    bi_model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device=device
    )
    
    # 경로 설정
    input_dir = config.FEATURES_JSON_DIR  # 03_output_json_features
    output_dir = config.OUTPUT_DIR / "05_gnn_dataset_v2"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    json_files = list(input_dir.glob("*.json"))
    print(f"📁 Found {len(json_files)} JSON files")
    
    for js in tqdm(json_files, desc="Building graphs"):
        try:
            build_graph_hybrid(
                js, 
                bi_model, 
                output_dir,
                top_k=TOP_K_SEMANTIC,
                use_visibility=USE_VISIBILITY
            )
        except Exception as e:
            print(f"❌ Error processing {js.name}: {e}")
            continue
    
    print("\n🎉 All graphs built!")

if __name__ == "__main__":
    main()