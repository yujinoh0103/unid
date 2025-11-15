# gnn_dataset_builder.py (최적화 + instruction edge 포함)

import re
import json
import pickle
import torch
import pandas as pd
from pathlib import Path
from itertools import permutations
from tqdm.auto import tqdm

import config
import utils

# CrossEncoder + BiEncoder
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.cross_encoder import CrossEncoder


VISUAL_ID_PATTERN = r"##(MI3_[^#]+)##"


def extract_visual_ids(text: str):
    return re.findall(VISUAL_ID_PATTERN, text or "")


def get_bbox_xywh(bbox):
    """
    bounding_box = [x, y, w, h] 형식을 지원하는 함수.
    """
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        x, y, w, h = bbox[:4]
        return x, y, w, h

    if isinstance(bbox, dict):
        # 혹시 dict로 올 수도 있으니 안정성 확보
        x = bbox.get("x", 0)
        y = bbox.get("y", 0)
        w = bbox.get("w", 0)
        h = bbox.get("h", 0)
        return x, y, w, h

    return 0, 0, 0, 0


def build_graph_for_document(json_path: Path, bi_model, cross_model, output_dir: Path, top_k=20):
    data = utils.load_json(json_path)
    annotations = utils.get_annotations(data)

    if not annotations:
        print(f"[!] Annotation 없음: {json_path.name}")
        return
    
    # resolution
    page_w, page_h = data["source_data_info"]["document_resolution"]

    # --- 1) 노드 생성 ---
    node_rows = []
    texts = []
    visual_map = {}  # instance_id → node_id

    for idx, ann in enumerate(annotations):
        text = ann.get("text", "")
        inst_id = ann.get("instance_id")
        vis_ids = extract_visual_ids(text)

        x, y, w, h = get_bbox_xywh(ann["bounding_box"])

        cls = ann.get("class_name", "기타")
        if cls not in config.ALLOWED_CLASS_NAMES:
            cls = "기타"

        # visual instruction / answer
        visual_instruction = ann.get("visual_instruction", "")
        visual_answer = ann.get("visual_answer", "")

        node_rows.append({
            "node_id": idx,
            "class": cls,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "text": text,
            "visual_ids": "|".join(vis_ids),
            "instruction": visual_instruction,
            "answer": visual_answer,
            "instance_id": inst_id,
        })

        texts.append(text)

        if inst_id:
            visual_map[inst_id] = idx

    node_df = pd.DataFrame(node_rows)

    # --- 2) Bi-Encoder semantic pre-filtering ---
    embeddings = bi_model.encode(texts, convert_to_tensor=True, show_progress_bar=False)

    semantic_candidates = []
    N = len(texts)

    for i in range(N):
        cos_scores = util.cos_sim(embeddings[i], embeddings)[0]
        top_results = torch.topk(cos_scores, k=min(top_k + 1, N))
        
        for score, j in zip(top_results[0], top_results[1]):
            if i == j:
                continue
            semantic_candidates.append((i, j, float(score)))

    # --- 3) CrossEncoder 정교 점수 계산 ---
    cross_inputs = [[texts[i], texts[j]] for (i, j, _) in semantic_candidates]
    cross_scores = cross_model.predict(cross_inputs, batch_size=32, show_progress_bar=False)

    # --- 4) Edge 생성 ---
    edge_rows = []
    k = 0
    for (i, j, coarse_sim) in semantic_candidates:
        score = cross_scores[k].item()
        k += 1

        # dx dy 계산
        ann_a = annotations[i]
        ann_b = annotations[j]
        cx_a, cy_a = utils.get_centroid(ann_a["bounding_box"])
        cx_b, cy_b = utils.get_centroid(ann_b["bounding_box"])

        norm_dx = (cx_b - cx_a) / page_w
        norm_dy = (cy_b - cy_a) / page_h

        # visual link
        vis_a = set(extract_visual_ids(texts[i]))
        vis_b = set(extract_visual_ids(texts[j]))
        visual_link = int(len(vis_a & vis_b) > 0)

        edge_rows.append({
            "src": i,
            "dst": j,
            "dx": norm_dx,
            "dy": norm_dy,
            "semantic_sim": score,
            "visual_link": visual_link,
            "in_topk": 1
        })

    # --- 5) Instruction Edge (Supervised) ---
    for idx, ann in enumerate(annotations):
        inst_text = ann.get("visual_instruction", "")
        inst_answer = ann.get("visual_answer", "")
        inst_id = ann.get("instance_id")

        if not inst_text or not inst_id:
            continue

        # instruction → 표/차트 본체 연결
        if inst_id in visual_map:
            target_node = visual_map[inst_id]
            edge_rows.append({
                "src": idx,
                "dst": target_node,
                "dx": 0,
                "dy": 0,
                "semantic_sim": 1.0,
                "visual_link": 1,
                "in_topk": 0
            })

    edge_df = pd.DataFrame(edge_rows)

    # --- 저장 ---
    (output_dir / "nodes").mkdir(parents=True, exist_ok=True)
    (output_dir / "edges").mkdir(parents=True, exist_ok=True)

    doc_name = json_path.stem
    node_df.to_csv(output_dir / "nodes" / f"{doc_name}_nodes.csv", index=False, encoding="utf-8-sig")
    edge_df.to_csv(output_dir / "edges" / f"{doc_name}_edges.csv", index=False, encoding="utf-8-sig")

    print(f"📄 그래프 생성 완료: {doc_name}")


def main():
    utils.setup_korean_font()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Bi-Encoder loading on {device}")
    bi_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device=device)

    print(f"CrossEncoder loading on {device}")
    cross_model = CrossEncoder(config.CROSS_ENCODER_MODEL_NAME, device=device)

    sample_pkl = config.OUTPUT_DIR / "sample_index_20percent.pkl"
    with open(sample_pkl, "rb") as f:
        sampled = pickle.load(f)

    feature_files = []
    for split in ["train", "valid"]:
        for doc_type in ["report", "press"]:
            for fi in sampled.get(split, {}).get(doc_type, []):
                p = Path(fi["json_path"])
                folder = p.parent.name
                json_f = config.FEATURES_JSON_DIR / folder / p.name
                if json_f.exists():
                    feature_files.append(json_f)

    output_dir = config.OUTPUT_DIR / "05_gnn_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"총 {len(feature_files)}개 문서 그래프 생성 시작")

    for f in tqdm(feature_files):
        build_graph_for_document(f, bi_model, cross_model, output_dir)

    print("🎉 모든 문서 GNN 데이터 생성 완료!")


if __name__ == "__main__":
    main()
