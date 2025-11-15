import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import permutations
import pickle
import torch
from tqdm.auto import tqdm # [추가] 프로그레스 바

# --- 1. [수정] config.py와 utils.py 임포트 ---
import config
import utils

# (중요) CrossEncoder를 import합니다.
from sentence_transformers.cross_encoder import CrossEncoder

# --- 2. [수정] config.py에서 경로 로드 ---
# [입력 1] VLM 샘플 인덱스
SAMPLE_INDEX_PATH = config.OUTPUT_DIR / "sample_index_20percent.pkl" 
# [입력 2] 텍스트 특징이 추출된 JSON 폴더
FEATURES_JSON_DIR = config.FEATURES_JSON_DIR
# [출력] 분석 결과
OUTPUT_DIR = config.ANALYSIS_RESULT_DIR 

# [모델] config.py에서 모델 이름 로드
CROSS_ENCODER_MODEL_NAME = config.CROSS_ENCODER_MODEL_NAME

# --- (get_centroid, get_annotations, setup_korean_font 함수는 utils.py로 이동하여 삭제) ---


# --- [수정] main 함수 (GPU 최적화 및 모듈화) ---
def main():
    # utils.py 함수 호출
    utils.setup_korean_font()
    
    # --- 1. CUDA 장치 설정 ---
    device = None
    if torch.cuda.is_available():
        device = 'cuda'
        print("✅ CUDA (GPU) 사용 가능. GPU로 실행합니다.")
    else:
        device = 'cpu'
        print("⚠️ CUDA (GPU)를 찾을 수 없습니다. CPU로 실행합니다. (매우 느릴 수 있음)")

    # --- 2. Cross-Encoder 모델 로드 (config.py 및 device 사용) ---
    print(f"Cross-Encoder '{CROSS_ENCODER_MODEL_NAME}' 모델을 로드합니다...")
    try:
        model = CrossEncoder(CROSS_ENCODER_MODEL_NAME, device=device)
        print("모델 로드 완료.")
    except Exception as e:
        print(f"[치명적 오류] Cross-Encoder 모델 로드 실패: {e}")
        return

    # --- 3. [수정] .pkl 파일 로드하여 all_json_files 리스트 생성 ---
    print(f"샘플 인덱스 파일 로드: {SAMPLE_INDEX_PATH}")
    if not SAMPLE_INDEX_PATH.exists():
        print(f"[치명적 오류] 샘플 인덱스 파일을 찾을 수 없습니다: {SAMPLE_INDEX_PATH}")
        print("  먼저 '06_create_ml_dataset.py'를 실행해야 합니다.")
        return

    with open(SAMPLE_INDEX_PATH, 'rb') as f:
        sampled_data = pickle.load(f)
    
    feature_json_files = [] # [수정] pkl 경로를 feature 경로로 변환하여 저장
    
    for split in ['train', 'valid']:
        for doc_type in ['report', 'press']:
            files_list = sampled_data.get(split, {}).get(doc_type, [])
            for file_info in files_list:
                # --- [핵심] 경로 재구성 ---
                pkl_path = Path(file_info['json_path'])
                doc_type_folder = pkl_path.parent.name # 예: "report_json"
                file_name = pkl_path.name             # 예: "MI3_...json"
                
                # 'output_json_features' 폴더 기준으로 경로 재조립
                feature_path = FEATURES_JSON_DIR / doc_type_folder / file_name
                
                if feature_path.exists():
                    feature_json_files.append(feature_path)
    if not feature_json_files:
        print(f"[오류] 분석할 Feature JSON 파일이 없습니다. '{FEATURES_JSON_DIR}' 폴더를 확인하세요.")
        return

    # --- 4. 모든 파일에서 '엣지(Edge)' 특징 추출 ---
    all_edge_data = []
    print(f"총 {len(feature_json_files)}개의 '샘플링된 Feature JSON' 파일을 분석합니다...")

    # [수정] tqdm 프로그레스 바 추가
    for json_file in tqdm(feature_json_files, desc="Analyzing Semantic Relations"):
        source_type = json_file.parent.name 
        try:
            # [수정] utils.py 함수 사용
            data = utils.load_json(json_file)
            annotations = utils.get_annotations(data)
            
            resolution = data.get("source_data_info", {}).get("document_resolution")
            if not annotations or not resolution: continue
            
            page_width, page_height = resolution

            # --- 5. (핵심) 모든 (A, B) 쌍 생성 ---
            text_pairs_to_predict = [] 
            edge_metadata = []         

            for i, j in permutations(range(len(annotations)), 2):
                item_a = annotations[i]
                item_b = annotations[j]
                
                text_a = item_a.get('text', '') # (03_ 스크립트가 생성한 text)
                text_b = item_b.get('text', '')
                text_pairs_to_predict.append([text_a, text_b])
                
                # [수정] utils.py 함수 사용
                cx_a, cy_a = utils.get_centroid(item_a['bounding_box'])
                cx_b, cy_b = utils.get_centroid(item_b['bounding_box'])
                norm_dx = (cx_b - cx_a) / page_width
                norm_dy = (cy_b - cy_a) / page_height
                
                edge_metadata.append((
                    item_a['class_name'], item_b['class_name'], 
                    norm_dx, norm_dy, source_type
                ))

            if not text_pairs_to_predict:
                continue 
                
            # --- 6. (핵심) 모델 실행 (GPU 사용 및 batch_size 지정) ---
            # [수정] print문 제거 (tqdm이 진행률 표시)
            # print(f"  > {json_file.name}: {len(text_pairs_to_predict)}개의 관계 쌍을 Cross-Encoding 중... (GPU 사용)")
            scores = model.predict(text_pairs_to_predict, 
                                   batch_size=32, # GPU 메모리에 맞게 조절
                                   show_progress_bar=False) 
            
            # --- 7. 결과 취합 ---
            for k in range(len(scores)):
                metadata = edge_metadata[k] 
                similarity = scores[k].item() # .item()으로 숫자만 추출
                all_edge_data.append(metadata + (similarity,))

        except Exception as e:
            print(f"\n  [오류] {json_file.name} 처리 중: {e}")
            continue

    if not all_edge_data:
        print("[분석 결과] 유효한 엣지(관계)를 찾지 못했습니다.")
        return

    # --- 8. (Pandas) 최종 통계 집계 ---
    df = pd.DataFrame(all_edge_data, columns=[
        'class_A', 'class_B', 'dx', 'dy', 'source', 'similarity'
    ])

    summary = df.groupby(['class_A', 'class_B']).agg(
        count=('dx', 'size'),
        mean_dx=('dx', 'mean'),
        std_dx=('dx', 'std'),
        mean_dy=('dy', 'mean'),
        std_dy=('dy', 'std'),
        mean_similarity=('similarity', 'mean')
    ).reset_index()

    # [수정] 결과 CSV 저장 (config.py 경로 사용)
    result_csv_path = OUTPUT_DIR / "semantic_analysis_summary.csv"
    summary.to_csv(result_csv_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 전체 분석 결과가 {result_csv_path} 에 저장되었습니다.")

    print("\n\n--- 📊 발견된 강력한 '의미' 규칙 (유사도 Top 10) ---")
    sim_sorted = summary[summary['count'] > 10].sort_values(by='mean_similarity', ascending=False)
    print(sim_sorted.head(10).to_string(index=False, float_format='{:.4f}'.format))

    print("\n\n--- 📊 발견된 강력한 '공간' 규칙 (일관성 Top 10) ---")
    summary['consistency'] = (summary['std_dx'].fillna(1) + 0.001) + (summary['std_dy'].fillna(1) + 0.001)
    summary_sorted = summary.sort_values(by=['consistency', 'count'], ascending=[True, False])
    print(summary_sorted.head(10).to_string(index=False, float_format='{:.4f}'.format))


if __name__ == "__main__":
    main()