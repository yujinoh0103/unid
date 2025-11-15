import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import permutations, product
import pickle

# --- 1. [수정] config.py와 utils.py 임포트 ---
import config
import utils

# --- 2. [수정] config.py에서 경로 로드 ---
SAMPLE_INDEX_PATH = config.OUTPUT_DIR / "sample_index_20percent.pkl" 
OUTPUT_DIR = config.ANALYSIS_RESULT_DIR 

# --- 3. [유지] 이 스크립트에서만 사용하는 함수 ---
def find_all_class_bboxes(annotations, class_name):
    """
    JSON의 annotation 리스트에서 특정 class_name의 모든 bbox 리스트를 반환합니다.
    """
    bboxes = []
    for item in annotations:
        if item.get("class_name") == class_name:
            bboxes.append(item["bounding_box"])
    return bboxes

# --- (get_centroid, get_annotations, setup_korean_font 함수는 utils.py로 이동하여 삭제) ---

def main():
    # --- [수정] utils.py의 함수 호출 ---
    utils.setup_korean_font() 
    
    # --- 4. .pkl 파일 로드 (이전과 동일) ---
    print(f"샘플 인덱스 파일 로드: {SAMPLE_INDEX_PATH}")
    if not SAMPLE_INDEX_PATH.exists():
        print(f"[치명적 오류] 샘플 인덱스 파일을 찾을 수 없습니다: {SAMPLE_INDEX_PATH}")
        print("  먼저 '06_create_ml_dataset.py'를 실행해야 합니다.")
        return

    with open(SAMPLE_INDEX_PATH, 'rb') as f:
        sampled_data = pickle.load(f)
    
    all_json_files = []
    for split in ['train', 'valid']:
        for doc_type in ['report', 'press']:
            files_list = sampled_data.get(split, {}).get(doc_type, [])
            for file_info in files_list:
                all_json_files.append(Path(file_info['json_path']))
    
    if not all_json_files:
        print("[오류] 샘플 인덱스에 분석할 JSON 파일이 없습니다.")
        return

    # --- 5. 샘플링된 JSON 스캔 (class_name 수집) ---
    all_class_names = set()
    print(f"총 {len(all_json_files)}개의 '샘플링된' JSON 파일에서 모든 클래스 이름을 수집합니다...")
    
    for json_file in all_json_files:
        try:
            # --- [수정] utils.py의 함수 호출 ---
            data = utils.load_json(json_file) 
            annotations = utils.get_annotations(data) 
            
            for item in annotations:
                if "class_name" in item:
                    all_class_names.add(item["class_name"])
        except Exception as e:
            print(f"  [경고] {json_file.name} 로드 실패: {e}")
            pass 

    if not all_class_names:
        print("[오류] 어떤 class_name도 찾지 못했습니다.")
        return
        
    print(f"--- 발견된 전체 클래스 (총 {len(all_class_names)}개) ---")
    print(all_class_names)
    
    # --- 6. 모든 쌍 생성 (이전과 동일) ---
    all_pairs = list(permutations(all_class_names, 2))
    print(f"\n총 {len(all_pairs)}개의 모든 위치 관계 조합을 분석합니다...")

    # --- 7. 오프셋 계산 ---
    all_offsets_data = [] 
    
    for json_file in all_json_files:
        source_type = json_file.parent.name 
        try:
            # --- [수정] utils.py의 함수 호출 ---
            data = utils.load_json(json_file)
            annotations = utils.get_annotations(data)
            
            resolution = data.get("source_data_info", {}).get("document_resolution")

            if not annotations or not resolution:
                continue
            
            page_width, page_height = resolution

            for class_a, class_b in all_pairs:
                bboxes_a = find_all_class_bboxes(annotations, class_a)
                bboxes_b = find_all_class_bboxes(annotations, class_b)

                if bboxes_a and bboxes_b:
                    for bbox_a, bbox_b in product(bboxes_a, bboxes_b):
                        # --- [수정] utils.py의 함수 호출 ---
                        cx_a, cy_a = utils.get_centroid(bbox_a)
                        cx_b, cy_b = utils.get_centroid(bbox_b)
                        
                        norm_dx = (cx_b - cx_a) / page_width
                        norm_dy = (cy_b - cy_a) / page_height
                        all_offsets_data.append((class_a, class_b, norm_dx, norm_dy, source_type))
        except Exception:
            continue 

    if not all_offsets_data:
        print("[분석 결과] 유효한 관계 쌍을 단 하나도 찾지 못했습니다.")
        return

    # --- 8. Pandas 통계 집계 (이전과 동일) ---
    df = pd.DataFrame(all_offsets_data, columns=['class_A', 'class_B', 'dx', 'dy', 'source'])

    summary = df.groupby(['class_A', 'class_B']).agg(
        count=('dx', 'size'),
        mean_dx=('dx', 'mean'),
        std_dx=('dx', 'std'),
        mean_dy=('dy', 'mean'),
        std_dy=('dy', 'std')
    ).reset_index()

    summary['consistency_score'] = (summary['std_dx'].fillna(1) + 0.0001) + (summary['std_dy'].fillna(1) + 0.0001)
    summary_sorted = summary.sort_values(by=['consistency_score', 'count'], ascending=[True, False])

    print("\n\n--- 📊 발견된 강력한 위치 규칙 (일관성 Top 10) ---")
    print("(std가 낮을수록, count가 높을수록 일관된 '규칙'입니다)")
    print(summary_sorted.head(10).to_string(index=False, float_format='{:.4f}'.format))

    # --- 9. 시각화 (이전과 동일, 경로만 config 사용) ---
    print(f"\n전체 관계에 대한 히트맵을 생성하여 '{OUTPUT_DIR}'에 저장합니다...")
    
    dy_heatmap_path = OUTPUT_DIR / "spatial_heatmap_delta_y.png"
    dx_heatmap_path = OUTPUT_DIR / "spatial_heatmap_delta_x.png"

    try:
        pivot_dy = summary.pivot(index='class_A', columns='class_B', values='mean_dy')
        plt.figure(figsize=(max(12, len(all_class_names)*0.5), max(10, len(all_class_names)*0.4)))
        sns.heatmap(pivot_dy, annot=True, cmap="coolwarm", fmt=".2f",
                    annot_kws={"size": 8})
        plt.title("평균 세로 오프셋 (Delta Y)\n(값이 -면 B가 A의 '위', +면 '아래'에 있다는 의미)")
        plt.xlabel("Class B (대상)")
        plt.ylabel("Class A (기준)")
        plt.tight_layout()
        plt.savefig(dy_heatmap_path) 
        plt.show() 
        print(f"✔ Delta Y 히트맵 저장 완료: {dy_heatmap_path}")
    except Exception as e:
        print(f"[시각화 오류 1] 세로 히트맵 생성 실패: {e}")

    try:
        pivot_dx = summary.pivot(index='class_A', columns='class_B', values='mean_dx')
        plt.figure(figsize=(max(12, len(all_class_names)*0.5), max(10, len(all_class_names)*0.4)))
        sns.heatmap(pivot_dx, annot=True, cmap="coolwarm", fmt=".2f",
                    annot_kws={"size": 8})
        plt.title("평균 가로 오프셋 (Delta X)\n(값이 -면 B가 A의 '왼쪽', +면 '오른쪽'에 있다는 의미)")
        plt.xlabel("Class B (대상)")
        plt.ylabel("Class A (기준)")
        plt.tight_layout()
        plt.savefig(dx_heatmap_path)
        plt.show()
        print(f"✔ Delta X 히트맵 저장 완료: {dx_heatmap_path}")
    except Exception as e:
        print(f"[시각화 오류 2] 가로 히트맵 생성 실패: {e}")

if __name__ == "__main__":
    main()