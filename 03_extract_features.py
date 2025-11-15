import json
import re
from pathlib import Path
import pickle  # [추가] .pkl 파일을 읽기 위해
from tqdm.auto import tqdm # [추가] 프로그레스 바

# --- 1. [수정] config.py와 utils.py 임포트 ---
import config
import utils

# --- 2. [수정] config.py에서 경로 로드 ---
# [입력 1] VLM 샘플 인덱스
SAMPLE_INDEX_PATH = config.OUTPUT_DIR / "sample_index_20percent.pkl" 
# [입력 2] OCR이 완료된 JSON 폴더
INPUT_DIR_BASE = config.OCR_JSON_DIR
# [출력] 특징을 추가할 JSON 폴더
OUTPUT_DIR_BASE = config.FEATURES_JSON_DIR

# --- (calc_text_features, get_annotations 함수는 utils.py로 이동하여 삭제) ---

def process_file(json_path, output_path):
    """
    JSON 파일을 읽고, 각 annotation에 텍스트 특징을 추가하여 저장합니다.
    (utils.py의 함수들을 호출)
    """
    try:
        # [수정] utils.py 함수 사용
        data = utils.load_json(json_path)
        annotations = utils.get_annotations(data)
        
        if not annotations:
            # 주석: print(f"  [경고] {json_path.name}: Annotation 리스트 없음")
            return

        # 3. 각 annotation을 순회하며 특징 추가
        for item in annotations:
            if 'text' not in item:
                item['text'] = ""
            
            # [수정] utils.py 함수 사용
            text_features = utils.calc_text_features(item['text'])
            item.update(text_features) # 딕셔너리에 특징들 추가

        # [수정] utils.py 함수 사용
        utils.save_json(data, output_path)

    except Exception as e:
        # [수정] 심각한 오류만 출력
        print(f"\n  [오류] 파일 처리 중 문제 발생 ({json_path.name}): {e}")


# --- [수정] 스크립트 실행 (main) ---
if __name__ == "__main__":
    print("스크립트 시작: (03_extract_features.py - 샘플링된 파일만 처리)")
    
    # --- 1. .pkl 파일 로드 ---
    if not SAMPLE_INDEX_PATH.exists():
        print(f"[치명적 오류] 샘플 인덱스 파일을 찾을 수 없습니다: {SAMPLE_INDEX_PATH}")
        print("  먼저 '06_create_ml_dataset.py'를 실행해야 합니다.")
        exit()
        
    print(f"샘플 인덱스 로드 중: {SAMPLE_INDEX_PATH}")
    with open(SAMPLE_INDEX_PATH, 'rb') as f:
        sampled_data = pickle.load(f)

    # --- 2. pkl 파일 목록을 기반으로 [입력/출력] 경로 리스트 생성 ---
    files_to_process = [] # (input_path, output_path) 튜플 리스트
    
    for split in ['train', 'valid']:
        for doc_type in ['report', 'press']:
            file_list = sampled_data.get(split, {}).get(doc_type, [])
            for file_info in file_list:
                pkl_path = Path(file_info['json_path'])
                doc_type_folder = pkl_path.parent.name # 예: "report_json"
                file_name = pkl_path.name
                
                # [입력 경로] (02_run_ocr.py의 출력물)
                input_path = INPUT_DIR_BASE / doc_type_folder / file_name
                
                # [출력 경로] (이 스크립트의 출력물)
                output_dir = OUTPUT_DIR_BASE / doc_type_folder
                output_path = output_dir / file_name
                
                # [수정] 입력 파일(OCR 결과)이 실제로 있는지 확인
                if input_path.exists():
                    files_to_process.append((input_path, output_path))
                else:
                    print(f"\n  [경고] OCR 입력 파일을 찾을 수 없습니다: {input_path}")
                    print("    '02_run_ocr.py'를 먼저 실행해야 합니다.")

    if not files_to_process:
        print("[경고] 처리할 샘플링된 OCR 파일이 없습니다.")
        exit()

    # --- 3. [수정] 특징 추출 실행 (tqdm 프로그레스 바 사용) ---
    print(f"총 {len(files_to_process)}개의 샘플링된 OCR 파일에 특징(feature)을 추가합니다...")
    
    for input_path, output_path in tqdm(files_to_process, desc="Extracting features"):
            # 출력 폴더 생성
        output_path.parent.mkdir(exist_ok=True, parents=True)
            
            # [수정] process_file은 심각한 오류가 아니면 print하지 않음
        process_file(input_path, output_path)

    print("\n--- 모든 작업 완료 ---")
    print(f"새로운 특징(feature)이 추가된 JSON 파일이 '{OUTPUT_DIR_BASE}' 폴더에 저장되었습니다.")