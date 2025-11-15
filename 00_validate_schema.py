import json
import pickle  # [추가] .pkl 파일을 읽기 위해
import os      # [추가]
from pathlib import Path

# --- 1. 설정: 허용되는 class_name 목록 (스키마) ---
# (원본과 동일 - config.py의 내용을 하드코딩)
ALLOWED_CLASS_NAMES = {
    "제목",
    "소제목",
    "시각요소 제목",
    "본문",
    "목록",
    "표",
    "차트(세로 막대형)",
    "차트(꺾은선형)",
    "차트(혼합형)",
    "다이어그램",
    "차트(가로 막대형)",
    "차트(원형)",
    "차트(분산형)",
    "차트(영역형)",
    "차트(방사형)",
    "발행정보",
    "머리말",
    "꼬리말",
    "페이지번호"
}

# --- 2. 경로 설정 (config.py의 구조를 따름) ---
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output" # (config.py와 동일한 구조)

# [수정] VLM 샘플러가 생성한 인덱스 파일 경로
SAMPLE_INDEX_PATH = OUTPUT_DIR / "sample_index_20percent.pkl" 
# [수정] 위반 파일 리포트 경로
OUTPUT_FILE = OUTPUT_DIR / "schema_violation_files.txt"

# --- (find_annotations 함수는 원본과 동일) ---
def find_annotations(data):
    """
    다양한 JSON 구조를 탐색하여 'annotation' 리스트를 찾습니다.
    """
    annotations = data.get("annotation")
    if annotations is not None:
        return annotations
    
    learning_info = data.get("learning_data_info")
    if learning_info and isinstance(learning_info, dict):
        annotations = learning_info.get("annotation")
        if annotations is not None:
            return annotations
            
    return []

def validate_files():
    """
    [수정] 'ml_sample_index.pkl'에 정의된 파일들만 스키마 검사를 수행합니다.
    """
    
    # --- [수정] 샘플 인덱스 파일 로드 ---
    print(f"샘플 인덱스 파일 로드: {SAMPLE_INDEX_PATH}")
    if not SAMPLE_INDEX_PATH.exists():
        print(f"[치명적 오류] 샘플 인덱스 파일을 찾을 수 없습니다: {SAMPLE_INDEX_PATH}")
        print("  먼저 '06_create_ml_dataset.py'를 실행해야 합니다.")
        return set()
        
    with open(SAMPLE_INDEX_PATH, 'rb') as f:
        sampled_data = pickle.load(f)
    
    # --- [수정] .pkl에서 모든 json_path 추출 ---
    json_paths_to_check = []
    for split in ['train', 'valid']:
        for doc_type in ['report', 'press']:
            files_list = sampled_data.get(split, {}).get(doc_type, [])
            for file_info in files_list:
                # pkl에 저장된 경로는 절대 경로일 수 있으므로 Path()로 감싸줌
                json_paths_to_check.append(Path(file_info['json_path']))
    
    print(f"총 {len(json_paths_to_check)}개의 샘플링된 JSON 파일 검사를 시작합니다...")
    print(f"허용되는 class_name: {ALLOWED_CLASS_NAMES}\n")
    
    violation_files = set()

    if not json_paths_to_check:
        print("[경고] 검사할 파일이 샘플 인덱스에 없습니다.")
        return set()

    # --- [수정] glob() 대신 pkl에서 추출한 리스트를 순회 ---
    for json_file in json_paths_to_check:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 3. JSON 데이터에서 annotation 리스트 찾기
            annotations = find_annotations(data)
            
            if not annotations:
                print(f"  [경고] {json_file.name}: 'annotation' 리스트를 찾을 수 없습니다.")
                continue

            # 4. 각 annotation 항목의 class_name 검사 (원본 로직과 동일)
            for item in annotations:
                class_name = item.get("class_name")
                
                if class_name is None:
                    # class_name 키 자체가 없는 경우
                    print(f"  [오류] {json_file.name}: 'class_name' 키가 없는 항목 발견.")
                    violation_files.add(json_file.name)
                    break # 이 파일은 더 검사할 필요 없음
                
                # 5. class_name이 허용 목록에 없는지 확인 (핵심)
                if class_name not in ALLOWED_CLASS_NAMES:
                    print(f"  [오류] {json_file.name}: 허용되지 않는 class_name '{class_name}' 발견.")
                    violation_files.add(json_file.name)
                    break # 이 파일은 더 검사할 필요 없음
        
        except json.JSONDecodeError:
            print(f"  [오류] {json_file.name}: JSON 형식이 올바르지 않습니다.")
            violation_files.add(json_file.name)
        except FileNotFoundError:
            print(f"  [오류] {json_file.name}: 파일을 찾을 수 없습니다 (경로: {json_file})")
            violation_files.add(f"{json_file.name} (File Not Found)")
        except Exception as e:
            print(f"  [오류] {json_file.name} 처리 중 알 수 없는 오류: {e}")
            violation_files.add(json_file.name)

    return violation_files

# --- 스크립트 실행 (원본과 거의 동일) ---
if __name__ == "__main__":
    invalid_files = validate_files()
    
    print("\n--- 📊 검사 완료 ---")
    
    if not invalid_files:
        print("🎉 모든 JSON 파일이 스키마를 준수합니다!")
    else:
        print(f"총 {len(invalid_files)}개의 파일에서 스키마 위반이 발견되었습니다.")
        print(f"자세한 내용은 '{OUTPUT_FILE}' 파일에 저장됩니다.")
        
        # 6. 오류 파일 목록을 .txt 파일로 저장
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write("--- 스키마 위반 파일 목록 ---\n")
            for file_name in sorted(list(invalid_files)):
                f.write(f"{file_name}\n")
    
    print("-------------------")