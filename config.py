# config.py
from pathlib import Path

# --- 기본 경로 ---
BASE_DIR = Path(__file__).parent
TRAIN_DIR = BASE_DIR / "train_valid/train"
OUTPUT_DIR = BASE_DIR / "output" # 모든 출력물을 한 곳에 모음

# --- 1. 원본 데이터 경로 ---
REPORT_JSON_DIR = TRAIN_DIR / "report_json"
REPORT_JPG_DIR = TRAIN_DIR / "report_jpg"
PRESS_JSON_DIR = TRAIN_DIR / "press_json"
PRESS_JPG_DIR = TRAIN_DIR / "press_jpg"

# --- 2. 파이프라인 출력 경로 ---
ANNOTATED_IMAGE_DIR = OUTPUT_DIR / "01_annotated_images"
OCR_JSON_DIR = OUTPUT_DIR / "02_output_json_easyocr"
FEATURES_JSON_DIR = OUTPUT_DIR / "03_output_json_features"
ANALYSIS_RESULT_DIR = OUTPUT_DIR / "04_analysis_results"

# --- 3. 모델 및 스키마 ---
CROSS_ENCODER_MODEL_NAME = 'bongsoo/albert-small-kor-cross-encoder-v1'

# 허용되는 class_name 목록 (스키마)
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
# --- 4. 디렉터리 생성 ---
# 스크립트 실행 전 필요한 폴더를 미리 생성
for path in [ANNOTATED_IMAGE_DIR, OCR_JSON_DIR, FEATURES_JSON_DIR, ANALYSIS_RESULT_DIR]:
    path.mkdir(parents=True, exist_ok=True)