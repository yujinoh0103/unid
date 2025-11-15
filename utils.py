# utils.py
import json
import re
import matplotlib.pyplot as plt
from matplotlib import rc
from pathlib import Path

# --- JSON 처리 ---

def load_json(json_path):
    """JSON 파일을 읽어 딕셔너리로 반환"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, json_path):
    """딕셔너리를 JSON 파일로 저장"""
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_annotations(data):
    """
    다양한 JSON 구조를 탐색하여 'annotation' 리스트를 반환
    (graph.py, ocr.py, feature_extractor.py 등에서 공통 사용)
    """
    # 1. 최상위 레벨에 'annotation'이 있는지 확인
    annotations = data.get("annotation")
    if annotations is not None:
        return annotations
    
    # 2. 'learning_data_info' 안에 있는지 확인
    learning_info = data.get("learning_data_info")
    if learning_info and isinstance(learning_info, dict):
        annotations = learning_info.get("annotation")
        if annotations is not None:
            return annotations
            
    # 둘 다 없으면 빈 리스트 반환
    return []

# --- 좌표계산 ---

def get_centroid(bbox):
    """
    [x, y, w, h] 형식의 bbox 중심점을 반환
    (bbox_analyzer.py, bbox_analyzer_advanced.py 공통 사용)
    """
    x, y, w, h = bbox
    return x + (w / 2), y + (h / 2)

# --- 텍스트 특징 추출 ---

def calc_text_features(text):
    """
    입력된 텍스트를 분석하여 특징 딕셔너리를 반환
    (feature_extractor.py의 핵심 로직)
    """
    if not isinstance(text, str):
        text = "" 
    stripped_text = text.strip()
    
    feat_text_length = len(stripped_text)
    
    list_starters = ['*', '-', '•', '○', '□', '▶', '※']
    feat_startsWithListSymbol = False
    if stripped_text:
        if any(stripped_text.startswith(s) for s in list_starters):
            feat_startsWithListSymbol = True
        elif re.match(r'^(\d+\.|[가-힣]\.|\(\d+\)|\([가-힣]\))', stripped_text):
            feat_startsWithListSymbol = True
            
    ref_symbols = ['※', '☞', '★']
    feat_containsRefSymbol = any(s in text for s in ref_symbols)
    
    feat_isPageNumberFormat = False
    if 0 < feat_text_length < 10:
        if re.fullmatch(r'[-–—\s]*\d+[-–—\s]*', stripped_text):
            feat_isPageNumberFormat = True
            
    table_keywords = ['표', '차트', '그림', '다음은', '아래는', '(단위:']
    feat_containsTableKeyword = any(k in text for k in table_keywords)
    
    return {
        "feat_text_length": feat_text_length,
        "feat_startsWithListSymbol": feat_startsWithListSymbol,
        "feat_containsRefSymbol": feat_containsRefSymbol,
        "feat_isPageNumberFormat": feat_isPageNumberFormat,
        "feat_containsTableKeyword": feat_containsTableKeyword
    }

# --- 시각화 ---

def setup_korean_font():
    rc('font', family='NanumGothic') # Windows
    plt.rcParams['axes.unicode_minus'] = False