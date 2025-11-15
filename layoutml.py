import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LayoutLMv3Processor 
from PIL import Image
from pathlib import Path
import json
import pickle
import numpy as np

# utils.py의 함수들 사용을 가정합니다.
import utils 
import config # 경로 설정을 위해

# --- 1. 모델 설정 ---
# LayoutLMv3는 Hugging Face에서 processor를 가져와야 합니다.
MODEL_NAME = "microsoft/layoutlmv3-base" # 또는 large 버전
# *사전 학습된 모델을 사용하기 위해 인터넷 연결이 필요합니다.*


class LayoutVQA_Grounding_Dataset(Dataset):
    """VQA BBox Localization을 위한 LayoutLMv3 Dataset"""
    
    def __init__(self, index_path, processor: LayoutLMv3Processor, split='train'):
        
        # 1. 샘플 인덱스 로드 (train 또는 valid)
        with open(index_path, 'rb') as f:
            full_data = pickle.load(f)
        
        # report/press의 모든 파일을 하나의 리스트로 통합
        self.samples = []
        for doc_type in ['report', 'press']:
            self.samples.extend(full_data.get(split, {}).get(doc_type, []))
        
        if not self.samples:
             raise ValueError(f"No samples found for split: {split}")

        self.processor = processor
        print(f"Dataset initialized with {len(self.samples)} samples for '{split}'.")
        
        # 2. BBox 변환을 위한 정규화 상수 (LayoutLMv3는 0~1000 정규화 사용)
        self.MAX_W, self.MAX_H = 1000, 1000 
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        json_path = Path(sample_info['json_path'])
        img_path = Path(sample_info['img_path'])
        
        # 3. 데이터 로드 및 전처리
        
        # 이미지 로드
        image = Image.open(img_path).convert("RGB")
        
        # JSON 로드 및 VQA 정답 추출
        data = utils.load_json(json_path)
        annotations = utils.get_annotations(data) 
        
        # 문서 해상도 (필요시 정규화를 위해)
        doc_width, doc_height = data.get("source_data_info", {}).get("document_resolution", [1, 1])

        # --- [핵심] 쿼리 및 정답 BBox 추출 ---
        
        # (주의: 샘플러가 유효한 어노테이션만 남겼다고 가정하지만, 여기서는 모든 어노테이션을 사용)
        # VLM Grounding 태스크는 쿼리(visual_instruction)에 해당하는 정답 BBox를 찾아야 합니다.
        # 샘플러가 저장한 '유효한 쌍' 중 첫 번째 쌍을 정답으로 사용합니다.
        
        # [수정 필요] 샘플러는 파일 단위로 저장했으므로, 
        # 파일 내 모든 VQA 쌍을 여기서 다시 찾아야 합니다.
        
        vqa_pairs = [
            ann for ann in annotations if 'visual_instruction' in ann and ann['bounding_box']
        ]
        
        if not vqa_pairs:
             raise IndexError("VQA annotation missing in file. This shouldn't happen.")
             
        # 모델은 단일 쿼리와 단일 BBox를 기대합니다. 첫 번째 쌍 사용
        query_ann = vqa_pairs[0]
        query_text = query_ann['visual_instruction']
        target_bbox = query_ann['bounding_box'] # [x, y, w, h] 형식
        
        # --- 4. BBox 정규화 (Normalization) ---
        
        # VLM Grounding은 토큰 레벨의 BBox가 아니라, 
        # 이미지 전체를 자르는 (Crop) 방식이므로 이 BBox만 정규화합니다.
        
        # [x, y, w, h] -> [x_min, y_min, x_max, y_max]
        x, y, w, h = target_bbox
        target_bbox_norm = [
            int(x / doc_width * self.MAX_W),
            int(y / doc_height * self.MAX_H),
            int((x + w) / doc_width * self.MAX_W),
            int((y + h) / doc_height * self.MAX_H)
        ]
        
        # --- 5. Processor를 통한 토큰화 및 최종 입력 생성 ---
        
        # LayoutLMv3 Processor는 이미지와 텍스트를 받아서 
        # 토큰, BBox, 픽셀 값 등을 자동으로 생성합니다.
        
        # *주의: LayoutLMv3는 VQA Grounding을 위한 구조가 복잡합니다.*
        # 여기서는 VQA 태스크의 표준 입력 방식을 따릅니다.
        
        encoding = self.processor(
            image, 
            query_text, 
            truncation=True, 
            padding="max_length", 
            max_length=512,
            return_tensors="pt"
        )
        
        # Batch 차원 제거 (Dataset이므로)
        for k, v in encoding.items():
            encoding[k] = v.squeeze()

        # VQA Grounding의 정답(Target)은 모델 외부에 별도로 준비
        encoding['bbox_target'] = torch.tensor(target_bbox_norm, dtype=torch.float)
        
        return encoding

# --- 실행 블록 (Dataset 초기화 예시) ---
if __name__ == '__main__':
    # config.py의 경로 사용 (예시)
    SAMPLE_INDEX_PATH = config.OUTPUT_DIR / "sample_index_20percent.pkl" 
    
    # [설정] LayoutLMv3 Processor 로드
    try:
        processor = LayoutLMv3Processor.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"LayoutLMv3 Processor 로드 실패. 인터넷 연결 확인: {e}")
        exit()

    # 데이터셋 초기화
    train_dataset = LayoutVQA_Grounding_Dataset(SAMPLE_INDEX_PATH, processor, split='train')
    
    # 첫 번째 항목 확인
    first_item = train_dataset[0]
    
    print("\n--- LayoutLMv3 입력 형식 확인 (Dataset[0]) ---")
    for k, v in first_item.items():
        if isinstance(v, torch.Tensor):
            print(f"{k:<15}: {v.shape}, dtype={v.dtype}")
        else:
            print(f"{k:<15}: {v}")