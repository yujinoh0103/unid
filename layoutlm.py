# ===================================
# LayoutLM 기반 Document VQA 구현
# ===================================

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm
import pickle

# ===================================
# 1. Config 설정
# ===================================
class Config:
    BASE_DIR = Path(__file__).parent
    TRAIN_DIR = BASE_DIR / "train_valid/train"
    VALID_DIR = BASE_DIR / "train_valid/valid"
    OUTPUT_DIR = BASE_DIR / "output"
    
    # PKL 파일 경로
    PKL_PATH = OUTPUT_DIR / "sample_index_20percent.pkl"
    
    # 모델 설정
    MODEL_NAME = "microsoft/layoutlmv3-base"
    MAX_LENGTH = 512
    BATCH_SIZE = 4
    LEARNING_RATE = 5e-5
    EPOCHS = 10
    
    # 좌표 정규화 (이미지 크기)
    IMAGE_WIDTH = 2480
    IMAGE_HEIGHT = 3508
    
    OUTPUT_DIR.mkdir(exist_ok=True)

# ===================================
# 2. 데이터셋 클래스 (PKL 파일 기반)
# ===================================
class DocumentVQADataset(Dataset):
    def __init__(self, pkl_path: Path, split: str, processor, is_train=True):
        """
        Args:
            pkl_path: sample_index_20percent.pkl 경로
            split: 'train' or 'valid'
            processor: LayoutLMv3Processor
            is_train: True면 labels 포함
        """
        self.processor = processor
        self.is_train = is_train
        self.split = split
        
        # PKL 파일 로드
        with open(pkl_path, 'rb') as f:
            pkl_data = pickle.load(f)
        
        # split에 해당하는 데이터 가져오기
        self.file_list = pkl_data[split]['report'] + pkl_data[split]['press']
        print(f"✅ Loaded {len(self.file_list)} files for {split}")
        
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict]:
        """PKL에 지정된 JSON 파일들만 로드"""
        data_list = []
        
        for file_info in tqdm(self.file_list, desc=f"Loading {self.split} data"):
            json_path = Path(file_info['json_path'])
            img_path = Path(file_info['img_path'])
            
            # JSON 로드
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"⚠️ Error loading {json_path.name}: {e}")
                continue
            
            # 이미지 파일 존재 확인
            if not img_path.exists():
                print(f"⚠️ Image not found: {img_path}")
                continue
            
            learning_info = data.get('learning_data_info', {})
            visual_context = learning_info.get('visual_context', '')
            
            # 모든 annotation 수집 (레이아웃 정보)
            annotations = learning_info.get('annotation', [])
            all_boxes = []
            all_labels = []
            
            for anno in annotations:
                bbox = anno.get('bounding_box', [])
                if len(bbox) == 4:
                    all_boxes.append(bbox)
                    all_labels.append(anno.get('class_id', 'UNK'))
            
            # V01(표) 클래스만 타겟으로 추출
            for anno in annotations:
                if anno.get('class_id') == 'V01':  # 표/차트
                    visual_instruction = anno.get('visual_instruction', '')
                    bbox = anno.get('bounding_box', [])
                    
                    # validation
                    if not visual_instruction:
                        continue
                    
                    if len(bbox) != 4:
                        continue
                    
                    data_list.append({
                        'image_path': str(img_path),
                        'question': visual_instruction,
                        'context': visual_context,
                        'bbox': bbox,  # [x, y, w, h] - 타겟
                        'all_boxes': all_boxes,  # 전체 레이아웃 정보
                        'all_labels': all_labels,
                        'instance_id': anno['instance_id']
                    })
        
        print(f"✅ Total {len(data_list)} VQA samples for {self.split}")
        return data_list
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 이미지 로드
        image = Image.open(item['image_path']).convert('RGB')
        
        # 질문
        question = item['question']
        
        # 전체 레이아웃의 bbox 정보 사용
        all_boxes = item['all_boxes']
        
        # 각 박스에 대해 간단한 텍스트 레이블 생성 (class_id)
        words = []
        boxes_normalized = []
        
        for i, (bbox, label) in enumerate(zip(all_boxes, item['all_labels'])):
            # bbox를 LayoutLM 형식(0-1000)으로 정규화
            x0 = int((bbox[0] / Config.IMAGE_WIDTH) * 1000)
            y0 = int((bbox[1] / Config.IMAGE_HEIGHT) * 1000)
            x1 = int(((bbox[0] + bbox[2]) / Config.IMAGE_WIDTH) * 1000)
            y1 = int(((bbox[1] + bbox[3]) / Config.IMAGE_HEIGHT) * 1000)
            
            # 좌표 클리핑
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(1000, x1), min(1000, y1)
            
            # 레이블을 단어로 추가
            words.append(label)
            boxes_normalized.append([x0, y0, x1, y1])
        
        # 질문을 맨 앞에 추가
        question_words = question.split()[:50]  # 질문은 최대 50단어
        for word in question_words:
            words.insert(0, word)
            boxes_normalized.insert(0, [0, 0, 1000, 1000])  # 질문은 전체 영역
        
        # MAX_LENGTH 제한
        words = words[:Config.MAX_LENGTH - 2]
        boxes_normalized = boxes_normalized[:Config.MAX_LENGTH - 2]
        
        # Processor로 인코딩
        encoding = self.processor(
            image,
            text=words,
            boxes=boxes_normalized,
            padding='max_length',
            truncation=True,
            max_length=Config.MAX_LENGTH,
            return_tensors='pt'
        )
        
        # tensor의 첫 번째 차원 제거
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        if self.is_train:
            # 타겟 Bounding box를 정규화 (0~1 범위)
            bbox = item['bbox']
            normalized_bbox = torch.tensor([
                bbox[0] / Config.IMAGE_WIDTH,   # x
                bbox[1] / Config.IMAGE_HEIGHT,  # y
                bbox[2] / Config.IMAGE_WIDTH,   # w
                bbox[3] / Config.IMAGE_HEIGHT   # h
            ], dtype=torch.float32)
            
            encoding['labels'] = normalized_bbox
        
        encoding['instance_id'] = item['instance_id']
        
        return encoding

# ===================================
# 3. 모델 정의 (BBox Regression)
# ===================================
class LayoutLMv3ForBBoxRegression(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.layoutlm = LayoutLMv3ForTokenClassification.from_pretrained(
            model_name,
            num_labels=4  # x, y, w, h
        )
        # CLS token의 출력을 bbox 예측에 사용
        self.bbox_head = nn.Linear(self.layoutlm.config.hidden_size, 4)
    
    def forward(self, input_ids, attention_mask, bbox=None, pixel_values=None, labels=None):
        # LayoutLMv3의 hidden states 추출
        outputs = self.layoutlm.layoutlmv3(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values
        )
        
        # [CLS] token의 representation 사용
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
        
        # Bounding box 예측
        bbox_pred = self.bbox_head(cls_output)  # (batch_size, 4)
        
        loss = None
        if labels is not None:
            # MSE Loss
            loss_fct = nn.MSELoss()
            loss = loss_fct(bbox_pred, labels)
        
        return {'loss': loss, 'bbox_pred': bbox_pred}

# ===================================
# 4. 학습 함수
# ===================================
def train_model(model, train_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        optimizer.zero_grad()
        
        # 데이터를 device로 이동
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        bbox = batch['bbox'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values,
            labels=labels
        )
        
        loss = outputs['loss']
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss

# ===================================
# 5. 예측 함수
# ===================================
def predict(model, test_loader, device):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            bbox = batch['bbox'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                pixel_values=pixel_values
            )
            
            bbox_pred = outputs['bbox_pred'].cpu()
            
            # 역정규화
            bbox_pred[:, 0] *= Config.IMAGE_WIDTH   # x
            bbox_pred[:, 1] *= Config.IMAGE_HEIGHT  # y
            bbox_pred[:, 2] *= Config.IMAGE_WIDTH   # w
            bbox_pred[:, 3] *= Config.IMAGE_HEIGHT  # h
            
            for i, instance_id in enumerate(batch['instance_id']):
                predictions.append({
                    'query_id': instance_id,
                    'pred_x': bbox_pred[i, 0].item(),
                    'pred_y': bbox_pred[i, 1].item(),
                    'pred_w': bbox_pred[i, 2].item(),
                    'pred_h': bbox_pred[i, 3].item()
                })
    
    return predictions

# ===================================
# 6. Main 실행
# ===================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Processor 로드 (OCR 비활성화)
    processor = LayoutLMv3Processor.from_pretrained(Config.MODEL_NAME, apply_ocr=False)
    
    # 데이터셋 생성 (PKL 기반)
    print("Loading datasets from PKL...")
    train_dataset = DocumentVQADataset(
        Config.PKL_PATH,
        split='train',
        processor=processor,
        is_train=True
    )
    
    valid_dataset = DocumentVQADataset(
        Config.PKL_PATH,
        split='valid',
        processor=processor,
        is_train=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(valid_dataset)}")
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )
    
    # 모델 초기화
    print("Initializing model...")
    model = LayoutLMv3ForBBoxRegression(Config.MODEL_NAME).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    
    # 학습
    print("Starting training...")
    for epoch in range(1, Config.EPOCHS + 1):
        avg_loss = train_model(model, train_loader, optimizer, device, epoch)
        print(f"Epoch {epoch}/{Config.EPOCHS} - Average Loss: {avg_loss:.4f}")
        
        # 모델 저장
        if epoch % 2 == 0:
            torch.save(
                model.state_dict(), 
                Config.OUTPUT_DIR / f"model_epoch_{epoch}.pt"
            )
    
    # 최종 모델 저장
    torch.save(model.state_dict(), Config.OUTPUT_DIR / "model_final.pt")
    print("Training completed!")

# ===================================
# 7. 테스트 데이터 예측
# ===================================
def test_and_submit():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Processor & Model 로드 (OCR 비활성화)
    processor = LayoutLMv3Processor.from_pretrained(Config.MODEL_NAME, apply_ocr=False)
    model = LayoutLMv3ForBBoxRegression(Config.MODEL_NAME).to(device)
    model.load_state_dict(torch.load(Config.OUTPUT_DIR / "model_final.pt"))
    
    # 테스트 데이터셋 (is_train=False)
    test_dataset = DocumentVQADataset(
        Config.TRAIN_DIR / "test_json",  # 테스트 경로로 수정 필요
        Config.TRAIN_DIR / "test_jpg",
        processor,
        is_train=False
    )
    
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE)
    
    # 예측
    predictions = predict(model, test_loader, device)
    
    # Submission 파일 생성
    df = pd.DataFrame(predictions)
    # query_id에서 query_text 추출 필요 (테스트 데이터에서 가져와야 함)
    df['query_text'] = ''  # 실제로는 테스트 데이터에서 매핑 필요
    
    df = df[['query_id', 'query_text', 'pred_x', 'pred_y', 'pred_w', 'pred_h']]
    df.to_csv(Config.OUTPUT_DIR / 'submission.csv', index=False)
    print("Submission file created!")

if __name__ == "__main__":
    main()
    # test_and_submit()  # 테스트 시 주석 해제