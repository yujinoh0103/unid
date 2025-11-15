import os
import pickle
import json
from pathlib import Path
from tqdm.auto import tqdm
import random
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# --- [수정] config.py 임포트 ---
import config

# --- [수정] config.py의 경로 설정 사용 ---
# Sampler가 탐색을 시작할 최상위 경로
DATA_ROOT = config.BASE_DIR 
# 학습/로그 경로는 config.py의 출력 폴더 하위에 생성
CHECKPOINT_DIR = config.OUTPUT_DIR / "checkpoints"
LOG_DIR = config.OUTPUT_DIR / "logs"
SUBMISSION_DIR = config.OUTPUT_DIR / "submissions"
# VLM 학습용 데이터셋(pkl)을 저장할 위치
SAMPLE_INDEX_PATH = config.OUTPUT_DIR / "sample_index_20percent.pkl"


# 디렉토리 생성
for dir_path in [CHECKPOINT_DIR, LOG_DIR, SUBMISSION_DIR]:
    os.makedirs(dir_path, exist_ok=True)

print("📁 경로 설정 완료 (config.py 기반):")
print(f"   데이터 루트: {DATA_ROOT}")
print(f"   인덱스 (출력): {SAMPLE_INDEX_PATH}")

# %%
# ==========================================
# 최적화된 SmartSampler (config.py 경로 사용)
# ==========================================

class OptimizedSmartSampler:
    """최적화된 Smart Sampler - 실제 구조에 맞춤"""

    def __init__(self, data_root, sample_ratio=0.2, seed=42):
        self.data_root = Path(data_root)
        self.sample_ratio = sample_ratio
        self.seed = seed
        random.seed(seed)

        self.sampled_files = {
            'train': {},
            'valid': {}
        }

    def create_sample_index(self, force_recreate=False):
        """샘플 인덱스 생성 - 실제 폴더 구조에 최적화"""

        # 기존 인덱스 확인
        if os.path.exists(SAMPLE_INDEX_PATH) and not force_recreate:
            print("✅ 기존 샘플 인덱스를 로드합니다...")
            with open(SAMPLE_INDEX_PATH, 'rb') as f:
                self.sampled_files = pickle.load(f)
            self._print_statistics()
            return self.sampled_files

        print(f"\n🔄 새로운 {self.sample_ratio*100:.0f}% 샘플 인덱스 생성 중...")

        # train과 valid 처리
        for split in ['train', 'valid']:
            # [수정] config.BASE_DIR / 'train_valid' / 'train' (or 'valid')
            split_path = self.data_root / 'train_valid' / split 

            if not split_path.exists():
                print(f"❌ {split_path} 경로가 없습니다. (config.py의 BASE_DIR 확인)")
                continue

            print(f"\n📁 {split} 처리 중...")

            # report와 press 각각 처리
            for doc_type in ['report', 'press']:
                json_dir = split_path / f'{doc_type}_json'
                img_dir = split_path / f'{doc_type}_jpg'

                if not json_dir.exists():
                    print(f"   ⚠️ {doc_type}_json 폴더 없음")
                    continue

                if not img_dir.exists():
                    print(f"   ⚠️ {doc_type}_jpg 폴더 없음")
                    continue

                # JSON 파일 목록
                json_files = list(json_dir.glob('*.json'))
                print(f"   {doc_type}: {len(json_files)} JSON files found")

                if not json_files:
                    continue

                # 유효한 파일 쌍 찾기
                valid_pairs = []
                invalid_count = 0

                # 샘플링할 파일 수 계산
                sample_size = max(1, int(len(json_files) * self.sample_ratio))

                # 먼저 샘플링
                sampled_json_files = random.sample(json_files, min(sample_size, len(json_files)))

                print(f"   Validating {len(sampled_json_files)} sampled files...")
                for json_path in tqdm(sampled_json_files, desc=f"      {doc_type}", leave=False):
                    try:
                        # JSON 파일 검증
                        with open(json_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)

                        # VQA annotation 확인
                        annotations = data.get('learning_data_info', {}).get('annotation', [])

                        # visual_instruction과 bounding_box가 있는 annotation 찾기
                        valid_annotations = [
                            ann for ann in annotations
                            if 'visual_instruction' in ann
                            and ann['visual_instruction']
                            and 'bounding_box' in ann
                            and len(ann.get('bounding_box', [])) == 4
                            and all(b >= 0 for b in ann['bounding_box'][:2])  # x, y >= 0
                            and all(b > 0 for b in ann['bounding_box'][2:])   # w, h > 0
                        ]

                        if not valid_annotations:
                            invalid_count += 1
                            continue

                        # 이미지 파일 찾기
                        img_name = json_path.stem.replace('MI3_', 'MI2_') + '.jpg'
                        img_path = img_dir / img_name

                        if img_path.exists():
                            valid_pairs.append({
                                'json_path': str(json_path),
                                'img_path': str(img_path),
                                'num_annotations': len(valid_annotations)
                            })
                        else:
                            invalid_count += 1

                    except Exception as e:
                        invalid_count += 1
                        continue

                # 결과 저장
                if valid_pairs:
                    self.sampled_files[split][doc_type] = valid_pairs
                    print(f"     ✅ {doc_type}: {len(valid_pairs)} valid pairs (invalid: {invalid_count})")
                else:
                    print(f"     ❌ {doc_type}: 유효한 파일 쌍 없음")

        # 인덱스 저장
        with open(SAMPLE_INDEX_PATH, 'wb') as f:
            pickle.dump(self.sampled_files, f)

        print(f"\n✅ 샘플 인덱스 저장 완료: {SAMPLE_INDEX_PATH}")
        self._print_statistics()

        return self.sampled_files

    def _print_statistics(self):
        """샘플 통계 출력"""
        print("\n📊 샘플 인덱스 통계:")
        print("-" * 40)

        total_files = 0
        total_annotations = 0

        for split in ['train', 'valid']:
            split_total = 0
            split_annotations = 0

            for doc_type in ['report', 'press']:
                files = self.sampled_files.get(split, {}).get(doc_type, [])
                if files:
                    num_files = len(files)
                    num_annotations = sum(f.get('num_annotations', 0) for f in files)
                    split_total += num_files
                    split_annotations += num_annotations
                    print(f"   {split}/{doc_type}: {num_files} files, {num_annotations} annotations")

            if split_total > 0:
                print(f"   {split} 총합: {split_total} files, {split_annotations} annotations")
                total_files += split_total
                total_annotations += split_annotations

        print("-" * 40)
        print(f"   전체: {total_files} files, {total_annotations} annotations")

# %%
# Sampler 실행
if __name__ == "__main__":
    print("🚀 Optimized Smart Sampler 실행...")
    # [수정] config.BASE_DIR을 data_root로 전달
    sampler = OptimizedSmartSampler(config.BASE_DIR, sample_ratio=0.2, seed=42)

    # 새로 생성 (force_recreate=True) 또는 기존 사용 (force_recreate=False)
    sampled_files = sampler.create_sample_index(force_recreate=True)