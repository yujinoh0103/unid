import os
import json
import cv2
import easyocr
from pathlib import Path
import torch
import pickle
from tqdm.auto import tqdm # [추가] 프로그레스 바

# --- 1. config.py와 utils.py 임포트 ---
import config
import utils

# --- 2. process_file 함수 (print문 수정) ---
def process_file(json_path, img_path, output_path, reader):
    """
    (모듈화/최적화됨) JSON과 이미지를 읽어 EasyOCR을 수행하고,
    'text' 키를 annotation에 추가하여 새 JSON 파일로 저장합니다.
    """
    # [제거] --- print(f"\n  --- 처리 시작: {json_path.name} ---") 
    
    try:
        data = utils.load_json(json_path)
        image = cv2.imread(str(img_path))
        if image is None:
            # [유지] 오류는 중요하므로 출력
            print(f"\n  [오류] 이미지를 로드할 수 없습니다: {img_path}") 
            return

        annotations = utils.get_annotations(data)
        if not annotations:
            # [유지] 경고는 중요하므로 출력
            print(f"\n  [경고] {json_path.name}: Annotation 리스트를 찾을 수 없습니다.")
            return

        for i, item in enumerate(annotations):
            class_id = item.get("class_id", f"항목{i}")
            
            if 'text' in item and item['text']: 
                continue
                
            bbox = item.get("bounding_box")
            if not bbox:
                item['text'] = ""
                continue

            try:
                x, y, w, h = map(int, [bbox[0], bbox[1], bbox[2], bbox[3]])
                x1, y1, x2, y2 = x, y, x + w, y + h
            except Exception as e:
                print(f"\n    [{class_id}] BBox 좌표 변환 오류: {bbox} -> {e}")
                item['text'] = ""
                continue
            
            if x1 >= x2 or y1 >= y2:
                # [유지] 오류는 중요하므로 출력
                print(f"\n    [{class_id}] 잘못된 BBox (x:{x}, y:{y}, w:{w}, h:{h})")
                item['text'] = ""
                continue
                
            cropped_image = image[y1:y2, x1:x2]
            
            if cropped_image.shape[0] < 5 or cropped_image.shape[1] < 5:
                item['text'] = ""
                continue

            try:
                ocr_result_list = reader.readtext(cropped_image, detail=1, paragraph=False)
                
                if not ocr_result_list:
                    # [제거] OCR 결과가 비어있는 것은 오류가 아니므로, 
                    # print(f"    [{class_id}] OCR 결과가 비어있습니다...")
                    ocr_text = ""
                else:
                    text_pieces = [result[1] for result in ocr_result_list]
                    ocr_text = " ".join(text_pieces)
                    # [제거] 성공 메시지는 너무 많으므로 제거
                    # print(f"    [{class_id}] OCR 성공: {ocr_text[:30]}...")
            
            except Exception as e:
                # [유지] 치명적 오류는 출력
                print(f"\n    [{class_id}] !!! EasyOCR 실행 중 치명적 오류: {e}")
                ocr_text = ""
            
            item['text'] = ocr_text

        utils.save_json(data, output_path)

    except Exception as e:
        print(f"\n  [전체 오류] 파일 처리 중 문제 발생 ({json_path.name}): {e}")

# --- 3. 스크립트 실행 (main) ---
if __name__ == "__main__":
    print("스크립트 시작: (02_run_ocr.py - 샘플링된 파일만 처리)")
    
    use_cuda = torch.cuda.is_available()
    print(f"EasyOCR 모델을 로드합니다... (CUDA 사용: {use_cuda})")
    try:
        reader = easyocr.Reader(['ko'], gpu=use_cuda) 
        print("EasyOCR 모델 로드 완료.")
    except Exception as e:
        print(f"[치명적 오류] EasyOCR 리더기를 초기화할 수 없습니다: {e}")
        exit()

    SAMPLE_INDEX_PATH = config.OUTPUT_DIR / "sample_index_20percent.pkl" 
    OUTPUT_DIR_BASE = config.OCR_JSON_DIR
    
    if not SAMPLE_INDEX_PATH.exists():
        print(f"[치명적 오류] 샘플 인덱스 파일을 찾을 수 없습니다: {SAMPLE_INDEX_PATH}")
        print("  먼저 '06_create_ml_dataset.py'를 실행해야 합니다.")
        exit()
        
    print(f"샘플 인덱스 로드 중: {SAMPLE_INDEX_PATH}")
    with open(SAMPLE_INDEX_PATH, 'rb') as f:
        sampled_data = pickle.load(f)

    files_to_process = []
    for split in ['train', 'valid']:
        for doc_type in ['report', 'press']:
            file_list = sampled_data.get(split, {}).get(doc_type, [])
            for file_info in file_list:
                files_to_process.append(
                    (file_info['img_path'], file_info['json_path'])
                )
    
    if not files_to_process:
        print("[경고] 샘플 인덱스에 처리할 파일이 없습니다.")
        exit()
        
    print(f"총 {len(files_to_process)}개의 샘플링된 파일에 대해 OCR을 실행합니다...")

    # --- [수정] tqdm 프로그레스 바 적용 및 이미 처리된 파일 건너뛰기 ---
    skipped_count = 0
    processed_count = 0
    
    # tqdm으로 메인 루프를 감쌉니다.
    for img_path_str, json_path_str in tqdm(files_to_process, desc="Running OCR"):
        img_path = Path(img_path_str)
        json_path = Path(json_path_str)

        doc_type_folder = json_path.parent.name
        output_json_dir = OUTPUT_DIR_BASE / doc_type_folder
        output_json_dir.mkdir(exist_ok=True, parents=True)
        
        output_path = output_json_dir / json_path.name
        
        # [수정] output_path가 이미 존재하면 process_file을 호출하지 않고 건너뜀
        if output_path.exists():
            skipped_count += 1
            continue
            
        if not img_path.exists():
            print(f"  [경고] 이미지를 찾을 수 없습니다: {img_path}")
            continue
            
        process_file(json_path, img_path, output_path, reader)
        processed_count += 1

    print("\n--- 모든 작업 완료 ---")
    print(f"결과물이 '{OUTPUT_DIR_BASE}' 폴더에 저장되었습니다.")
    print(f"  새로 처리된 파일: {processed_count}개")
    print(f"  이미 처리되어 건너뛴 파일: {skipped_count}개")