import pickle
from pathlib import Path

# PKL 파일 로드
pkl_path = Path("output/sample_index_20percent.pkl")

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

print("PKL 파일 구조:")
print("="*60)

for split in ['train', 'valid']:
    print(f"\n[{split.upper()}]")
    for doc_type in ['report', 'press']:
        content = data[split][doc_type]
        print(f"  {doc_type}:")
        print(f"    타입: {type(content)}")
        print(f"    길이: {len(content) if hasattr(content, '__len__') else 'N/A'}")
        
        if isinstance(content, (list, set, tuple)):
            print(f"    첫 10개 샘플:")
            for i, item in enumerate(list(content)[:10]):
                print(f"      [{i}] {item}")
        elif isinstance(content, dict):
            print(f"    키: {list(content.keys())[:5]}")
            first_key = list(content.keys())[0]
            print(f"    첫 번째 값 예시 ({first_key}): {content[first_key]}")
        else:
            print(f"    내용: {content}")

print("\n" + "="*60)
print(f"총 train 샘플 수: {len(data['train']['report']) + len(data['train']['press'])}")
print(f"총 valid 샘플 수: {len(data['valid']['report']) + len(data['valid']['press'])}")