# %%
# ==========================================
# Cell 1: 환경 설정 및 초기화
# ==========================================

print("="*60)
print("🚀 Uni-DTHON VQA Ultimate Model - Smart Sampling Edition")
print("📊 Dataset: 20% sampling (NO COPY, Direct Access)")
print("🎯 Target: mIoU 0.85+ → Dacon Submission")
print("="*60)

# GPU 확인 및 최적화
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

print(f"\n🔥 PyTorch 버전: {torch.__version__}")
print(f"🔥 CUDA 사용 가능: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
    print(f"🔥 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # A100 최적화 설정
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True

    # A100 감지
    if 'A100' in torch.cuda.get_device_name(0):
        print("✨ A100 GPU 감지! 최적 설정 적용")
        BATCH_SIZE = 64
        ACCUMULATION_STEPS = 1
    else:
        BATCH_SIZE = 32
        ACCUMULATION_STEPS = 2
else:
    device = torch.device('cpu')
    BATCH_SIZE = 8
    ACCUMULATION_STEPS = 4

# 재현성을 위한 시드 고정
import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

print(f"\n✅ 환경 설정 완료! Batch Size: {BATCH_SIZE}")
