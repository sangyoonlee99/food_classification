# config.py
from __future__ import annotations

import os
from pathlib import Path

# =====================================================
# 프로젝트 루트
# =====================================================
BASE_DIR = Path(__file__).resolve().parent

# =====================================================
# YOLO 데이터셋 경로 (학습용)  ✅ PC별로 바뀌는 값
# - 우선순위: 환경변수 YOLO_DATA_ROOT > 기본값 D:\yolo_dataset
# =====================================================
YOLO_DATA_ROOT = Path(os.getenv("YOLO_DATA_ROOT", r"D:\yolo_dataset"))

YOLO_IMAGES_DIR = YOLO_DATA_ROOT / "images"
YOLO_LABELS_DIR = YOLO_DATA_ROOT / "labels"
YOLO_CLASSES_FILE = YOLO_DATA_ROOT / "classes.txt"
YOLO_DATA_YAML = YOLO_DATA_ROOT / "data.yaml"

# =====================================================
# YOLO 모델 / 결과 저장 경로
# =====================================================
YOLO_MODEL_DIR = BASE_DIR / "models" / "weights"
YOLO_MODEL_DIR.mkdir(parents=True, exist_ok=True)

YOLO_MODEL_NAME = "food_yolov8n"
YOLO_BASE_MODEL = "yolov8n.pt"

YOLO_RUNS_DIR = BASE_DIR / "models" / "runs"
YOLO_RUNS_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# ⭐ YOLO 서비스 추론용 모델 경로 (가장 중요)
# =====================================================
YOLO_SERVICE_MODEL_PATH = YOLO_RUNS_DIR / YOLO_MODEL_NAME / "weights" / "best.pt"
if not YOLO_SERVICE_MODEL_PATH.exists():
    print(f"[WARN] YOLO_SERVICE_MODEL_PATH not found: {YOLO_SERVICE_MODEL_PATH}")

# =====================================================
# YOLO 서비스 실행 디바이스  ✅ CUDA 없으면 자동 CPU
# =====================================================
def _pick_device() -> str:
    # 기본은 CPU (안전)
    device = os.getenv("YOLO_DEVICE", "cpu").strip().lower()
    if device.startswith("cuda"):
        try:
            import torch  # noqa
            if torch.cuda.is_available():
                return os.getenv("YOLO_DEVICE", "cuda:0")
            # CUDA 지정했는데 사용 불가면 CPU로 강등
            print("[WARN] CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        except Exception:
            print("[WARN] Torch CUDA check failed. Falling back to CPU.")
            return "cpu"
    return "cpu"

YOLO_SERVICE_DEVICE = _pick_device()

# =====================================================
# YOLO 서비스 추론 파라미터 (Detection Inference)
# =====================================================
YOLO_SERVICE_CONF_THRESHOLD = float(os.getenv("YOLO_CONF", "0.25"))
YOLO_SERVICE_IOU_THRESHOLD = float(os.getenv("YOLO_IOU", "0.45"))
