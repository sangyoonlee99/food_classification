from ultralytics import YOLO
from pathlib import Path
import shutil
import cv2

# ==================================================
# 설정
# ==================================================
MODEL_PATH = Path("./weights/yolo_hard_finetune_best.pt")

VAL_IMG_DIR = Path("./yolo_dataset/images/val")
VAL_LBL_DIR = Path("./yolo_dataset/labels/val")

OUT_IMG_DIR = Path("./near_hard/images")
OUT_LBL_DIR = Path("./near_hard/labels")

OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_LBL_DIR.mkdir(parents=True, exist_ok=True)

# ==================================================
# Near-Hard 기준 (핵심)
# ==================================================
CONF_LOW = 0.25        # 너무 낮은 완전 실패 제외
CONF_HIGH = 0.55       # 애매한 구간만
AREA_RATIO_TH = 0.05   # 5% 이하 = 작은 반찬
MIN_BOX_CNT = 2        # 다중 음식 기준

IMGSZ = 960
CONF_INFER = 0.15      # 추론 시 박스는 나오게

# ==================================================
# 모델 로드
# ==================================================
model = YOLO(MODEL_PATH)

near_hard = set()
total = 0

print("[STEP] Collecting near-hard cases...")

for img_path in VAL_IMG_DIR.glob("*.*"):
    if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    total += 1
    result = model(img_path, imgsz=IMGSZ, conf=CONF_INFER)[0]

    # 박스 없음 → 제외
    if result.boxes is None or len(result.boxes) == 0:
        continue

    img = cv2.imread(str(img_path))
    H, W = img.shape[:2]
    img_area = H * W

    is_near_hard = False

    # 박스 개수 부족 (완전 실패는 아님)
    if len(result.boxes) < MIN_BOX_CNT:
        is_near_hard = True

    # 애매한 confidence or 작은 박스
    for box, conf in zip(result.boxes.xyxy, result.boxes.conf):
        x1, y1, x2, y2 = box.cpu().numpy()
        area_ratio = ((x2 - x1) * (y2 - y1)) / img_area

        if (
            CONF_LOW <= conf <= CONF_HIGH
            or area_ratio < AREA_RATIO_TH
        ):
            is_near_hard = True
            break

    if is_near_hard:
        near_hard.add(img_path.name)

        shutil.copy(img_path, OUT_IMG_DIR / img_path.name)

        lbl = VAL_LBL_DIR / f"{img_path.stem}.txt"
        if lbl.exists():
            shutil.copy(lbl, OUT_LBL_DIR / lbl.name)

        print(f"[NEAR-HARD] {img_path.name}")

print("=" * 60)
print(f"Total val images     : {total}")
print(f"Near-hard cases      : {len(near_hard)}")
print(f"Saved to             : {OUT_IMG_DIR}")
