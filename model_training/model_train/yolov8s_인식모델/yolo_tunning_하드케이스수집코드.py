from ultralytics import YOLO
from pathlib import Path
import shutil
import random
import cv2

# ==================================================
# 설정
# ==================================================
MODEL_PATH = "runs/detect/food_detect_v8s/weights/best.pt"

# 원본 데이터
VAL_IMG_DIR = Path(r"C:/Users/asia/PycharmProjects/food_classification/Yolo/yolo_dataset/images/val")
VAL_LBL_DIR = Path(r"C:/Users/asia/PycharmProjects/food_classification/Yolo/yolo_dataset/labels/val")

TRAIN_IMG_DIR = Path(r"C:/Users/asia/PycharmProjects/food_classification/Yolo/yolo_dataset/images/train")
TRAIN_LBL_DIR = Path(r"C:/Users/asia/PycharmProjects/food_classification/Yolo/yolo_dataset/labels/train")

# 출력 (혼합 데이터셋)
OUT_IMG_DIR = Path(r"C:/Users/asia/PycharmProjects/food_classification/Yolo/hardeasy_cases/images")
OUT_LBL_DIR = Path(r"C:/Users/asia/PycharmProjects/food_classification/Yolo/hardeasy_cases/labels")

OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_LBL_DIR.mkdir(parents=True, exist_ok=True)


# **하드케이스 판별 기준**
CONF_LOW = 0.35
AREA_RATIO_TH = 0.03
MIN_BOX_CNT = 3

IMGSZ = 960
CONF_INFER = 0.05

# 하드:이지 비율
HARD_RATIO = 0.6
SEED = 42
random.seed(SEED)


# 모델 로드
model = YOLO(MODEL_PATH)

hard_images = set()
easy_images = set()


# 하드케이스 자동 수집 (val 기준)

print("[STEP 1] Collecting hard cases...")

for img_path in VAL_IMG_DIR.glob("*.*"):
    if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    result = model(img_path, imgsz=IMGSZ, conf=CONF_INFER)[0]
    img = cv2.imread(str(img_path))
    H, W = img.shape[:2]
    img_area = H * W

    is_hard = False

    # 박스 없음
    if result.boxes is None or len(result.boxes) == 0:
        is_hard = True

    # 박스 개수 적음
    elif len(result.boxes) < MIN_BOX_CNT:
        is_hard = True

    else:
        for box, conf in zip(result.boxes.xyxy, result.boxes.conf):
            x1, y1, x2, y2 = box.cpu().numpy()
            area_ratio = ((x2 - x1) * (y2 - y1)) / img_area

            if area_ratio < AREA_RATIO_TH or conf < CONF_LOW:
                is_hard = True
                break

    if is_hard:
        hard_images.add(img_path.name)


#이지케이스 샘플링 (train 기준)

hard_count = len(hard_images)
easy_target = int(hard_count * (1 - HARD_RATIO) / HARD_RATIO)

all_train_imgs = [
    p for p in TRAIN_IMG_DIR.glob("*.*")
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
]

easy_samples = random.sample(
    all_train_imgs,
    min(easy_target, len(all_train_imgs))
)

easy_images = {p.name for p in easy_samples}

print(f"Easy cases sampled: {len(easy_images)}")



def copy_pair(img_name, src_img_dir, src_lbl_dir):
    shutil.copy(
        src_img_dir / img_name,
        OUT_IMG_DIR / img_name
    )

    lbl = src_lbl_dir / f"{Path(img_name).stem}.txt"
    if lbl.exists():
        shutil.copy(
            lbl,
            OUT_LBL_DIR / lbl.name
        )

for name in hard_images:
    copy_pair(name, VAL_IMG_DIR, VAL_LBL_DIR)


for name in easy_images:
    copy_pair(name, TRAIN_IMG_DIR, TRAIN_LBL_DIR)


#결과
print("=" * 60)
print(f"Hard cases : {len(hard_images)}")
print(f"Easy cases : {len(easy_images)}")
print(f"Total      : {len(hard_images) + len(easy_images)}")
print(f"Output dir : {OUT_IMG_DIR}")
