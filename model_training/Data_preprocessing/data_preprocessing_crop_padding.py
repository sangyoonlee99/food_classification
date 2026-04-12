# scripts/crop_pad_for_efficientnet.py

import json
from pathlib import Path
import cv2
import numpy as np

# =========================
# Config
# =========================
JSON_PATH = Path(r"E:\dataset_for_crop_padding\bboxes_all.json")
SRC_IMG_ROOT = Path(r"E:\dataset_for_crop_padding\images")
OUT_ROOT = Path(r"E:\efficientnet_dataset_ver_croppadding")

IMG_SIZE = 352
IMG_EXT = ".jpg"   

# =========================
# Utils
# =========================
def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def imwrite_unicode(path: Path, img, params=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        raise ValueError(f"Unsupported extension: {ext}")

    if params is None:
        params = []

    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        return False

    buf.tofile(str(path))
    return True

def yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
    bw = w * img_w
    bh = h * img_h
    cx = xc * img_w
    cy = yc * img_h

    x1 = int(cx - bw / 2)
    y1 = int(cy - bh / 2)
    x2 = int(cx + bw / 2)
    y2 = int(cy + bh / 2)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w - 1, x2)
    y2 = min(img_h - 1, y2)

    return x1, y1, x2, y2

def pad_to_square(img):
    h, w, _ = img.shape
    size = max(h, w)

    pad_top = (size - h) // 2
    pad_bottom = size - h - pad_top
    pad_left = (size - w) // 2
    pad_right = size - w - pad_left

    return cv2.copyMakeBorder(
        img,
        pad_top, pad_bottom,
        pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

# =========================
# Main
# =========================
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

count = 0
missing = 0
read_fail = 0
write_fail = 0

for idx, item in enumerate(data, start=1):
    image_name = item["image"]
    split = item["split"]
    bboxes = item["bboxes_yolo"]

    img_path = SRC_IMG_ROOT / split / image_name
    if not img_path.exists():
        missing += 1
        continue

    image = imread_unicode(img_path)
    if image is None:
        read_fail += 1
        continue

    h, w, _ = image.shape

    for i, (xc, yc, bw, bh) in enumerate(bboxes):
        x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, bw, bh, w, h)

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_sq = pad_to_square(crop)
        crop_resized = cv2.resize(crop_sq, (IMG_SIZE, IMG_SIZE))

        out_dir = OUT_ROOT / split
        orig_stem = img_path.stem
        out_name = f"{orig_stem}_{i}{IMG_EXT}"
        out_file = out_dir / out_name

        # 🔽 파일이 이미 존재하면 건너뜀
        if out_file.exists():
            continue

        ok = imwrite_unicode(out_file, crop_resized)
        if not ok:
            write_fail += 1
            continue

        count += 1

    # 진행 로그
    if idx % 500 == 0:
        print(f"[INFO] processed {idx}/{len(data)} | saved={count} | missing={missing} | read_fail={read_fail} | write_fail={write_fail}")

print(f"\nDone | Saved crops: {count}")
print(f" - missing images: {missing}")
print(f" - read failed: {read_fail}")
print(f" - write failed: {write_fail}")
print(f"Output: {OUT_ROOT}")
