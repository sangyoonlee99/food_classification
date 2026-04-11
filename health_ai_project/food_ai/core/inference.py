# food_ai/core/inference.py
import cv2
import numpy as np
from pathlib import Path
import torch

from food_ai.core.detector import FoodDetector
from food_ai.core.prototype import PrototypeMatcher
from food_ai.embeddings import image_to_embedding

# ✅ CPU/GPU 자동
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def crop_and_pad(img, x1, y1, x2, y2):
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    h, w = crop.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side, 3), dtype=np.uint8)

    y_off = (side - h) // 2
    x_off = (side - w) // 2
    canvas[y_off:y_off+h, x_off:x_off+w] = crop
    return canvas


def infer_image(
    image_path: Path,
    detector: FoodDetector,
    matcher: PrototypeMatcher,
    embedder,
    min_bbox_ratio=0.02,
    top_k=5,
):
    img = cv2.imread(str(image_path))
    if img is None:
        return []

    H, W = img.shape[:2]
    results = []

    boxes = detector.detect(img) or []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if (x2 - x1) * (y2 - y1) / (H * W) < min_bbox_ratio:
            continue

        crop = crop_and_pad(img, x1, y1, x2, y2)
        if crop is None:
            continue

        emb = image_to_embedding(crop, embedder, device=DEVICE)
        candidates = matcher.topk(emb, top_k) or []

        results.append({
            "bbox": [x1, y1, x2, y2],
            "candidates": candidates,  # 다양한 포맷 가능 (서비스에서 정규화)
        })

    return results
