import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from pathlib import Path
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from ultralytics import YOLO

from embeddings import EfficientNetEmbedding, image_to_embedding

# =========================
# Config
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

YOLO_WEIGHT = Path("./weights/yolo.pt")
PROTO_PATH  = Path("./weights/prototypes_wo_치킨커리.pt")

SAVE_DIR = Path("./data/bbox확인")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


TOP_K = 5
YOLO_CONF = 0.15
MIN_BBOX_RATIO = 0.02   # 너무 작은 박스 제거

CONTAIN_THRES = 0.8   # 포함 비율
AREA_RATIO_THRES = 2.0  # 면적 비율 (큰 박스 제거 기준)

# =========================
# Load YOLO
# =========================
yolo = YOLO(YOLO_WEIGHT)

# =========================
# Load prototypes
# =========================
proto_data = torch.load(PROTO_PATH, map_location="cpu")
prototypes = proto_data["prototypes"]
class_names = proto_data["classes"]
# proto_data = torch.load(
#     PROTO_PATH,
#     map_location="cpu",
#     weights_only=True
# )

proto_matrix = torch.stack(
    [prototypes[c] for c in class_names]
).to(DEVICE)  # (C, D)

# =========================
# Load embedder
# =========================
embedder = EfficientNetEmbedding(
    model_name="efficientnet_b4",
    weight_path=r"C:\Users\asia\PycharmProjects\food_classification\efficientnet_b4\checkpoints_b4_68class\efficientnet_b4_68class_best.pth",
    device=DEVICE
)

# =========================
# Utils
# =========================
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


def box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def containment_ratio(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return 0.0

    areaB = box_area(boxB)
    return inter_area / areaB


# =========================
# 🔥 핵심: 포함관계 제거 (정식 버전)
# =========================
def suppress_contained_boxes(
    results,
    contain_thres=0.8,
    area_ratio_thres=2.0
):
    keep = [True] * len(results)

    for i in range(len(results)):
        if not keep[i]:
            continue

        for j in range(len(results)):
            if i == j or not keep[j]:
                continue

            # 같은 음식만 비교
            if results[i]["top1"]["class"] != results[j]["top1"]["class"]:
                continue

            boxA = results[i]["bbox"]
            boxB = results[j]["bbox"]

            # B가 A 안에 포함?
            contain = containment_ratio(boxA, boxB)
            if contain < contain_thres:
                continue

            areaA = box_area(boxA)
            areaB = box_area(boxB)

            # 🔥 핵심: 충분히 큰 박스만 제거 후보
            if areaA < area_ratio_thres * areaB:
                continue

            simA = results[i]["top1"]["similarity"]
            simB = results[j]["top1"]["similarity"]

            if simA >= simB:
                keep[j] = False
            else:
                keep[i] = False
                break

    return [r for r, k in zip(results, keep) if k]


# =========================
# Inference
# =========================
def infer_image(image_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError("이미지 로드 실패")

    H, W = img.shape[:2]
    results = []

    yolo_res = yolo(img, conf=YOLO_CONF)[0]
    print(f"▶ YOLO boxes: {len(yolo_res.boxes)}")

    #  YOLO 박스가 그려진 원본 이미지 저장
    annotated = yolo_res.plot()
    out_path = SAVE_DIR / f"{image_path.stem}_yolo.jpg"
    cv2.imwrite(str(out_path), annotated)

    # 🔥 YOLO 박스 실시간 시각화
    cv2.imshow("YOLO Detection", annotated)
    cv2.waitKey(0)  # 1ms 대기 (실시간용)

    # 1️⃣ YOLO → embedding → similarity
    for box in yolo_res.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if (x2-x1)*(y2-y1)/(H*W) < MIN_BBOX_RATIO:
            continue

        crop = crop_and_pad(img, x1, y1, x2, y2)
        if crop is None:
            continue

        # 🔥 bbox crop 저장
        crop_path = SAVE_DIR / f"{image_path.stem}_crop_{x1}_{y1}_{x2}_{y2}.jpg"
        cv2.imwrite(str(crop_path), crop)

        emb = image_to_embedding(crop, embedder, DEVICE).to(DEVICE)

        sims = F.cosine_similarity(
            emb.unsqueeze(0),
            proto_matrix,
            dim=1
        )

        scores, indices = torch.topk(sims, TOP_K)

        candidates = [
            {
                "class": class_names[idx.item()],
                "similarity": round(float(s), 4)
            }
            for s, idx in zip(scores, indices)
        ]

        results.append({
            "bbox": [x1, y1, x2, y2],
            "top1": candidates[0],
            "candidates": candidates
        })
    # 2️⃣ 🔥 여기서 딱 한 번만 포함관계 제거
    results = suppress_contained_boxes(
        results,
        contain_thres=CONTAIN_THRES,
        area_ratio_thres=AREA_RATIO_THRES
    )

    return results


# =========================
# Run
# =========================
if __name__ == "__main__":
    image = Path("./data/닭가슴살고구마.jpg")
    outputs = infer_image(image)

    #for o in outputs:
    #    print(o)

    for i, o in enumerate(outputs, 1):
        print(f"\n[결과 {i}]")
        print(f"음식명 : {o['top1']['class']}")
        print(f"유사도 : {o['top1']['similarity']}")

        print("top-k 후보:")
        for k, cand in enumerate(o["candidates"], 1):
            print(f"  {k}. {cand['class']} | {cand['similarity']}")
