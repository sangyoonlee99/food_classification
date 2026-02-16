# build_prototypes.py
from pathlib import Path
import json
import torch
from tqdm import tqdm

from embeddings import EfficientNetEmbedding, image_to_embedding


# Config

DATASET_ROOT = Path(r"E:/classification_dataset_split_5to1_filtered_copied/train")
OUT_PATH = Path("weights/prototypes.pt")
CLASS_JSON = Path("weights/classes.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Load embedder

embedder = EfficientNetEmbedding(
    model_name="efficientnet_b4",
    weight_path=r"C:\Users\asia\PycharmProjects\food_classification\efficientnet_b4\checkpoints_b4_68class\efficientnet_b4_68class_best.pth",
    device=DEVICE
)

# =========================
# Build prototypes
# =========================
prototypes = {}
class_names = []

for class_dir in sorted(DATASET_ROOT.iterdir()):
    if not class_dir.is_dir():
        continue

    class_name = class_dir.name
    class_names.append(class_name)

    embeddings = []

    image_paths = [
        p for p in class_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]

    for img_path in tqdm(image_paths, desc=f"Embedding {class_name}"):
        emb = image_to_embedding(img_path, embedder, DEVICE)
        embeddings.append(emb)

    emb_stack = torch.stack(embeddings)  # (N, D)
    proto = emb_stack.mean(dim=0)
    proto = torch.nn.functional.normalize(proto, dim=0)

    prototypes[class_name] = proto







# 저장
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
CLASS_JSON.parent.mkdir(parents=True, exist_ok=True)


torch.save(
    {
        "prototypes": prototypes,
        "classes": class_names
    },
    OUT_PATH
)

with open(CLASS_JSON, "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)

print(f"Prototype saved to {OUT_PATH}")
print(f"Total classes: {len(class_names)}")
