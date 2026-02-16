from pathlib import Path
import torch
import json
from tqdm import tqdm

# =========================
# Config
# =========================
DATASET_ROOT = Path("E:/classification_dataset_split_5to1_filtered/train")
OUT_PATH = Path("weights/prototypes.pt")
CLASS_JSON = Path("weights/classes.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Load embedder
# =========================
embedder = EfficientNetEmbedding(
    model_name="efficientnet_b4",
    weight_path="weights/efficientnet_b4.pth",
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

    image_paths = list(class_dir.glob("*.jpg"))
    if len(image_paths) == 0:
        continue

    for img_path in tqdm(image_paths, desc=f"Embedding {class_name}"):
        emb = image_to_embedding(img_path, embedder, DEVICE)
        embeddings.append(emb)

    # (N, D)
    emb_stack = torch.stack(embeddings)

    # 평균 + 정규화
    proto = emb_stack.mean(dim=0)
    proto = torch.nn.functional.normalize(proto, dim=0)

    prototypes[class_name] = proto

# =========================
# Save
# =========================
torch.save(
    {
        "prototypes": prototypes,
        "classes": class_names
    },
    OUT_PATH
)

with open(CLASS_JSON, "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)

print(f"✅ Prototype saved to {OUT_PATH}")
print(f"✅ Total classes: {len(class_names)}")
