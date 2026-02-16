# infer_with_prototype_basic.py
from pathlib import Path
import torch
import torch.nn.functional as F

from embeddings import EfficientNetEmbedding, image_to_embedding

# =========================
# Config
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 5

PROTO_PATH = Path("./weights/prototypes.pt")

# =========================
# Load prototypes
# =========================
data = torch.load(PROTO_PATH, map_location="cpu")
prototypes = data["prototypes"]   # dict: class -> Tensor(D)
class_names = data["classes"]

# (C, D)
proto_matrix = torch.stack([prototypes[c] for c in class_names]).to(DEVICE)

# =========================
# Load embedder
# =========================
embedder = EfficientNetEmbedding(
    model_name="efficientnet_b4",
    weight_path=r"C:\Users\asia\PycharmProjects\food_classification\efficientnet_b4\checkpoints_b4_68class\efficientnet_b4_68class_best.pth",
    device=DEVICE
)

# =========================
# Inference
# =========================
def infer_image(img_path: Path, top_k=5):
    emb = image_to_embedding(img_path, embedder, DEVICE)  # (D,)
    emb = emb.to(DEVICE)

    # cosine similarity
    sims = F.cosine_similarity(
        emb.unsqueeze(0),      # (1, D)
        proto_matrix,          # (C, D)
        dim=1
    )  # (C,)

    scores, indices = torch.topk(sims, top_k)

    results = []
    for s, i in zip(scores, indices):
        results.append({
            "class": class_names[i.item()],
            "similarity": round(float(s), 4)
        })

    return results

# =========================
# Test
# =========================
if __name__ == "__main__":
    img = Path(r"C:\Users\asia\PycharmProjects\food_classification\infer_combined\data\돌솥비빔밥1.jpg")  # 🔥 테스트 이미지
    out = infer_image(img, TOP_K)

    for r in out:
        print(r)
