# main.py
from pathlib import Path
import torch

from core.detector import FoodDetector
from core.prototype import PrototypeMatcher
from core.inference import infer_image
from embeddings import EfficientNetEmbedding

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

detector = FoodDetector("./weights/yolo.pt", conf=0.4)
matcher = PrototypeMatcher("./weights/prototypes.pt", DEVICE)

embedder = EfficientNetEmbedding(
    model_name="efficientnet_b4",
    weight_path="weights/efficientnet_b4.pth",
    device=DEVICE
)

image = Path("./data/제미나이3.jpg")
results = infer_image(image, detector, matcher, embedder)

for r in results:
    print(r)
