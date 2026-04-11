# food_ai/main.py

from pathlib import Path
import torch

from core.detector import FoodDetector
from core.prototype import PrototypeMatcher
from core.inference import infer_image
from embeddings import EfficientNetEmbedding

from decision.ui_policy import attach_ui_mode
from service.schemas import build_inference_response

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    detector = FoodDetector("weights/yolo.pt", conf=0.4)
    matcher = PrototypeMatcher("weights/prototypes.pt", DEVICE)

    embedder = EfficientNetEmbedding(
        model_name="efficientnet_b4",
        weight_path="weights/efficientnet_b4.pth",
        device=DEVICE
    )

    image_path = Path("./data/제미나이3.jpg")
    image_id = image_path.name

    results = infer_image(image_path, detector, matcher, embedder)
    results = attach_ui_mode(results)

    response = build_inference_response(image_id, results)

    print(response)


if __name__ == "__main__":
    main()
