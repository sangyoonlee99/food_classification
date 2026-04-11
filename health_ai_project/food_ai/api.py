from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import uuid
import torch

from core.detector import FoodDetector
from core.prototype import PrototypeMatcher
from core.inference import infer_image
from embeddings import EfficientNetEmbedding
from decision.ui_policy import attach_ui_mode
from service.schemas import build_inference_response

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()

# 🔥 서버 시작 시 1번만 로딩
detector = FoodDetector("weights/yolo.pt", conf=0.4)
matcher = PrototypeMatcher("weights/prototypes.pt", DEVICE)
embedder = EfficientNetEmbedding(
    model_name="efficientnet_b4",
    weight_path="weights/efficientnet_b4.pth",
    device=DEVICE
)

######################  경로 설정 필요 ##################
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    image_path = UPLOAD_DIR / f"{uuid.uuid4()}.jpg"

    with open(image_path, "wb") as f:
        f.write(await file.read())

    results = infer_image(image_path, detector, matcher, embedder)
    results = attach_ui_mode(results)

    response = build_inference_response(image_path.name, results)
    return response
