# api/endpoints/meal_api.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import shutil
import uuid

from services.meal_service import MealService
from config import BASE_DIR

router = APIRouter()
meal_service = MealService()

UPLOAD_DIR = BASE_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/meal/analyze")
async def analyze_meal(file: UploadFile = File(...)):
    print("🔥🔥🔥 /meal/analyze ENDPOINT CALLED 🔥🔥🔥")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in [".jpg", ".jpeg", ".png", ".webp"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    original_name = Path(file.filename).name
    temp_name = f"{uuid.uuid4()}__{original_name}"
    save_path = UPLOAD_DIR / temp_name

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print("📂 Image saved at:", save_path)

    result = meal_service.analyze_meal(
        image_path=str(save_path),
        context=None,
    )

    print("✅ MealService result returned")
    return result
