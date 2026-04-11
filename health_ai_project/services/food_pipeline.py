# services/food_pipeline.py

print("🔥🔥🔥 services/food_pipeline.py LOADED 🔥🔥🔥")

from pathlib import Path
from typing import Dict, Any

from models.detection_yolo import detect_food_yolo


class FoodPipeline:
    def __init__(self):
        pass

    def predict(self, image_path: Path) -> Dict[str, Any]:
        """
        항상 food_name을 반환하는 표준 파이프라인
        """
        image_path = Path(image_path)

        # 1️⃣ YOLO 시도
        yolo_result = detect_food_yolo(image_path)

        if yolo_result.get("num_detections", 0) > 0:
            top = yolo_result["predictions"][0]
            return {
                "source": "yolo",
                "food_name": top.get("label"),
                "confidence": float(top.get("confidence", 0.0)),
                "image": str(image_path),
            }

        # 2️⃣ filename fallback (🔥 반드시 food_name 리턴)
        stem = image_path.stem
        parts = stem.split("_")

        food_name = None
        if len(parts) >= 4:
            food_name = parts[3]
        else:
            food_name = stem

        return {
            "source": "filename",
            "food_name": food_name,
            "confidence": 0.5,
            "image": str(image_path),
        }
