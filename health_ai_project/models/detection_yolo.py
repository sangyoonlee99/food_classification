# models/detection_yolo.py
# Mock 유지 + 결과 JSON 표준화 + class_id 리스트 호환
# config.py 기반 Inference 경로 단일 관리
# 이미지 존재 여부 검사 + BASE_DIR 기준 경로 보정 (프로덕션 안전)

from __future__ import annotations

from ultralytics import YOLO
from typing import List, Dict, Any, Optional
import os
import uuid
from pathlib import Path

from config import (
    BASE_DIR,
    YOLO_SERVICE_MODEL_PATH,
    YOLO_SERVICE_DEVICE,
    YOLO_SERVICE_CONF_THRESHOLD,
    YOLO_SERVICE_IOU_THRESHOLD,
)


class FoodDetector:
    """
    - 기존 구조 유지 (model_path/device/conf_threshold)
    - 반환을 표준 DetectionResult(JSON)로 확장
    - 기존 호환을 위해 class_id 리스트만 뽑는 메서드도 제공
    - config.py 기준으로 서비스용 모델 경로 단일 관리
    - 이미지 경로를 BASE_DIR 기준으로 보정하여 실행 위치 무관
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
        conf_threshold: float | None = None,
        iou: float | None = None,
    ):
        self.model: Optional[YOLO] = None

        # 🔹 config 기본값 사용 (명시적으로 주어지면 override)
        self.model_path = model_path or str(YOLO_SERVICE_MODEL_PATH)
        self.device = device or YOLO_SERVICE_DEVICE
        self.conf_threshold = (
            conf_threshold
            if conf_threshold is not None
            else YOLO_SERVICE_CONF_THRESHOLD
        )
        self.iou = iou if iou is not None else YOLO_SERVICE_IOU_THRESHOLD

        if self.model_path and os.path.exists(self.model_path):
            self.model = YOLO(self.model_path)
            self.model.to(self.device)

    def _resolve_image_path(self, image_path: str) -> Path:
        """
        이미지 경로를 프로젝트 루트(BASE_DIR) 기준으로 보정
        """
        p = Path(image_path)
        if not p.is_absolute():
            p = BASE_DIR / p
        return p

    def detect_json(self, image_path: str) -> Dict[str, Any]:
        """
        표준 DetectionResult(JSON) 반환

        {
          "image_id": "...",
          "model": "...",
          "detections": [
             {
               "food_class": str,
               "class_id": int,
               "confidence": float,
               "bbox": {"x1": int, "y1": int, "x2": int, "y2": int}
             }
          ],
          "error": Optional[str]
        }
        """
        resolved_path = self._resolve_image_path(image_path)
        image_id = resolved_path.name or str(uuid.uuid4())

        # 🔒 [1] 이미지 파일 존재 여부 검사
        if not resolved_path.exists():
            return {
                "image_id": image_id,
                "model": "ERROR",
                "detections": [],
                "error": f"Image file not found: {resolved_path}",
            }

        # 🔹 [2] MOCK 모드 (모델 없을 때)
        if self.model is None:
            return {
                "image_id": image_id,
                "model": "MOCK",
                "detections": [
                    {
                        "food_class": "mock_food_0",
                        "class_id": 0,
                        "confidence": 0.99,
                        "bbox": {"x1": 10, "y1": 10, "x2": 200, "y2": 200},
                    },
                    {
                        "food_class": "mock_food_3",
                        "class_id": 3,
                        "confidence": 0.95,
                        "bbox": {"x1": 220, "y1": 40, "x2": 420, "y2": 240},
                    },
                ],
            }

        # 🔹 [3] 실제 YOLO 추론
        results = self.model(
            str(resolved_path),
            device=self.device,
            conf=self.conf_threshold,
            iou=self.iou,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < self.conf_threshold:
                    continue

                cls_id = int(box.cls[0])
                food_class = self.model.names.get(cls_id, str(cls_id))

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detections.append(
                    {
                        "food_class": food_class,
                        "class_id": cls_id,
                        "confidence": round(conf, 4),
                        "bbox": {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                        },
                    }
                )

        return {
            "image_id": image_id,
            "model": os.path.basename(self.model_path),
            "detections": detections,
        }

    def detect(self, image_path: str) -> List[int]:
        """
        ✅ 기존 호환: class_id 리스트만 반환
        """
        data = self.detect_json(image_path)
        return [d["class_id"] for d in data.get("detections", [])]

# --- service wrapper -------------------------------------------------

_detector = FoodDetector()

def detect_food_yolo(image_path) -> dict:
    """
    services 레이어에서 사용하는 표준 YOLO 인터페이스
    """
    data = _detector.detect_json(str(image_path))

    detections = data.get("detections", [])

    return {
        "num_detections": len(detections),
        "predictions": [
            {
                "label": d["food_class"],
                "confidence": d["confidence"],
                "bbox": d.get("bbox"),
            }
            for d in detections
        ],
        "model": data.get("model"),
        "error": data.get("error"),
    }