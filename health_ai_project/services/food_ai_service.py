from pathlib import Path
import torch
from typing import Any, List, Dict

from food_ai.core.inference import infer_image
from food_ai.core.detector import FoodDetector
from food_ai.core.prototype import PrototypeMatcher
from food_ai.embeddings import EfficientNetEmbedding

#########닭가슴살2 도 닭가슴살로 인식하도록 #####
import re

class FoodAIService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.detector = FoodDetector("food_ai/weights/yolo.pt")
        self.matcher = PrototypeMatcher(
            "food_ai/weights/prototypes.pt",
            device=self.device,
        )
        self.embedder = EfficientNetEmbedding(
            model_name="efficientnet_b4",
            weight_path="food_ai/weights/efficientnet_b4.pth",
            device=self.device,
        )

        self.class_name_map = getattr(self.matcher, "class_name_map", None)

    def _normalize_label(self, label: Any) -> str | None:
        if isinstance(label, str):
            s = label.strip()
            if not s or s.lower().startswith("class"):
                return None
            ###### 닭가슴살2도 닭가슴살로 인식하도록
            s = re.sub(r"\d+$", "", s).strip()

            return s

        if isinstance(label, int) and self.class_name_map:
            return self.class_name_map.get(label)

        if isinstance(label, dict):
            return (
                label.get("food_name")
                or label.get("name")
                or label.get("label")
            )

        return None

    def _normalize_score(self, raw_score: Any) -> float:
        try:
            if isinstance(raw_score, (int, float)):
                return float(raw_score)
            if isinstance(raw_score, dict):
                return float(
                    raw_score.get("similarity")
                    or raw_score.get("score")
                    or raw_score.get("conf")
                    or 0.0
                )
        except Exception:
            pass
        return 0.0

    def analyze(self, image_path: Path, top_k: int = 4) -> List[Dict[str, float]]:
        results = infer_image(
            image_path=image_path,
            detector=self.detector,
            matcher=self.matcher,
            embedder=self.embedder,
            top_k=top_k,
        )

        if not results or not isinstance(results, list):
            return []

        ui_items: List[Dict[str, float]] = []

        # 1️⃣ 정상 케이스: candidates 존재
        candidates = results[0].get("candidates", [])
        if candidates:
            source = candidates
        else:
            # 2️⃣ fallback: YOLO 결과 직접 사용
            source = results

        for item in source:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                label, raw_score = item[0], item[1]
            elif isinstance(item, dict):
                label = (
                    item.get("food_name")
                    or item.get("label")
                    or item.get("name")
                    or item.get("class")
                )
                raw_score = (
                    item.get("score")
                    or item.get("similarity")
                    or item.get("conf")
                )
            else:
                continue

            food_name = self._normalize_label(label)
            if not food_name:
                continue

            score = self._normalize_score(raw_score)

            ui_items.append(
                {
                    "food_name": food_name,
                    # 점수 제거 위해 주석처리해봄
                    "score": float(score),
                }
            )
        # 점수 제거 위해 주석처리
        ui_items.sort(key=lambda x: x["score"], reverse=True)
        return ui_items[:top_k]
