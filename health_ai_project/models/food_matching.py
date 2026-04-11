# models/food_matching.py
# food_name 기반 + class_id 기반 영양 매핑 통합 버전

from typing import List, Dict, Any
from pathlib import Path
import pandas as pd

from config import BASE_DIR

print("🔥🔥🔥 models/food_matching.py ACTUALLY LOADED 🔥🔥🔥")

# =========================
# 설정
# =========================
NUTRITION_XLSX_PATH = BASE_DIR / "data" / "food_nutrition.xlsx"
CONF_THRESHOLD_DEFAULT = 0.5


class FoodMatcher:
    """
    ✔ food_name 기반 매칭 (현재 서비스 주력)
    ✔ class_id 기반 매칭 (YOLO 완성 대비)
    """

    def __init__(self, nutrition_path: Path | None = None):
        self.nutrition_path = nutrition_path or NUTRITION_XLSX_PATH
        self.df = self._load_nutrition()

    def _load_nutrition(self) -> pd.DataFrame:
        df = pd.read_excel(self.nutrition_path)

        required_cols = {
            "class_id",
            "음식명",
            "에너지(kcal)",
            "탄수화물(g)",
            "단백질(g)",
            "지방(g)",
        }
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"food_nutrition.xlsx missing columns: {missing}")

        return df

    # =====================================================
    # 🔧 문자열 정규화 (🔥 이번 에러의 핵심)
    # =====================================================
    def _normalize(self, text: str) -> str:
        if not text:
            return ""
        return (
            str(text)
            .strip()
            .lower()
            .replace(" ", "")
        )

    # =====================================================
    # ✅ 1️⃣ food_name 기반 매칭 (현재 API 사용)
    # =====================================================
    def match_by_name(self, food_name: str):
        if not food_name:
            return None

        target = self._normalize(food_name)

        # norm 컬럼 캐싱 (중복 생성 방지)
        if "norm" not in self.df.columns:
            self.df["norm"] = self.df["음식명"].astype(str).apply(self._normalize)

        def is_match(x: str) -> bool:
            return target in x or x in target

        candidates = self.df[self.df["norm"].apply(is_match)]

        if candidates.empty:
            print("❌ No match for:", food_name, "->", target)
            return None

        row = candidates.iloc[0]
        return self._row_to_nutrition(row)

    # =====================================================
    # ✅ 2️⃣ class_id 기반 매칭 (YOLO detection용)
    # =====================================================
    def match_by_detection(
        self,
        detections: List[Dict[str, Any]],
        conf_threshold: float = CONF_THRESHOLD_DEFAULT,
    ) -> List[Dict[str, Any]]:

        results: List[Dict[str, Any]] = []

        for det in detections:
            class_id = det.get("class_id")
            confidence = float(det.get("confidence", 0.0))

            if confidence < conf_threshold:
                results.append({**det, "status": "low_confidence"})
                continue

            row = self.df[self.df["class_id"] == class_id]
            if row.empty:
                results.append({**det, "status": "not_found"})
                continue

            nutrition = self._row_to_nutrition(row.iloc[0])

            results.append({
                "class_id": class_id,
                "food_name": nutrition["food_name"],
                "confidence": confidence,
                "nutrition": nutrition["nutrition"],
                "status": "matched",
            })

        return results

    # =====================================================
    # ✅ 3️⃣ food_name 리스트 기반 매칭 (FoodPipeline용)
    # =====================================================
    def match_by_predictions(
        self,
        predictions: List[Dict[str, Any]],
        conf_threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:

        results: List[Dict[str, Any]] = []

        for p in predictions:
            confidence = float(p.get("confidence", 0.0))
            if confidence < conf_threshold:
                continue

            if p.get("nutrition"):
                results.append({
                    "food_name": p["food_name"],
                    "confidence": confidence,
                    "nutrition": p["nutrition"],
                    "status": "matched",
                    "source": p.get("source", "pipeline"),
                })
                continue

            nutrition = self.match_by_name(p["food_name"])
            if nutrition:
                results.append({
                    "food_name": nutrition["food_name"],
                    "confidence": confidence,
                    "nutrition": nutrition["nutrition"],
                    "status": "matched",
                    "source": "matcher_fallback",
                })

        return results

    # =====================================================
    # 🔧 공통 변환 함수
    # =====================================================
    def _row_to_nutrition(self, row: pd.Series) -> Dict[str, Any]:
        return {
            "food_name": row["음식명"],
            "nutrition": {
                "kcal": float(row["에너지(kcal)"]),
                "carbs_g": float(row["탄수화물(g)"]),
                "protein_g": float(row["단백질(g)"]),
                "fat_g": float(row["지방(g)"]),
            },
        }
