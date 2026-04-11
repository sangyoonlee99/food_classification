# services/nutrient_boost_service.py
# STEP 5-7: 부족 영양소 보완 추천 (1줄 + 1음식)

from typing import Dict, Optional
from common.schemas import NutritionSummary


class NutrientBoostService:
    """
    부족 영양소 1개를 골라 보완 음식 1개 추천
    """

    # 기준치 (MVP)
    THRESHOLD = {
        "protein_g": 40,
        "fat_g": 15,
        "kcal": 1500,
    }

    # 보완 음식 맵 (단일, 명확)
    BOOST_FOODS = {
        "protein_g": {"food": "닭가슴살", "amount": "100g", "kcal": 165},
        "fat_g": {"food": "올리브유", "amount": "1큰술", "kcal": 120},
        "kcal": {"food": "현미밥", "amount": "1공기", "kcal": 520},
    }

    def recommend(
        self,
        nutrition: NutritionSummary,
    ) -> Optional[Dict[str, str]]:

        total = nutrition.total

        # 1️⃣ 우선순위: 단백질 → 지방 → 열량
        if total.get("protein_g", 0) < self.THRESHOLD["protein_g"]:
            key = "protein_g"
            reason = "단백질 섭취가 부족합니다."

        elif total.get("fat_g", 0) < self.THRESHOLD["fat_g"]:
            key = "fat_g"
            reason = "지방 섭취가 부족합니다."

        elif total.get("kcal", 0) < self.THRESHOLD["kcal"]:
            key = "kcal"
            reason = "전체 섭취 열량이 낮습니다."

        else:
            return None  # 보완 불필요

        food = self.BOOST_FOODS[key]

        return {
            "message": f"{reason} {food['food']} {food['amount']}을(를) 추가해 보세요.",
            "food": food["food"],
            "amount": food["amount"],
            "kcal": str(food["kcal"]),
        }
