# models/nutrition_calculator.py
# 표준: calculate_meal_nutrition
# 호환: NutritionCalculator (legacy wrapper)

from typing import List, Dict, Any


# =========================
# ✅ 표준 방식 (서비스 기준)
# =========================
def calculate_meal_nutrition(
    matched_foods: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    FoodMatcher 결과 → 한 끼 영양 합산
    """
    total = {
        "kcal": 0.0,
        "carbs_g": 0.0,
        "protein_g": 0.0,
        "fat_g": 0.0,
    }

    count = 0

    for item in matched_foods:
        if item.get("status") != "matched":
            continue

        nutrition = item.get("nutrition", {})

        total["kcal"] += float(nutrition.get("kcal", 0.0))
        total["carbs_g"] += float(nutrition.get("carbs_g", 0.0))
        total["protein_g"] += float(nutrition.get("protein_g", 0.0))
        total["fat_g"] += float(nutrition.get("fat_g", 0.0))

        count += 1

    for k in total:
        total[k] = round(total[k], 3)

    return {
        "total": total,
        "items_count": count,
    }


# =====================================
# 🔁 기존 코드 호환용 (Legacy Adapter)
# =====================================
class NutritionCalculator:
    """
    ⚠️ Legacy 호환용
    - 기존 food_names 기반 구조를
      새로운 calculate_meal_nutrition 구조로 연결
    """

    def __init__(self, nutrition_db: Dict[str, Dict[str, float]]):
        self.db = nutrition_db

    def calculate(self, food_names: List[str]) -> Dict[str, Any]:
        """
        food_names → legacy 결과
        (내부적으로는 새로운 계산 로직 사용)
        """
        matched_foods = []

        for food in food_names:
            data = self.db.get(food)
            if not data:
                continue

            matched_foods.append({
                "food_name": food,
                "nutrition": {
                    "kcal": data.get("calories", 0.0),
                    "carbs_g": data.get("carb", 0.0),
                    "protein_g": data.get("protein", 0.0),
                    "fat_g": data.get("fat", 0.0),
                },
                "status": "matched",
            })

        result = calculate_meal_nutrition(matched_foods)

        return {
            "summary": result["total"],
            "details": matched_foods,
        }
