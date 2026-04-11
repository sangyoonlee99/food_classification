# services/next_meal_service.py
# STEP 6: 다음 끼니 추천 서비스

from common.schemas import DailyDietPlan, UserGoal


class NextMealService:
    def recommend(
        self,
        diet_plan: DailyDietPlan,
        goal: UserGoal,
    ) -> dict:

        # 기본 전략: 단백질 우선
        return {
            "meal": "다음 끼니",
            "headline": "다음 끼니에서는 단백질을 먼저 보충하세요.",
            "foods": [
                {
                    "name": "닭가슴살",
                    "gram": 100,
                    "kcal": 165,
                }
            ],
        }
