# services/monthly_summary_service.py
# STEP 4-5: 월간 식사 요약 생성 서비스

from datetime import date
from calendar import monthrange
from typing import List

from common.schemas import (
    DailyMealSummary,
    MonthlyMealSummary,
    NutritionSummary,
)

from services.meal_service import MealService


class MonthlySummaryService:
    """
    한 달치 DailyMealSummary → MonthlyMealSummary 생성
    """

    def __init__(self):
        self.meal_service = MealService()

    def build_monthly_summary(
        self,
        user_id: int,
        year: int,
        month: int,
    ) -> MonthlyMealSummary:

        # 1️⃣ 월 범위 계산
        start_day = date(year, month, 1)
        end_day = date(year, month, monthrange(year, month)[1])

        # 2️⃣ Daily summaries 로드 (MealService 경유)
        days: List[DailyMealSummary] = self.meal_service.load_daily_range(
            user_id=user_id,
            start=start_day,
            end=end_day,
        )

        # 3️⃣ 월간 영양 합산
        total = {
            "kcal": 0.0,
            "carbs_g": 0.0,
            "protein_g": 0.0,
            "fat_g": 0.0,
        }

        scores: list[int] = []

        for day in days:
            ns = day.daily_nutrition.total
            for k in total:
                total[k] += ns.get(k, 0.0)

            if day.daily_score:
                scores.append(day.daily_score)

        monthly_nutrition = NutritionSummary(
            total={k: round(v, 1) for k, v in total.items()},
            items_count=len(days),
        )

        # 4️⃣ 월 평균 점수
        if scores:
            monthly_score = round(sum(scores) / len(scores))
        else:
            monthly_score = 0

        monthly_grade = self._grade(monthly_score)

        return MonthlyMealSummary(
            user_id=user_id,
            month=f"{year}-{month:02d}",
            days=days,
            monthly_nutrition=monthly_nutrition,
            monthly_score=monthly_score,
            monthly_grade=monthly_grade,
        )

    # ----------------------------------
    def _grade(self, score: int) -> str:
        if score >= 85:
            return "우수"
        if score >= 70:
            return "보통"
        if score >= 55:
            return "주의"
        return "위험"
