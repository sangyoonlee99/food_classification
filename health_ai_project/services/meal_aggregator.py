# services/meal_aggregator.py
# STEP 4-6: Daily → Weekly / Monthly 집계

from pathlib import Path
from datetime import date
import json
from typing import List

from common.schemas import (
    DailyMealSummary,
    WeeklyMealSummary,
    MonthlyMealSummary,
    NutritionSummary,
)


class MealAggregator:
    """
    DailyMealSummary들을 모아
    - WeeklyMealSummary
    - MonthlyMealSummary
    생성
    """

    # ----------------------------------
    # 공통 유틸
    # ----------------------------------
    def _load_daily(self, path: Path) -> DailyMealSummary:
        with open(path, "r", encoding="utf-8") as f:
            return DailyMealSummary(**json.load(f))

    def _sum_nutrition(self, summaries: List[DailyMealSummary]) -> NutritionSummary:
        total = {
            "kcal": 0.0,
            "carbs_g": 0.0,
            "protein_g": 0.0,
            "fat_g": 0.0,
        }
        items_count = 0

        for d in summaries:
            for k in total:
                total[k] += d.daily_nutrition["total"].get(k, 0.0)
            items_count += d.daily_nutrition.get("items_count", 0)

        return NutritionSummary(
            total={k: round(v, 1) for k, v in total.items()},
            items_count=items_count,
        )

    # ----------------------------------
    # 1️⃣ 주간 집계 (YYYY-MM-N주)
    # ----------------------------------
    def aggregate_weekly(
        self,
        daily_files: List[Path],
        user_id: int | None = None,
    ) -> WeeklyMealSummary:

        dailies = [self._load_daily(p) for p in daily_files]
        dailies.sort(key=lambda x: x.date)

        nutrition = self._sum_nutrition(dailies)

        avg_score = (
            sum(d.daily_score for d in dailies) // len(dailies)
            if dailies else 0
        )

        grade = self._grade(avg_score)

        return WeeklyMealSummary(
            user_id=user_id,
            week_start=dailies[0].date,
            week_end=dailies[-1].date,
            days=dailies,
            weekly_nutrition=nutrition,
            weekly_score=avg_score,
            weekly_grade=grade,
        )

    # ----------------------------------
    # 2️⃣ 월간 집계 (YYYY-MM)
    # ----------------------------------
    def aggregate_monthly(
        self,
        daily_files: List[Path],
        year: int,
        month: int,
        user_id: int | None = None,
    ) -> MonthlyMealSummary:

        dailies = [self._load_daily(p) for p in daily_files]
        dailies.sort(key=lambda x: x.date)

        nutrition = self._sum_nutrition(dailies)

        avg_score = (
            sum(d.daily_score for d in dailies) // len(dailies)
            if dailies else 0
        )

        grade = self._grade(avg_score)

        return MonthlyMealSummary(
            user_id=user_id,
            month=f"{year}-{month:02d}",
            days=dailies,
            monthly_nutrition=nutrition,
            monthly_score=avg_score,
            monthly_grade=grade,
        )

    # ----------------------------------
    # 등급 계산
    # ----------------------------------
    def _grade(self, score: int) -> str:
        if score >= 85:
            return "아주 좋음"
        if score >= 70:
            return "보통"
        if score >= 50:
            return "주의"
        return "개선 필요"
