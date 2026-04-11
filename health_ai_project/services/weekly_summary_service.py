# services/weekly_summary_service.py
# STEP 4-4: 주간 식사 + 운동 요약 생성 서비스 (최종)

from __future__ import annotations

from datetime import date, timedelta
from typing import List

from infra.db_server import get_db_conn
from common.schemas import (
    DailyMealSummary,
    WeeklyMealSummary,
    NutritionSummary,
)

from services.feedback_service import FeedbackService
from services.nutrient_boost_service import NutrientBoostService
from services.meal_service import MealService


def _week_range(ref_date: date) -> tuple[date, date]:
    start = ref_date - timedelta(days=ref_date.weekday())
    end = start + timedelta(days=6)
    return start, end


class WeeklySummaryService:
    """
    한 주치 DailyMealSummary + ExerciseSummary → WeeklyMealSummary 생성
    """

    def __init__(self):
        self.meal_service = MealService()
        self.feedback_svc = FeedbackService()
        self.boost_svc = NutrientBoostService()

    # -------------------------------------------------
    def build_weekly_summary(
        self,
        user_id: int,
        ref_date: date,
    ) -> WeeklyMealSummary:

        week_start, week_end = _week_range(ref_date)

        # =================================================
        # 1️⃣ 일별 식사 로드
        # =================================================
        days: List[DailyMealSummary] = self.meal_service.load_daily_range(
            user_id=user_id,
            start=week_start,
            end=week_end,
        )

        # =================================================
        # 2️⃣ 주간 영양 합산
        # =================================================
        total = {"kcal": 0.0, "carbs_g": 0.0, "protein_g": 0.0, "fat_g": 0.0}
        scores: List[int] = []

        for d in days:
            for k in total:
                total[k] += float(d.daily_nutrition.total.get(k, 0.0))
            if d.daily_score is not None:
                scores.append(d.daily_score)

        weekly_nutrition = NutritionSummary(
            total={k: round(v, 1) for k, v in total.items()},
            items_count=len(days),
        )

        weekly_score = round(sum(scores) / len(scores)) if scores else 0
        weekly_grade = self._grade(weekly_score)

        # =================================================
        # 3️⃣ ✅ 주간 운동 요약 (신규)
        # =================================================
        weekly_exercise = {
            "total_minutes": 0,
            "cardio_minutes": 0,
            "strength_minutes": 0,
        }

        with get_db_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT
                    NVL(SUM(total_minutes), 0),
                    NVL(SUM(cardio_minutes), 0),
                    NVL(SUM(strength_minutes), 0)
                FROM daily_exercise_summary
                WHERE user_id = :user_id
                  AND summary_date BETWEEN :start_date AND :end_date
            """, {
                "user_id": user_id,
                "start_date": week_start,
                "end_date": week_end,
            })

            row = cur.fetchone()
            if row:
                weekly_exercise = {
                    "total_minutes": int(row[0] or 0),
                    "cardio_minutes": int(row[1] or 0),
                    "strength_minutes": int(row[2] or 0),
                }

        # =================================================
        # 4️⃣ 주간 피드백
        # =================================================
        raw_feedback = self.feedback_svc.weekly_feedback(
            score=weekly_score,
            nutrition=weekly_nutrition,
        )

        used_feedback = self.meal_service.load_used_messages(
            user_id=user_id,
            scope="weekly_feedback",
            day=week_start,
        )

        feedback = [m for m in raw_feedback if m not in used_feedback]
        feedback = feedback or raw_feedback

        self.meal_service.save_used_messages(
            user_id=user_id,
            scope="weekly_feedback",
            day=week_start,
            messages=feedback,
        )

        # =================================================
        # 5️⃣ 부족 영양소 보완
        # =================================================
        raw_boost = self.boost_svc.recommend(weekly_nutrition)

        if isinstance(raw_boost, dict) and "message" in raw_boost:
            boost_messages = [raw_boost["message"]]
        elif isinstance(raw_boost, list):
            boost_messages = raw_boost
        else:
            boost_messages = []

        used_boost = self.meal_service.load_used_messages(
            user_id=user_id,
            scope="weekly_boost",
            day=week_start,
        )

        nutrient_boost = [m for m in boost_messages if m not in used_boost]
        nutrient_boost = nutrient_boost or boost_messages

        self.meal_service.save_used_messages(
            user_id=user_id,
            scope="weekly_boost",
            day=week_start,
            messages=nutrient_boost,
        )

        # =================================================
        # 6️⃣ 결과 반환
        # =================================================
        return WeeklyMealSummary(
            user_id=user_id,
            week_start=week_start,
            week_end=week_end,
            days=days,
            weekly_nutrition=weekly_nutrition,
            weekly_score=weekly_score,
            weekly_grade=weekly_grade,
            feedback=feedback,
            nutrient_boost=nutrient_boost,
            weekly_exercise=weekly_exercise,  # ✅ 핵심
        )

    # -------------------------------------------------
    def _grade(self, score: int) -> str:
        if score >= 85:
            return "우수"
        if score >= 70:
            return "보통"
        if score >= 55:
            return "주의"
        return "위험"
