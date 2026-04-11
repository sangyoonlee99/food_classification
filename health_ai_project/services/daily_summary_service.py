# services/daily_summary_service.py
# 하루 식사 요약 + 하루 운동 요약 조회 서비스 (최종)

from __future__ import annotations

from typing import List, Optional
from datetime import date
from pathlib import Path
import json

from common.schemas import (
    MealRecord,
    DailyMealSummary,
    NutritionSummary,
)
from config import BASE_DIR
from infra.db_server import get_db_conn


class DailySummaryService:
    """
    ✔ 하루치 MealRecord → DailyMealSummary 계산
    ✔ JSON 기반 식사 요약 조회
    ✔ DB 기반 운동 요약 조회
    """

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or (BASE_DIR / "data" / "meal_logs")

    # =================================================
    # 1️⃣ Daily 식사 요약 조회 (기존)
    # =================================================
    def get_daily(
        self,
        user_id: int | str,
        day: date,
    ) -> DailyMealSummary | None:

        path = self.base_dir / f"user_{user_id}" / f"{day}.json"
        if not path.exists():
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                return DailyMealSummary(**json.load(f))
        except Exception:
            return None

    # =================================================
    # 2️⃣ Daily 식사 요약 계산 (집계 전용)
    # =================================================
    def build_daily_summary(
        self,
        meals: List[MealRecord],
        summary_date: date,
        user_id: int | None = None,
    ) -> DailyMealSummary:

        daily_total = {
            "kcal": 0.0,
            "carbs_g": 0.0,
            "protein_g": 0.0,
            "fat_g": 0.0,
        }

        scores: list[int] = []

        for meal in meals:
            ns = meal.nutrition_summary.total
            for k in daily_total:
                daily_total[k] += float(ns.get(k, 0.0))

            if meal.meal_evaluation:
                scores.append(int(meal.meal_evaluation.meal_score))

        daily_nutrition = NutritionSummary(
            total={k: round(v, 1) for k, v in daily_total.items()},
            items_count=len(meals),
        )

        daily_score = round(sum(scores) / len(scores)) if scores else 0
        daily_grade = self._grade(daily_score)

        return DailyMealSummary(
            user_id=user_id,
            date=summary_date,
            meals=meals,
            daily_nutrition=daily_nutrition,
            daily_score=daily_score,
            daily_grade=daily_grade,
        )

    # =================================================
    # 3️⃣ Daily 운동 요약 조회 (✅ 추가)
    # =================================================
    def get_daily_exercise(
        self,
        user_id: bytes,
        day: date,
    ) -> Optional[dict]:
        """
        DAILY_EXERCISE_SUMMARY 테이블에서 하루 운동 요약 조회
        Home / Report 화면용
        """
        with get_db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT
                    total_minutes,
                    total_met,
                    cardio_minutes,
                    strength_minutes,
                    intensity_level
                FROM daily_exercise_summary
                WHERE user_id = :user_id
                  AND summary_date = :d
                """,
                {
                    "user_id": user_id,
                    "d": day,
                },
            )

            row = cur.fetchone()
            if not row:
                return None

            return {
                "total_minutes": row[0],
                "total_met": row[1],
                "cardio_minutes": row[2],
                "strength_minutes": row[3],
                "intensity_level": row[4],
            }

    # =================================================
    # 점수 → 등급
    # =================================================
    def _grade(self, score: int) -> str:
        if score >= 85:
            return "아주 좋음"
        if score >= 70:
            return "보통"
        if score >= 55:
            return "주의"
        return "개선 필요"
