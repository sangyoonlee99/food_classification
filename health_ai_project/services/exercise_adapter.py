# services/exercise_adapter.py
from __future__ import annotations

from datetime import date
from typing import Optional, List
import json

from common.schemas import DailyExercisePlan, ExerciseItem
from infra.db_server import get_db_conn
from infra.repositories.user_repository import UserRepository


class ExerciseAdapter:
    def __init__(self):
        self.user_repo = UserRepository()

    def get_daily_plan(
        self,
        *,
        user_id: bytes,
        date: date,
    ) -> Optional[DailyExercisePlan]:

        profile = self.user_repo.get_user_profile(user_id=user_id) or {}
        prefs = profile.get("preferences") or {}

        preferred_cardio = prefs.get("preferred_cardio", [])
        preferred_strength = prefs.get("preferred_strength", [])

        with get_db_conn() as conn:
            cur = conn.cursor()

            # 요약
            cur.execute(
                """
                SELECT total_minutes, total_met
                FROM daily_exercise_summary
                WHERE user_id = :u AND summary_date = :d
                """,
                {"u": user_id, "d": date},
            )
            summary = cur.fetchone()
            if not summary:
                return None

            total_minutes, total_met = summary

            # 개별 기록
            cur.execute(
                """
                SELECT exercise_type, minutes, met
                FROM exercise_record
                WHERE user_id = :u
                  AND TRUNC(performed_at) = :d
                """,
                {"u": user_id, "d": date},
            )
            rows = cur.fetchall()

        items: List[ExerciseItem] = []
        total_calorie = 0.0

        for name, minutes, met in rows:
            calorie = float(minutes or 0) * float(met or 0)
            total_calorie += calorie

            category = (
                "cardio"
                if name in ("걷기", "조깅", "러닝", "자전거")
                else "strength"
            )

            items.append(
                ExerciseItem(
                    name=name,
                    category=category,
                    minutes=int(minutes or 0),
                    met_value=float(met or 0),
                    calorie_burn=round(calorie, 1),
                )
            )

        # 🔥 선호 운동 우선 정렬
        def pref_rank(item: ExerciseItem) -> int:
            if item.category == "cardio" and item.name in preferred_cardio:
                return 0
            if item.category == "strength" and item.name in preferred_strength:
                return 0
            return 1

        items.sort(key=pref_rank)

        return DailyExercisePlan(
            user_id=user_id,
            date=date,
            exercises=items,
            total_minutes=int(total_minutes or 0),
            total_calorie_burn=round(total_calorie, 1),
        )
