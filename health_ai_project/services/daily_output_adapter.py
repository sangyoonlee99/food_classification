from __future__ import annotations

from datetime import date
from typing import Dict, Any

from infra.db_server import get_db_conn

from common.schemas import (
    UserGoal,
    DailyDietPlan,
    DailyExercisePlan,
    Event,
)

from services.daily_output_service import DailyOutputService


class DailyOutputAdapter:
    """
    DB 기반 데이터를
    DailyOutputService가 요구하는 도메인 객체로 변환하는 어댑터
    """

    def __init__(self):
        self.svc = DailyOutputService()

    def build_for_today(
        self,
        *,
        user_id: bytes,
        today: date,
    ) -> Dict[str, Any]:

        goal = self._load_goal(user_id)
        diet = self._load_daily_diet(user_id, today)
        exercise = self._load_daily_exercise(user_id, today)
        event = self._load_today_event(user_id, today)

        if not goal or not diet or not exercise:
            return {}

        return self.svc.build(
            goal=goal,
            diet=diet,
            exercise=exercise,
            event=event,
        )

    # -------------------------------------------------
    # Loaders
    # -------------------------------------------------
    def _load_goal(self, user_id: bytes) -> UserGoal | None:
        with get_db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT
                    goal_type,
                    kcal_target,
                    macro_target
                FROM user_goal
                WHERE user_id = :uid
                  AND is_active = 'Y'
                """,
                {"uid": user_id},
            )
            row = cur.fetchone()

        if not row:
            return None

        goal_type, kcal, macro_json = row

        return UserGoal(
            goal_type=goal_type,
            kcal_target=kcal,
            macro_target=macro_json,
        )

    def _load_daily_diet(
        self,
        user_id: bytes,
        target_date: date,
    ) -> DailyDietPlan | None:

        with get_db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT
                    total_kcal,
                    carb_g,
                    protein_g,
                    fat_g
                FROM daily_meal_summary
                WHERE user_id = :uid
                  AND summary_date = :d
                """,
                {"uid": user_id, "d": target_date},
            )
            row = cur.fetchone()

        if not row:
            return None

        kcal, carb, protein, fat = row

        return DailyDietPlan(
            total_kcal=kcal or 0,
            carb_g=carb or 0,
            protein_g=protein or 0,
            fat_g=fat or 0,
        )

    def _load_daily_exercise(
        self,
        user_id: bytes,
        target_date: date,
    ) -> DailyExercisePlan | None:

        with get_db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT
                    total_minutes,
                    intensity_level
                FROM daily_exercise_summary
                WHERE user_id = :uid
                  AND summary_date = :d
                """,
                {"uid": user_id, "d": target_date},
            )
            row = cur.fetchone()

        if not row:
            return None

        minutes, intensity = row

        return DailyExercisePlan(
            total_minutes=minutes or 0,
            intensity=intensity or "low",
        )

    def _load_today_event(
        self,
        user_id: bytes,
        target_date: date,
    ) -> Event | None:

        with get_db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT event_type, severity, note
                FROM event_log
                WHERE user_id = :uid
                  AND event_date = :d
                """,
                {"uid": user_id, "d": target_date},
            )
            row = cur.fetchone()

        if not row:
            return None

        etype, severity, note = row

        return Event(
            event_type=etype,
            severity=severity,
            note=note,
        )
