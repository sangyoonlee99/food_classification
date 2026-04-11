# services/daily_diet_loader.py
from __future__ import annotations

from datetime import date
from infra.db_server import get_db_conn
from common.schemas import UserProfile, UserGoal
from services.diet_service import DietService


class DailyDietLoader:
    """
    UI 전용: 오늘 식사 가이드 로더
    """

    def __init__(self):
        self.diet_svc = DietService()

    def load_today(self, *, user_id: bytes):
        # 1️⃣ profile
        with get_db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT sex, birth_year, height_cm, weight_kg_baseline, activity_level
                FROM user_profile
                WHERE user_id = :user_id
                """,
                {"user_id": user_id},
            )
            p = cur.fetchone()

        if not p:
            return None

        profile = UserProfile(
            user_id=user_id,
            sex=p[0],
            birth_year=p[1],
            height_cm=p[2],
            weight_kg_baseline=p[3],
            activity_level=p[4],
        )

        # 2️⃣ goal
        with get_db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT goal_type, kcal_target
                FROM user_goal
                WHERE user_id = :user_id
                  AND is_active = 'Y'
                """,
                {"user_id": user_id},
            )
            g = cur.fetchone()

        if not g:
            return None

        goal = UserGoal(
            user_id=user_id,
            goal_type=g[0],
            kcal_target=g[1],
        )

        # 3️⃣ diet plan
        return self.diet_svc.generate_today_diet(
            profile=profile,
            goal=goal,
        )
