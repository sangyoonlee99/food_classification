# services/daily_output_loader.py
from __future__ import annotations

from datetime import date
from typing import Dict

from services.daily_output_service import DailyOutputService

# ✅ 실제 존재 & 역할 일치
from services.daily_goal_service import DailyGoalService
from services.diet_service import DietService
from services.exercise_adapter import ExerciseAdapter

# (이벤트는 선택 – 없으면 None 처리)


class DailyOutputLoader:
    """
    UI 전용 Facade
    - 기존 Domain / Service 절대 수정 안 함
    - Loader는 '조립'만 담당
    """

    def __init__(self):
        self.goal_svc = DailyGoalService()
        self.diet_svc = DietService()
        self.exercise_adapter = ExerciseAdapter()
        self.output_svc = DailyOutputService()

    def build_today(self, *, user_id: bytes) -> Dict[str, str]:
        today = date.today()

        goal = self.goal_svc.build_today(user_id=user_id)
        diet = self.diet_svc.get_daily_summary(user_id=user_id, date=today)
        exercise = self.exercise_adapter.get_daily_plan(
            user_id=user_id,
            date=today,
        )

        # 데이터 부족 시 UI 가드
        if not goal or not diet or not exercise:
            return {}

        return self.output_svc.build(
            goal=goal,
            diet=diet,
            exercise=exercise,
            event=None,   # 이벤트 없으면 None
        )
