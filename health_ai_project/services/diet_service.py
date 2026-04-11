# services/diet_service.py

from datetime import date
from common.schemas import UserProfile, UserGoal
from diet.daily_diet_planner import generate_daily_diet


class DietService:
    """
    Daily Diet Plan 서비스 진입점
    (FastAPI / Batch / Scheduler 공용)
    """

    def generate_today_diet(
        self,
        profile: UserProfile,
        goal: UserGoal,
        elderly_mode: bool = False,
    ):
        return generate_daily_diet(
            profile=profile,
            goal=goal,
            date_value=date.today(),
            elderly_mode=elderly_mode,
        )
    def get_daily_summary(self, *, user_id: bytes, date):
        """
        오늘 식단 요약 (임시)
        아직 meal_log 미연결 단계이므로 None 반환
        """
        return None