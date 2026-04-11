# services/weekly_output_service.py
# STEP 6-4: 주간 요약 출력 전용 서비스

from typing import Dict
from datetime import date

from services.weekly_summary_service import WeeklySummaryService


class WeeklyOutputService:
    """
    주간 요약을 UI용 dict로 변환
    """

    def __init__(self):
        self.weekly_svc = WeeklySummaryService()

    def build(
        self,
        user_id: int,
        ref_date: date,
    ) -> Dict[str, any]:

        weekly = self.weekly_svc.build_weekly_summary(
            user_id=user_id,
            ref_date=ref_date,
        )

        return {
            "score": weekly.weekly_score,
            "grade": weekly.weekly_grade,
            "feedback": weekly.feedback,
            "nutrient_boost": weekly.nutrient_boost,
        }
