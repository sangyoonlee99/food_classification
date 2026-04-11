from infra.db_server import get_db_conn
from datetime import date

class DailyMealFeedbackService:
    """
    식사 기록 직후 UI에 보여줄 요약/피드백 생성
    """

    def build_today_feedback(self, *, user_id: bytes) -> dict:
        today = date.today()

        with get_db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT total_kcal, daily_score, daily_grade
                FROM daily_meal_summary
                WHERE user_id = :user_id
                  AND summary_date = :d
                """,
                {"user_id": user_id, "d": today},
            )
            row = cur.fetchone()

        if not row:
            return {}

        total_kcal, score, grade = row

        if score >= 85:
            status = "good"
            message = "👍 잘 관리되고 있어요. 현재 페이스 유지하세요."
        elif score >= 65:
            status = "warning"
            message = "⚠️ 조금만 조절하면 더 좋아질 수 있어요."
        else:
            status = "danger"
            message = "🚨 섭취량 조절이 필요해요."

        return {
            "total_kcal": total_kcal,
            "score": score,
            "grade": grade,
            "status": status,
            "message": message,
        }
