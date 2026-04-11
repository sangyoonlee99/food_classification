from __future__ import annotations

from datetime import date
from typing import Dict, Optional

from infra.db_server import get_db_conn


# ==================================================
# 🏃 운동 이모지 매핑
# ==================================================
EXERCISE_EMOJI = {
    "러닝": "🏃",
    "조깅": "🏃",
    "걷기": "🚶",
    "자전거": "🚴",
    "사이클": "🚴",
    "헬스": "🏋️",
    "근력": "🏋️",
    "요가": "🧘",
    "필라테스": "🧘",
    "수영": "🏊",
}


def _with_emoji(name: str | None) -> str | None:
    if not name:
        return None
    for k, e in EXERCISE_EMOJI.items():
        if k in name:
            return f"{e} {name}"
    return name


class DailyExerciseFeedbackService:
    """
    홈 화면용 '오늘 운동 상태' 피드백 생성

    - 집계 테이블(daily_exercise_summary)만 사용
    - (옵션) 사용자 선호 운동 문구 반영
    - user_exercise_preference 테이블이 없을 수 있으므로 안전 처리
    """

    def build_today_feedback(self, *, user_id: bytes) -> Optional[Dict]:
        today = date.today()

        with get_db_conn() as conn:
            cur = conn.cursor()

            # 1) 오늘 운동 집계
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
                {"user_id": user_id, "d": today},
            )
            row = cur.fetchone()

            # 2) 선호 운동(옵션) - 테이블 없으면 무시
            pref = None
            try:
                cur.execute(
                    """
                    SELECT cardio_pref, strength_pref
                    FROM user_exercise_preference
                    WHERE user_id = :u
                    """,
                    {"u": user_id},
                )
                pref = cur.fetchone()
            except Exception:
                pref = None

        if not row:
            return None

        (
            total_minutes,
            total_met,
            cardio_minutes,
            strength_minutes,
            intensity_level,
        ) = row

        total_minutes = int(total_minutes or 0)
        cardio_minutes = int(cardio_minutes or 0)
        strength_minutes = int(strength_minutes or 0)
        total_met = float(total_met or 0.0)

        cardio_pref = _with_emoji(pref[0]) if pref and pref[0] else None
        strength_pref = _with_emoji(pref[1]) if pref and pref[1] else None

        # -------------------------
        # 상태 판단 (UI 기준)
        # -------------------------
        if total_minutes == 0:
            status = "none"
            if cardio_pref:
                message = f"오늘은 아직 운동 기록이 없어요. {cardio_pref}부터 가볍게 시작해보세요."
            elif strength_pref:
                message = f"오늘은 {strength_pref} 위주로 가볍게 시작해보는 건 어떨까요?"
            else:
                message = "오늘은 아직 운동 기록이 없어요."

        elif total_minutes < 20:
            status = "low"
            if cardio_pref and cardio_minutes == 0:
                message = f"{cardio_pref}를 10분 정도만 추가해도 좋아요."
            elif strength_pref and strength_minutes == 0:
                message = f"{strength_pref}을 가볍게 한 세트만 더 해보세요."
            else:
                message = "가벼운 활동을 하셨어요. 짧게라도 한 번 더 움직여볼까요?"

        elif total_minutes < 40:
            status = "normal"
            message = "오늘 운동량은 적절한 편이에요 👍"

        else:
            status = "good"
            message = "충분한 운동을 하셨어요! 아주 좋아요 💪"

        return {
            "total_minutes": total_minutes,
            "total_met": round(total_met, 1),
            "cardio_minutes": cardio_minutes,
            "strength_minutes": strength_minutes,
            "intensity": intensity_level,   # low | normal | high
            "status": status,               # none | low | normal | good
            "message": message,
            "preferred": {
                "cardio": cardio_pref,
                "strength": strength_pref,
            },
        }
