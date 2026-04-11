from __future__ import annotations

from datetime import date
from typing import Optional, Dict

from infra.db_server import get_db_conn
from services.exercise_service import get_today_exercise_summary


# ==================================================
# 🏃 운동 이모지 매핑
# ==================================================
EXERCISE_EMOJI = {

    "걷기": "🚶",
    "러닝": "🏃",
    "러닝머신": "🏃",
    "자전거": "🚴",
    "사이클": "🚴",
    "수영": "🏊",
    "조깅": "🏃",
    "필라테스": "🧘",
    "요가": "🧘",
    "헬스": "🏋️",
    "홈트레이능": "🏋️",
    "크로스핏": "🏋️",
    "근력운동": "🏋️",

}


def _with_emoji(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    for k, e in EXERCISE_EMOJI.items():
        if k in name:
            return f"{e} {name}"
    return name


class DailyExerciseGuideService:
    """
    오늘 운동 가이드 카드 생성 (최종 UX 확정판)

    ✅ message는 문자열 하나
    ✅ 문장 + 빈 줄 + 행동 가이드 구조
    """

    def build_today_guide(self, *, user_id: bytes) -> Optional[Dict]:
        with get_db_conn() as conn:
            cur = conn.cursor()

            # 나이
            cur.execute(
                "SELECT birth_year FROM user_profile WHERE user_id = :u",
                {"u": user_id},
            )
            r = cur.fetchone()
            age = date.today().year - int(r[0]) if r and r[0] else None

            # 목표 타입
            cur.execute(
                """
                SELECT goal_type
                FROM user_goal
                WHERE user_id = :u
                  AND is_active = 'Y'
                """,
                {"u": user_id},
            )
            g = cur.fetchone()
            goal_type = g[0] if g else "maintenance"

            # 선호 운동
            cur.execute(
                """
                SELECT preferred_cardio, preferred_strength
                FROM user_profile
                WHERE user_id = :u
                """,
                {"u": user_id},
            )
            p = cur.fetchone()

        cardio_pref = _with_emoji(p[0]) if p and p[0] else None
        strength_pref = _with_emoji(p[1]) if p and p[1] else None

        # 오늘 운동 요약
        summary = get_today_exercise_summary(user_id=user_id)
        total_minutes = summary["total_minutes"] if summary else 0
        intensity = summary["intensity_level"] if summary else "low"

        # 목표 운동 시간
        if goal_type == "weight_loss":
            target_min = 40
        elif goal_type == "weight_gain":
            target_min = 30
        else:
            target_min = 25

        # -------------------------------------------------
        # 메시지 생성 (🔥 줄바꿈 보장 🔥)
        # -------------------------------------------------
        if total_minutes == 0:
            status = "none"
            main = "오늘은 아직 운동을 하지 않았어요."

            if cardio_pref and strength_pref:
                action = (
                    f" {cardio_pref} {target_min}분 또는 "
                    f"{strength_pref} {max(20, target_min - 10)}분 중 하나를 목표로 해보세요."
                )
            elif cardio_pref:
                action = f" {cardio_pref} {target_min}분을 목표로 시작해보세요."
            elif strength_pref:
                action = f" {strength_pref} {max(20, target_min - 10)}분을 목표로 해보세요."
            else:
                action = f" 가벼운 활동 {target_min}분을 목표로 시작해보세요."

        elif total_minutes < target_min * 0.7:
            status = "low"
            main = "운동량이 조금 부족해요."
            if cardio_pref:
                action = f" {cardio_pref}로 10~15분만 더 해볼까요?"
            elif strength_pref:
                action = f" {strength_pref} 한 세트만 추가해도 좋아요."
            else:
                action = "가볍게라도 조금 더 움직여볼까요?"

        elif total_minutes <= target_min * 1.3:
            status = "good"
            main = "오늘 운동량은 아주 적절해요 👍"
            action = "이 흐름을 유지해보세요."

        else:
            status = "over"
            main = "오늘은 충분히 운동했어요."
            action = "무리하지 말고 회복에 집중하세요."

        # ✅ 핵심: 문자열 하나 + \n\n
        message = f"{main}\n\n{action}"

        # 심박수 가이드
        hr_range = None
        if age:
            max_hr = 220 - age
            if intensity == "high":
                hr_range = (int(max_hr * 0.75), int(max_hr * 0.85))
            elif intensity == "normal":
                hr_range = (int(max_hr * 0.60), int(max_hr * 0.75))
            else:
                hr_range = (int(max_hr * 0.50), int(max_hr * 0.60))

        return {
            "target_minutes": target_min,
            "today_minutes": total_minutes,
            "intensity": intensity,
            "status": status,
            "message": message,   # ✅ UI에서 그대로 출력
            "heart_rate": hr_range,
            "preferred": {
                "cardio": cardio_pref,
                "strength": strength_pref,
            },
        }
