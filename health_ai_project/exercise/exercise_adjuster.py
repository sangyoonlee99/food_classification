# exercise/exercise_adjuster.py
"""
운동 조정 로직 (Rule-based MVP)

역할:
- planner가 만든 운동 계획을
- 이벤트 / 정체 상태에 따라 조정
- 서비스 전반에서 공통으로 사용할 '운동 상태(status)'를 결정

"""

from typing import Dict, List, Optional


def adjust_exercise(
    planned_exercise: Dict,
    events: Optional[List[Dict]] = None,
    plateau_days: Optional[int] = None,
) -> Dict:
    """
    Args:
        planned_exercise: {
            "exercise_type": str,
            "minutes": int,
            "intensity": "low" | "mid" | "high"
        }
        events: [{"type": "fatigue" | "sick" | ...}, ...]
        plateau_days: 체중 정체 일수

    Returns:
        {
            "status": "applied" | "partial" | "skipped",
            "exercise_type": str,
            "adjusted_minutes": int,
            "intensity": "keep" | "down" | "up",
            "reason": str,
        }
    """

    # 기본값 안전 처리
    events = events or []
    plateau_days = plateau_days or 0

    exercise_type = planned_exercise.get("exercise_type")
    base_minutes = int(planned_exercise.get("minutes", 0))

    event_types = {e.get("type") for e in events}

    # --------------------------------------------------
    # 1️⃣ sick 이벤트 (최우선)
    # --------------------------------------------------
    if "sick" in event_types:
        return {
            "status": "skipped",
            "exercise_type": exercise_type,
            "adjusted_minutes": 0,
            "intensity": "down",
            "reason": "컨디션 회복을 위해 오늘은 운동을 쉬어요",
        }

    # --------------------------------------------------
    # 2️⃣ fatigue 이벤트
    # --------------------------------------------------
    if "fatigue" in event_types:
        adjusted = max(10, int(base_minutes * 0.5)) if base_minutes > 0 else 0
        return {
            "status": "partial",
            "exercise_type": exercise_type,
            "adjusted_minutes": adjusted,
            "intensity": "down",
            "reason": "피로도를 고려해 운동량을 줄였어요",
        }

    # --------------------------------------------------
    # 3️⃣ 체중 정체 (plateau)
    # --------------------------------------------------
    if plateau_days >= 7:
        return {
            "status": "applied",
            "exercise_type": exercise_type,
            "adjusted_minutes": base_minutes + 10,
            "intensity": "up",
            "reason": "체중 정체 개선을 위해 활동량을 조금 늘렸어요",
        }

    # --------------------------------------------------
    # 4️⃣ 정상 상태
    # --------------------------------------------------
    return {
        "status": "applied",
        "exercise_type": exercise_type,
        "adjusted_minutes": base_minutes,
        "intensity": "keep",
        "reason": "계획된 운동을 그대로 진행해요",
    }
