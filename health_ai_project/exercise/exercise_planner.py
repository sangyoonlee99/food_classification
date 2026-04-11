# exercise/exercise_planner.py

"""
Exercise Planner (MVP)

- 사용자 선호 운동 기반 추천
- 강도/활동량 반영
- 이벤트/정체에 따른 최소 조정
- '오늘 할 수 있는 운동' 가이드 생성

"""

from typing import Dict, List, Optional
from exercise.exercise_rules import (
    get_supported_exercises,
    apply_activity_multiplier,
    DEFAULT_PREFERRED_EXERCISES,
    DEFAULT_INTENSITY_LEVEL,
    DEFAULT_ACTIVITY_LEVEL,
)


# =====================================================
# 기본 추천 템플릿 (분)
# =====================================================
BASE_MINUTES_BY_INTENSITY = {
    "low": 20,
    "mid": 30,
    "high": 40,
}

STRETCH_MINUTES = 10


# =====================================================
# Planner
# =====================================================
def plan_daily_exercise(
    user_profile: Dict[str, object],
    events: Optional[List[str]] = None,
    plateau_days: int = 0,
) -> Dict[str, object]:
    """
    오늘의 운동 가이드 생성

    Args:
        user_profile:
            {
              "preferred_exercises": [str],
              "intensity_level": "low|mid|high",
              "activity_level": "low|mid|high"
            }
        events: ["sick", "fatigue", ...]
        plateau_days: 정체 일수

    Returns:
        {
          "status": "applied|partial|skipped",
          "recommendations": [
              {"exercise": str, "minutes": int, "note": str}
          ],
          "notes": [str]
        }
    """

    events = events or []

    # -------------------------------
    # 1) 운동 스킵 조건
    # -------------------------------
    if "sick" in events:
        return {
            "status": "skipped",
            "recommendations": [],
            "notes": ["컨디션 이슈로 오늘은 운동을 쉬는 것을 권장합니다."],
        }

    # -------------------------------
    # 2) 사용자 입력 정리
    # -------------------------------
    preferred = user_profile.get(
        "preferred_exercises", DEFAULT_PREFERRED_EXERCISES
    )
    intensity = user_profile.get(
        "intensity_level", DEFAULT_INTENSITY_LEVEL
    )
    activity = user_profile.get(
        "activity_level", DEFAULT_ACTIVITY_LEVEL
    )

    supported = set(get_supported_exercises())
    preferred = [e for e in preferred if e in supported]

    if not preferred:
        preferred = DEFAULT_PREFERRED_EXERCISES

    # -------------------------------
    # 3) 기본 운동 시간 산정
    # -------------------------------
    base_minutes = BASE_MINUTES_BY_INTENSITY.get(intensity, 30)
    minutes = apply_activity_multiplier(base_minutes, activity)

    # -------------------------------
    # 4) 정체/이벤트에 따른 미세 조정
    # -------------------------------
    notes: List[str] = []
    status = "applied"

    if plateau_days >= 7:
        minutes += 5
        notes.append("최근 정체가 있어 운동 시간을 소폭 늘렸습니다.")

    if "fatigue" in events:
        minutes = max(15, minutes - 10)
        status = "partial"
        notes.append("피로 이벤트로 운동 강도를 조정했습니다.")

    # -------------------------------
    # 5) 추천 리스트 구성
    # -------------------------------
    main_exercise = preferred[0]
    recommendations = [
        {
            "exercise": main_exercise,
            "minutes": minutes,
            "note": "선호 운동 기반 추천",
        },
        {
            "exercise": "스트레칭",
            "minutes": STRETCH_MINUTES,
            "note": "마무리 스트레칭",
        },
    ]

    return {
        "status": status,
        "recommendations": recommendations,
        "notes": notes,
    }
