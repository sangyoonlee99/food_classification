# exercise/exercise_rules.py

"""
Exercise Rules (MVP)

- 사용자 선호 운동
- 운동 강도 레벨
- 평소 활동량
- 운동 종류별 표준 MET 범위

"""

from typing import Dict, List, Tuple, Literal


# =====================================================
# 타입 정의
# =====================================================
IntensityLevel = Literal["low", "mid", "high"]
ActivityLevel = Literal["low", "mid", "high"]


# =====================================================
# 사용자 입력 기본값
# =====================================================
DEFAULT_PREFERRED_EXERCISES: List[str] = [
    "걷기",
    "스트레칭",
]

DEFAULT_INTENSITY_LEVEL: IntensityLevel = "mid"
DEFAULT_ACTIVITY_LEVEL: ActivityLevel = "mid"


# =====================================================
# 운동 종류별 MET 범위 (표준 기반)
# (Compendium of Physical Activities 참고)
# =====================================================
EXERCISE_MET_TABLE: Dict[str, Dict[IntensityLevel, Tuple[float, float]]] = {
    "걷기": {
        "low": (2.0, 2.8),
        "mid": (3.0, 3.8),
        "high": (4.0, 4.8),
    },
    "조깅": {
        "low": (6.0, 6.8),
        "mid": (7.0, 7.8),
        "high": (8.0, 9.0),
    },
    "자전거": {
        "low": (4.0, 5.0),
        "mid": (6.0, 7.0),
        "high": (8.0, 10.0),
    },
    "근력운동": {
        "low": (3.0, 4.0),
        "mid": (4.5, 6.0),
        "high": (6.0, 8.0),
    },
    "스트레칭": {
        "low": (2.0, 2.5),
        "mid": (2.5, 3.0),
        "high": (3.0, 3.5),
    },
}


# =====================================================
# 활동량 기반 시간 가중치
# =====================================================
ACTIVITY_TIME_MULTIPLIER: Dict[ActivityLevel, float] = {
    "low": 0.8,
    "mid": 1.0,
    "high": 1.2,
}


# =====================================================
# Helper Functions
# =====================================================
def get_supported_exercises() -> List[str]:
    """지원하는 운동 목록 반환"""
    return list(EXERCISE_MET_TABLE.keys())


def get_default_user_exercise_profile() -> Dict[str, object]:
    """신규 사용자 기본 운동 프로필"""
    return {
        "preferred_exercises": DEFAULT_PREFERRED_EXERCISES,
        "intensity_level": DEFAULT_INTENSITY_LEVEL,
        "activity_level": DEFAULT_ACTIVITY_LEVEL,
    }


def get_met_range(
    exercise_type: str,
    intensity: IntensityLevel,
) -> Tuple[float, float]:
    """
    운동 종류 + 강도 → MET 범위 반환
    """
    if exercise_type not in EXERCISE_MET_TABLE:
        raise ValueError(f"Unsupported exercise type: {exercise_type}")

    return EXERCISE_MET_TABLE[exercise_type][intensity]


def apply_activity_multiplier(
    base_minutes: int,
    activity_level: ActivityLevel,
) -> int:
    """
    활동량 레벨에 따른 권장 시간 보정
    """
    factor = ACTIVITY_TIME_MULTIPLIER.get(activity_level, 1.0)
    return max(5, int(base_minutes * factor))
