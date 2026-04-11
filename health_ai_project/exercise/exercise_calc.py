# exercise/exercise_calc.py

"""
Exercise MET Calculator (MVP)

- 표준 MET 범위에서 대표값 선택
- (선택) 시간/체중 기반 소모 칼로리 추정
- 순수 함수 모음

"""

from typing import Tuple, Literal, Optional
from exercise.exercise_rules import get_met_range

IntensityLevel = Literal["low", "mid", "high"]


# =====================================================
# MET 선택 로직
# =====================================================
def select_representative_met(
    met_range: Tuple[float, float],
    method: Literal["mid", "low", "high"] = "mid",
) -> float:
    """
    MET 범위에서 대표값 선택
    - mid: 평균값
    - low: 하한
    - high: 상한
    """
    low, high = met_range
    if method == "low":
        return low
    if method == "high":
        return high
    return round((low + high) / 2, 2)


def get_exercise_met(
    exercise_type: str,
    intensity: IntensityLevel,
    select_method: Literal["mid", "low", "high"] = "mid",
) -> float:
    """
    운동 종류 + 강도 → 대표 MET 반환
    """
    met_range = get_met_range(exercise_type, intensity)
    return select_representative_met(met_range, select_method)


# =====================================================
# 칼로리 추정 (선택 기능)
# =====================================================
def estimate_calories(
    met: float,
    minutes: int,
    weight_kg: Optional[float] = None,
) -> float:
    """
    MET 기반 칼로리 추정

    공식(표준):
    kcal = MET × 체중(kg) × 시간(h)

    - weight_kg가 없으면 체중 60kg 가정
    """
    weight = weight_kg if weight_kg and weight_kg > 0 else 60.0
    hours = max(1, minutes) / 60.0
    return round(met * weight * hours, 1)
