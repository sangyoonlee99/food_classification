# services/record_service.py
from __future__ import annotations
from typing import Dict, List

# --------------------------------------------------
# 임시 In-Memory Store (user_id 기준)
# --------------------------------------------------
_MEAL_LOGS: Dict[int, List[Dict]] = {}
_EXERCISE_LOGS: Dict[int, List[Dict]] = {}


# --------------------------------------------------
# 식사 기록
# --------------------------------------------------
def record_meal(
    user_id: int,
    carb: int,
    protein: int,
    fat: int,
    kcal: int,
):
    logs = _MEAL_LOGS.setdefault(user_id, [])
    logs.append({
        "carb": carb,
        "protein": protein,
        "fat": fat,
        "kcal": kcal,
    })


def get_meal_logs(user_id: int, days: int = 7) -> List[Dict]:
    return _MEAL_LOGS.get(user_id, [])[-days:]


# --------------------------------------------------
# 운동 기록
# --------------------------------------------------
def record_exercise(
    user_id: int,
    cardio_min: int = 0,
    strength_min: int = 0,
):
    logs = _EXERCISE_LOGS.setdefault(user_id, [])
    logs.append({
        "cardio": cardio_min,
        "strength": strength_min,
    })


def get_exercise_logs(user_id: int, days: int = 7) -> List[Dict]:
    return _EXERCISE_LOGS.get(user_id, [])[-days:]
