# services/exercise_replan_service.py
from __future__ import annotations

from typing import List, Dict, Any, Optional
from copy import deepcopy

from common.schemas import ExerciseItem


class ExerciseReplanService:
    """
    이벤트 정책 기반 운동 replan (보조 로직)
    - policy 형태 예:
      {"same_day": {"skip": True}, "next_day": {"cardio_min": +20, "strength_min": +10}}
    """

    def replan(
        self,
        base_plan: List[ExerciseItem],
        event_type: str,
        intensity: Optional[str] = None,
        policy: Optional[Dict[str, Any]] = None,
        apply_scope: str = "next_day",
    ) -> Dict[str, Any]:

        notes: List[str] = []
        exercises = deepcopy(base_plan) if isinstance(base_plan, list) else []

        if not policy:
            return {"exercises": exercises, "notes": ["정책(policy)이 없어 운동 조정이 적용되지 않았습니다."]}

        scope = policy.get(apply_scope, {}) if isinstance(policy, dict) else {}
        if not scope:
            return {"exercises": exercises, "notes": [f"{apply_scope}에 적용할 운동 정책이 없습니다."]}

        if scope.get("skip") is True:
            notes.append("이벤트 당일 운동은 스킵(정책).")
            return {"exercises": exercises, "notes": notes}

        cardio_min = int(scope.get("cardio_min", 0) or 0)
        strength_min = int(scope.get("strength_min", 0) or 0)

        if cardio_min > 0:
            exercises.append(
                ExerciseItem(
                    name="추가 유산소(가이드)",
                    category="cardio",
                    minutes=cardio_min,
                    intensity=scope.get("intensity", "medium"),
                    met_value=0.0,
                    calorie_burn=0.0,
                )
            )
            notes.append(f"유산소 {cardio_min}분 추가(가이드)")

        if strength_min > 0:
            exercises.append(
                ExerciseItem(
                    name="추가 근력(가이드)",
                    category="strength",
                    minutes=strength_min,
                    intensity=scope.get("intensity", "medium"),
                    met_value=0.0,
                    calorie_burn=0.0,
                )
            )
            notes.append(f"근력 {strength_min}분 추가(가이드)")

        return {"exercises": exercises, "notes": notes}
