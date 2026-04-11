# services/action_executor.py
from __future__ import annotations
from typing import List, Dict, Any


def execute_actions(
    *,
    actions: List[Dict[str, Any]],
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Step 6 Final
    - recommendation_parser 결과 실행
    - 실제 수치 조정 ❌
    - 상태(state) 전이만 담당 (Single Source of Truth는 RoutineEngine)
    """

    applied: List[Dict[str, Any]] = []
    summary: List[str] = []

    next_state = dict(state)

    for action in actions:
        action_type = action.get("type")
        mode = action.get("mode")  # micro | macro

        # -------------------------
        # DIET
        # -------------------------
        if action_type == "diet":
            next_state["last_diet_action"] = mode
            next_state["recommendation_signature"] = (
                "diet:adjust" if mode == "micro" else "diet:menu"
            )

            summary.append(
                "식단을 소폭 조정했어요." if mode == "micro"
                else "식단 구성을 변경했어요."
            )

            applied.append(action)

        # -------------------------
        # EXERCISE
        # -------------------------
        elif action_type == "exercise":
            next_state["last_exercise_action"] = mode
            next_state["recommendation_signature"] = (
                "exercise:adjust" if mode == "micro" else "exercise:routine"
            )

            summary.append(
                "운동량을 조금 조정했어요." if mode == "micro"
                else "운동 루틴을 변경했어요."
            )

            applied.append(action)

    return {
        "state": next_state,
        "summary": summary,
        "applied_actions": applied,
    }
