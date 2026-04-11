from __future__ import annotations
from typing import Dict, Any, List


def build_meal_explanation(
    *,
    meal_actions: Dict[str, Any],
) -> Dict[str, Any]:
    """
    UI 표시 전용 설명 빌더
    """
    if not meal_actions:
        return {}

    msgs: List[str] = []
    badges: List[str] = []

    reason = meal_actions.get("reason")
    adjust = meal_actions.get("adjust_grams")
    events = meal_actions.get("event_flags")

    if isinstance(adjust, dict):
        targets = adjust.get("apply_to_meals", [])
        kcal_delta = adjust.get("kcal_delta", 0)

        if targets and kcal_delta < 0:
            meal_label = ", ".join(targets)
            msgs.append(
                f"이미 섭취한 식사량을 고려해 **{meal_label}** 위주로 양을 조절했어요."
            )
            badges.append("GRAM_ADJUST")

    if reason == "over_kcal_after_meals":
        msgs.append("오늘 섭취 열량이 목표를 초과해 자동 조정이 적용됐어요.")
        badges.append("AUTO_ADJUST")

    if events:
        msgs.append("오늘 기록된 이벤트가 식단 판단에 반영됐어요.")
        badges.append("EVENT")

    return {
        "messages": msgs,
        "badges": badges,
        "tone": "soft",
    }
