from __future__ import annotations

from typing import Any, Dict


def apply_meal_policy(
    *,
    meal_plan: Dict[str, Any],
    meal_actions: Dict[str, Any],
    user_profile: Dict[str, Any],
) -> Dict[str, Any]:
    if not meal_actions:
        return meal_plan

    adjust = meal_actions.get("adjust_grams")
    if isinstance(adjust, dict):
        kcal_delta = int(adjust.get("kcal_delta", 0))
        targets = adjust.get("apply_to_meals", [])

        if kcal_delta != 0 and targets:
            per_meal_delta = kcal_delta / max(len(targets), 1)

            for meal_type in targets:
                meal = meal_plan.get("meals", {}).get(meal_type)
                if not meal:
                    continue

                items = meal.get("items") or []
                if not items:
                    continue

                current_kcal = sum(i.get("calorie", 0) for i in items)
                if current_kcal <= 0:
                    continue

                scale = max(0.6, (current_kcal + per_meal_delta) / current_kcal)

                for i in items:
                    i["grams"] = int(i.get("grams", 0) * scale)
                    i["calorie"] = int(i.get("calorie", 0) * scale)

    return meal_plan


def apply_exercise_policy(
    *,
    exercise_plan: Dict[str, Any],
    exercise_actions: Dict[str, Any],
    user_profile: Dict[str, Any],
) -> Dict[str, Any]:
    if not exercise_actions:
        return exercise_plan

    adj = exercise_actions.get("adjust_minutes") or {}
    if "cardio" in adj:
        exercise_plan["cardio_min"] = int(exercise_plan.get("cardio_min", 0)) + int(adj["cardio"])
    if "strength" in adj:
        exercise_plan["strength_min"] = int(exercise_plan.get("strength_min", 0)) + int(adj["strength"])

    if exercise_actions.get("intensity"):
        exercise_plan["intensity"] = exercise_actions["intensity"]

    return exercise_plan
