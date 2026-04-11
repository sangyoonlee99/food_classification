# routine/routine_template.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import date

from common.schemas import UserProfile, UserGoal


def _as_list(v) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str):
        parts = [p.strip() for p in v.split(",")]
        return [p for p in parts if p]
    s = str(v).strip()
    return [s] if s else []


def _map_repo_profile_to_schema_dict(user_profile: Dict[str, Any]) -> Dict[str, Any]:
    sex = user_profile.get("sex")
    gender = None
    if sex in ("male", "female"):
        gender = sex
    elif sex in ("M", "m", "남", "남성"):
        gender = "male"
    elif sex in ("F", "f", "여", "여성"):
        gender = "female"

    return {
        "user_id": user_profile.get("user_id"),  # bytes(16)
        "gender": gender,
        "birth_year": user_profile.get("birth_year"),
        "height_cm": user_profile.get("height_cm"),
        "weight_kg": user_profile.get("weight_kg") or user_profile.get("weight_kg_baseline"),
        "activity_level": user_profile.get("activity_level", "medium"),
        "has_diabetes": bool(user_profile.get("has_diabetes", False)),
        "has_hypertension": bool(user_profile.get("has_hypertension", False)),
        "banned_foods": user_profile.get("banned_foods", []) or [],
        "meal_priority": user_profile.get("meal_priority", "balanced") or "balanced",
        "preferred_exercise": user_profile.get("preferred_exercise", []) or [],
    }


def _map_goal_dict_to_schema_dict(user_goal: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "user_id": user_goal.get("user_id"),
        "goal_type": user_goal.get("goal_type", "maintenance"),
        "target_weight_kg": user_goal.get("target_weight_kg"),
        "target_weeks": user_goal.get("target_weeks"),
        "kcal_target": user_goal.get("kcal_target") or user_goal.get("target_kcal"),
        "macro_target": user_goal.get("macro_target"),
    }


# ==================================================
# Meal Template (daily_diet_planner 기반)
# ==================================================
def build_meal_template(
    user_profile: Dict[str, Any],
    user_goal: Dict[str, Any],
    *,
    user_id: Optional[bytes] = None,          # ✅ 호환(있으면 주입, 없어도 동작)
    target_date: Optional[date] = None,
    elderly_mode: Optional[bool] = None,
) -> Dict[str, Any]:
    if target_date is None:
        target_date = date.today()

    # 1) 입력 정규화
    prof_in = dict(user_profile or {})
    goal_in = dict(user_goal or {})

    # ✅ user_id를 함수 인자로 받았으면 profile에 주입 (둘 중 하나만 있어도 OK)
    if user_id is not None and not prof_in.get("user_id"):
        prof_in["user_id"] = user_id

    prof_in = _map_repo_profile_to_schema_dict(prof_in)
    goal_in = _map_goal_dict_to_schema_dict(goal_in)

    profile = UserProfile(**prof_in)
    goal = UserGoal(**goal_in)

    if elderly_mode is None:
        elderly_mode = bool(getattr(profile, "is_elderly", False))

    # 2) 고급 식단 생성 (지연 import)
    from diet.daily_diet_planner import generate_daily_diet

    plan = generate_daily_diet(
        profile=profile,
        goal=goal,
        date_value=target_date,
        elderly_mode=elderly_mode,
        used_foods=None,
    )

    # 3) UI 표준 변환
    meals_dict = getattr(plan, "meals", None) or {}

    def _meal_items_to_ui(items) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for it in (items or []):
            out.append(
                {
                    "name": getattr(it, "food_name", "") or "",
                    "grams": int(round(float(getattr(it, "portion_gram", 0) or 0))),
                    "group": getattr(it, "category", "") or "",
                    "kcal": float(getattr(it, "calorie", 0) or 0),
                    "carb_g": float(getattr(it, "carb", 0) or 0),
                    "protein_g": float(getattr(it, "protein", 0) or 0),
                    "fat_g": float(getattr(it, "fat", 0) or 0),
                }
            )
        return out

    def _sum_kcal(items) -> int:
        return int(round(sum(float(getattr(i, "calorie", 0) or 0) for i in (items or []))))

    breakfast_items = meals_dict.get("breakfast") or getattr(plan, "breakfast", []) or []
    lunch_items = meals_dict.get("lunch") or getattr(plan, "lunch", []) or []
    dinner_items = meals_dict.get("dinner") or getattr(plan, "dinner", []) or []
    snack_items = meals_dict.get("snack") or meals_dict.get("snacks") or getattr(plan, "snacks", []) or []

    total_kcal = int(round(float(getattr(plan, "target_kcal", 0) or getattr(plan, "total_kcal", 0) or 0)))

    return {
        "total_kcal": total_kcal,
        "meals": {
            "breakfast": {"target_kcal": _sum_kcal(breakfast_items), "items": _meal_items_to_ui(breakfast_items)},
            "lunch": {"target_kcal": _sum_kcal(lunch_items), "items": _meal_items_to_ui(lunch_items)},
            "dinner": {"target_kcal": _sum_kcal(dinner_items), "items": _meal_items_to_ui(dinner_items)},
            "snack": {"target_kcal": _sum_kcal(snack_items), "items": _meal_items_to_ui(snack_items)},
        },
    }


# ==================================================
# Exercise Template (A-6 최소 버전 유지)
# ==================================================
def build_exercise_template(
    user_profile: Dict[str, Any],
    user_goal: Dict[str, Any],
    *,
    state: Optional[Dict[str, Any]] = None,
    exercise_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    exercise_context = exercise_context or {}

    preferred_cardio = _as_list(exercise_context.get("preferred_cardio")) or _as_list(user_profile.get("preferred_cardio"))
    preferred_strength = _as_list(exercise_context.get("preferred_strength")) or _as_list(user_profile.get("preferred_strength"))

    if not preferred_cardio:
        preferred_cardio = ["걷기"]
    if not preferred_strength:
        preferred_strength = ["홈트레이닝"]

    cardio_candidates = list(preferred_cardio)
    strength_candidates = list(preferred_strength)

    plateau_days = int((state or {}).get("plateau_days", 0) or 0)
    if plateau_days >= 14:
        if "걷기" in cardio_candidates and "자전거" not in cardio_candidates:
            cardio_candidates.append("자전거")
        if "홈트레이닝" in strength_candidates and "헬스" not in strength_candidates:
            strength_candidates.append("헬스")

    return {
        "cardio_candidates": cardio_candidates,
        "strength_candidates": strength_candidates,
        "cardio_min": int(user_goal.get("cardio_min", 30) or 30),
        "strength_min": int(user_goal.get("strength_min", 20) or 20),
        "intensity": str(user_goal.get("intensity", "keep") or "keep"),
    }
