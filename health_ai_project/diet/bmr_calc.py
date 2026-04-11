# diet/bmr_calc.py

from typing import Dict, Literal
from common.schemas import UserProfile, UserGoal

# -----------------------------------------------------
# 활동량 계수
# -----------------------------------------------------
ACTIVITY_FACTOR_MAP: Dict[str, float] = {
    "sedentary": 1.2,
    "light": 1.375,
    "moderate": 1.55,
    "active": 1.725,
    "very_active": 1.9,

    # fallback
    "low": 1.2,
    "medium": 1.55,
    "high": 1.9,
}


# -----------------------------------------------------
# BMR (Mifflin–St Jeor)
# -----------------------------------------------------
def calculate_bmr(
    gender: Literal["male", "female"],
    age: int,
    height_cm: float,
    weight_kg: float,
) -> float:
    if gender == "male":
        return float(10 * weight_kg + 6.25 * height_cm - 5 * age + 5)
    return float(10 * weight_kg + 6.25 * height_cm - 5 * age - 161)


def get_activity_factor(activity_level: str) -> float:
    return ACTIVITY_FACTOR_MAP.get(activity_level, 1.2)


def calculate_tdee(bmr: float, activity_level: str) -> float:
    return float(bmr * get_activity_factor(activity_level))


GOAL_CALORIE_FACTOR: Dict[str, float] = {
    "weight_loss": 0.80,
    "maintenance": 1.00,
    "muscle_gain": 1.15,
}


def calculate_target_calorie(goal_type: str, tdee: float) -> float:
    return float(tdee * GOAL_CALORIE_FACTOR.get(goal_type, 1.0))


def is_elderly(age: int, user_elderly_flag: bool = False) -> bool:
    return age >= 65 or user_elderly_flag


# -----------------------------------------------------
# ✅ 핵심 함수 (dict 반환 유지)
# -----------------------------------------------------
def calculate_user_daily_targets(
    profile: UserProfile,
    goal: UserGoal,
    elderly_mode: bool = False,
) -> Dict[str, float | bool]:

    bmr = calculate_bmr(
        profile.gender,
        profile.age,
        profile.height_cm,
        profile.weight_kg,
    )

    tdee = calculate_tdee(bmr, profile.activity_level)
    target_calorie = calculate_target_calorie(goal.goal_type, tdee)
    elderly_flag = is_elderly(profile.age, elderly_mode or profile.is_elderly)

    return {
        "bmr": bmr,
        "tdee": tdee,
        "target_calorie": target_calorie,
        "is_elderly": elderly_flag,
    }
