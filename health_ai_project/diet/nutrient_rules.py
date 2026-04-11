# 템플릿 선택 & 매크로(탄단지) 규칙
# diet/nutrient_rules.py

from typing import Dict
from copy import deepcopy

from common.schemas import UserProfile, UserGoal
from diet.diet_templates import DIET_TEMPLATES
from diet.bmr_calc import is_elderly


# -----------------------------------------------------
# 템플릿 선택 로직
#   우선순위:
#   1) 당뇨 → "diabetes"
#   2) 고혈압 → "hypertension"
#   3) 고령자 → "elderly"
#   4) 그 외 → 목표(goal_type)에 따라 weight_loss / maintenance / muscle_gain
# -----------------------------------------------------
def choose_diet_template_key(
    profile: UserProfile,
    goal: UserGoal,
    elderly_mode: bool = False,
) -> str:
    # 고령자 여부 최종 판정
    elderly_flag = is_elderly(profile.age, elderly_mode or profile.is_elderly)

    if profile.has_diabetes:
        return "diabetes"
    if profile.has_hypertension:
        return "hypertension"
    if elderly_flag:
        return "elderly"

    # 일반 목표 타입 기반
    if goal.goal_type == "weight_loss":
        return "weight_loss"
    if goal.goal_type == "muscle_gain":
        return "muscle_gain"
    return "maintenance"


# -----------------------------------------------------
# 템플릿에서 기본 매크로 비율 가져오기
# -----------------------------------------------------
def get_base_macros(template_key: str) -> Dict[str, float]:
    tmpl = DIET_TEMPLATES[template_key]
    return deepcopy(tmpl.get("macros", {}))


# -----------------------------------------------------
# 매크로 비율 정규화 (합이 1.0 이 되도록)
# -----------------------------------------------------
def normalize_macros(macros: Dict[str, float]) -> Dict[str, float]:
    total = sum(macros.values())
    if total <= 0:
        return macros
    return {k: v / total for k, v in macros.items()}


# -----------------------------------------------------
# 고령자: 단백질 +5%, 지방 -5% (합 1로 다시 정규화)
# -----------------------------------------------------
def apply_elderly_macro_adjustment(macros: Dict[str, float]) -> Dict[str, float]:
    macros = deepcopy(macros)

    if "protein" in macros and "fat" in macros:
        macros["protein"] += 0.05
        macros["fat"] -= 0.05

    # 음수가 되지 않도록 최소 0 처리
    for k in macros:
        if macros[k] < 0:
            macros[k] = 0.0

    return normalize_macros(macros)


# -----------------------------------------------------
# 저염식 여부 (고혈압 템플릿)
# -----------------------------------------------------
def is_low_sodium_required(template_key: str) -> bool:
    tmpl = DIET_TEMPLATES[template_key]
    return bool(tmpl.get("low_sodium", False))


# -----------------------------------------------------
# 최종 영양 설정 생성:
#   - 사용할 템플릿 key
#   - 매크로 비율 (탄/단/지)
#   - 저염식 여부
#   - 내부 정책 설명 텍스트 (옵션)
# -----------------------------------------------------
def build_nutrition_settings(
    profile: UserProfile,
    goal: UserGoal,
    elderly_mode: bool = False,
) -> dict:
    template_key = choose_diet_template_key(profile, goal, elderly_mode)
    base_macros = get_base_macros(template_key)

    elderly_flag = is_elderly(profile.age, elderly_mode or profile.is_elderly)
    macros = normalize_macros(base_macros)

    notes = [f"template={template_key}"]

    # 고령자면 매크로 추가 조정
    if elderly_flag:
        macros = apply_elderly_macro_adjustment(macros)
        notes.append("elderly_macro_adjusted(+protein, -fat)")

    low_sodium = is_low_sodium_required(template_key)
    if low_sodium:
        notes.append("low_sodium_required")

    if profile.has_diabetes:
        notes.append("diabetes_rules_applied")
    if profile.has_hypertension:
        notes.append("hypertension_rules_applied")

    return {
        "template_key": template_key,
        "macros": macros,        # {"carb": x, "protein": y, "fat": z}
        "low_sodium": low_sodium,
        "is_elderly": elderly_flag,
        "notes": notes,
    }
