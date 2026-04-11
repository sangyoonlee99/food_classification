# diet/diet_validator.py

from typing import Dict, List
from common.schemas import DailyDietPlan, UserProfile, UserGoal


def validate_daily_diet(
    plan: DailyDietPlan,
    profile: UserProfile,
    goal: UserGoal,
    *,
    kcal_tolerance: float = 0.05,
) -> Dict:
    """
    일일 식단 최종 검증
    - 허용 / 초과 / 부족 판단의 '근거 데이터' 생성
    """

    reasons: List[str] = []

    # ---------------------------
    # 1️⃣ kcal 검증
    # ---------------------------
    target_kcal = plan.target_kcal
    total_kcal = plan.total_kcal

    lower = target_kcal * (1 - kcal_tolerance)
    upper = target_kcal * (1 + kcal_tolerance)

    kcal_status = "allowed"
    if total_kcal < lower:
        kcal_status = "under"
        reasons.append(
            f"kcal too low: {total_kcal:.0f} < {lower:.0f}"
        )
    elif total_kcal > upper:
        kcal_status = "over"
        reasons.append(
            f"kcal too high: {total_kcal:.0f} > {upper:.0f}"
        )

    # ---------------------------
    # 2️⃣ 단백질 최소량
    # ---------------------------
    protein_g = plan.total_macro.get("protein_g", 0.0)

    if profile.is_elderly or profile.has_disease:
        min_protein = profile.weight_kg * 1.2
    else:
        min_protein = profile.weight_kg * 0.8

    if protein_g < min_protein:
        reasons.append(
            f"protein too low: {protein_g:.1f}g < {min_protein:.1f}g"
        )

    # ---------------------------
    # 3️⃣ 권장도 기반 제한 음식
    # ---------------------------
    for meal_name, items in plan.meals.items():
        for item in items:
            if getattr(item, "recommend_level", None) == "제한":
                reasons.append(
                    f"restricted food detected: {item.food_name}"
                )

    # ---------------------------
    # severity 판단 (엔진용)
    # ---------------------------
    if not reasons:
        severity = "low"
    elif kcal_status == "over":
        severity = "high" if total_kcal > target_kcal * 1.15 else "medium"
    else:
        severity = "medium"

    # ---------------------------
    # 결과 (🔥 고정 포맷)
    # ---------------------------
    return {
        "status": (
            "allowed" if not reasons
            else "over" if kcal_status == "over"
            else "under"
        ),
        "severity": severity,
        "reasons": reasons,
        "metrics": {
            "target_kcal": target_kcal,
            "total_kcal": total_kcal,
            "delta_kcal": total_kcal - target_kcal,
            "protein_g": protein_g,
            "min_protein_required": min_protein,
        },
        "is_valid": len(reasons) == 0,
    }
