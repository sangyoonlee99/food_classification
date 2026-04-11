from __future__ import annotations

from typing import List, Dict, Optional, Any, Tuple
from datetime import date
import random
import pandas as pd

from common.schemas import MealItem, DailyDietPlan, UserProfile, UserGoal
from diet.diet_templates import DIET_TEMPLATES
from diet.food_categories import filter_food_candidates
from diet.bmr_calc import calculate_user_daily_targets
from diet.nutrient_rules import build_nutrition_settings
from infra.file_loaders.food_csv_loader import load_food_nutrition_db
from diet.diet_scaler import scale_meal_items

# ==================================================
# 설정
# ==================================================
GRAM_LIMITS = {
    "carb": (60, 220),
    "carb_low_gi": (60, 220),
    "protein": (80, 250),
    "fat": (5, 30),
    "vegetable": (50, 200),
}

ROUND_UNIT_G = 10
REQUIRED_GROUPS = ["carb_group", "protein", "fat", "vegetable"]

# 끼니 비중: goal_type에 따라 다르게
MEAL_KCAL_RATIO_BY_GOAL = {
    "weight_loss": {"breakfast": 0.30, "lunch": 0.40, "dinner": 0.30, "snack": 0.00},
    "maintenance": {"breakfast": 0.28, "lunch": 0.40, "dinner": 0.27, "snack": 0.05},
    "muscle_gain": {"breakfast": 0.27, "lunch": 0.38, "dinner": 0.25, "snack": 0.10},
}

# ==================================================
# Utils
# ==================================================
def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _round_gram(g: float) -> int:
    return int(round(g / ROUND_UNIT_G) * ROUND_UNIT_G)


def _normalize_goal_type(goal_type: str | None) -> str:
    gt = (goal_type or "").strip().lower()
    if gt in ("weight_gain", "gain", "bulk", "muscle_gain"):
        return "muscle_gain"
    if gt in ("maintain", "maintenance", "keep"):
        return "maintenance"
    if gt in ("weight_loss", "loss", "diet"):
        return "weight_loss"
    return "maintenance"


def _map_category(cat: str) -> str:
    s = (cat or "").strip()
    if s in ("carb", "carbs"):
        return "carb"
    if s in ("carb_low_gi", "carb_lowGI", "carb_lowgi"):
        return "carb_low_gi"
    if s in ("protein", "prot"):
        return "protein"
    if s in ("fat", "fats"):
        return "fat"
    if s in ("vegetable", "veg", "veggie"):
        return "vegetable"
    return s


def _category_group(cat: str) -> str:
    c = _map_category(cat)
    if c in ("carb", "carb_low_gi"):
        return "carb_group"
    return c


def _coerce_profile_goal(profile: Any, goal: Any) -> Tuple[UserProfile, UserGoal]:
    if not isinstance(profile, UserProfile):
        profile = UserProfile.model_validate(profile)

    if isinstance(goal, dict):
        gd = dict(goal)
        gd["goal_type"] = _normalize_goal_type(gd.get("goal_type"))
        goal = gd

    if not isinstance(goal, UserGoal):
        goal = UserGoal.model_validate(goal)

    # goal.goal_type 확정(정규화)
    goal.goal_type = _normalize_goal_type(getattr(goal, "goal_type", None))

    return profile, goal


def _sum_kcal(items: List[MealItem]) -> float:
    return float(sum(float(i.calorie or 0.0) for i in items))


def _dynamic_meal_scale_clamp(target_kcal: float) -> Tuple[float, float]:
    """
    목표 kcal이 커질수록 상한을 넉넉히 풀어줘야 함.
    (기존 상한이 낮으면 고칼로리 목표를 절대 못 맞춤)
    """
    if target_kcal >= 3000:
        return (0.50, 2.80)
    if target_kcal >= 2600:
        return (0.55, 2.50)
    if target_kcal >= 2200:
        return (0.60, 2.20)
    if target_kcal >= 1800:
        return (0.65, 1.90)
    return (0.65, 1.60)


def _dynamic_total_scale_clamp(target_kcal: float) -> Tuple[float, float]:
    """
    ✅ 여기서가 핵심
    total_adjust가 상한 1.35로 막혀 있으면
    (예: total_now 1000 -> target 2500) 절대 못 올라감.
    """
    if target_kcal >= 3000:
        return (0.45, 3.00)
    if target_kcal >= 2600:
        return (0.50, 2.70)
    if target_kcal >= 2200:
        return (0.55, 2.40)
    if target_kcal >= 1800:
        return (0.60, 2.10)
    return (0.70, 1.70)


# ==================================================
# Food chooser
# ==================================================
def choose_food_row(
    category: str,
    nutrition_df: pd.DataFrame,
    used_foods: Dict[str, int],
    rng: random.Random,
):
    major = {
        "carb": "탄수화물",
        "carb_low_gi": "탄수화물",
        "protein": "단백질",
        "fat": "지방",
        "vegetable": "채소",
    }.get(category)

    if not major:
        return None

    df = filter_food_candidates(nutrition_df, category=major, recommend_only=True)
    if df is None or df.empty:
        df = filter_food_candidates(nutrition_df, category=major, recommend_only=False)

    if df is None or df.empty:
        return None

    name_col = "food_name" if "food_name" in df.columns else "음식명"

    weights = []
    for _, r in df.iterrows():
        nm = str(r.get(name_col, ""))
        cnt = used_foods.get(nm, 0)
        weights.append(max(0.3, 1.0 - 0.3 * cnt))

    idx = rng.choices(df.index.tolist(), weights=weights, k=1)[0]
    row = df.loc[idx]
    nm = str(row.get(name_col, ""))
    used_foods[nm] = used_foods.get(nm, 0) + 1
    return row


def build_meal_item(row: pd.Series, portion_g: float, category: str) -> MealItem:
    category = _map_category(category)
    serving_g = float(row.get("중량(g)") or 100.0)

    lo, hi = GRAM_LIMITS.get(category, (30, 300))
    portion_g = _round_gram(_clamp(float(portion_g), lo, hi))

    factor = portion_g / serving_g if serving_g > 0 else 1.0

    return MealItem(
        food_name=row.get("food_name") or row.get("음식명") or "",
        category=category,
        portion_gram=float(portion_g),
        calorie=round(float(row.get("에너지(kcal)") or 0.0) * factor, 1),
        carb=round(float(row.get("탄수화물(g)") or 0.0) * factor, 1),
        protein=round(float(row.get("단백질(g)") or 0.0) * factor, 1),
        fat=round(float(row.get("지방(g)") or 0.0) * factor, 1),
        recommend_level=row.get("권장도"),
    )


# ==================================================
# Main
# ==================================================
def generate_daily_diet(
    profile: Any,
    goal: Any,
    date_value: date,
    elderly_mode: bool = False,
    used_foods: Optional[Dict[str, int]] = None,
) -> DailyDietPlan:
    profile, goal = _coerce_profile_goal(profile, goal)

    nutrition_df = load_food_nutrition_db()
    used_foods = used_foods or {}

    rng = random.Random(f"{profile.user_id.hex()}:{date_value.isoformat()}")

    # ✅ 단일 진실: 사용자가 설정한 goal.kcal_target(=DB값) 우선
    # (calc 결과가 goal을 덮어쓰면 UI 목표와 추천 합계가 계속 어긋남)
    if getattr(goal, "kcal_target", None):
        target_kcal = float(goal.kcal_target)
    else:
        targets = calculate_user_daily_targets(profile, goal, elderly_mode)
        target_kcal = float(targets.get("target_calorie") or 1800)

    # 템플릿 선택
    settings = build_nutrition_settings(profile, goal, elderly_mode)
    template_key = settings.get("template_key") or goal.goal_type or "maintenance"
    template_key = _normalize_goal_type(template_key)
    template_key = template_key if template_key in DIET_TEMPLATES else "maintenance"
    template = DIET_TEMPLATES[template_key]["meals"]

    force_low_gi = bool(getattr(profile, "has_diabetes", False))
    ratios = MEAL_KCAL_RATIO_BY_GOAL.get(goal.goal_type, MEAL_KCAL_RATIO_BY_GOAL["maintenance"])

    meal_scale_lo, meal_scale_hi = _dynamic_meal_scale_clamp(target_kcal)
    total_scale_lo, total_scale_hi = _dynamic_total_scale_clamp(target_kcal)

    def build(meal_key: str) -> List[MealItem]:
        meal_tpl = template.get(meal_key, []) or []
        items: List[MealItem] = []
        carb_used = False

        for comp in meal_tpl:
            cat = _map_category(comp.get("category"))
            portion = float(comp.get("portion") or 0)

            if cat in ("carb", "carb_low_gi"):
                if carb_used:
                    continue
                if force_low_gi:
                    cat = "carb_low_gi"
                carb_used = True

            row = choose_food_row(cat, nutrition_df, used_foods, rng)
            if row is None:
                continue
            items.append(build_meal_item(row, portion, cat))

        # 4대 요소 보장
        present_groups = {_category_group(i.category) for i in items}
        for need in REQUIRED_GROUPS:
            if need in present_groups:
                continue

            if need == "carb_group":
                cat = "carb_low_gi" if force_low_gi else "carb"
                row = choose_food_row(cat, nutrition_df, used_foods, rng)
                if row is not None:
                    items.append(build_meal_item(row, GRAM_LIMITS[cat][0], cat))
            else:
                cat = need
                row = choose_food_row(cat, nutrition_df, used_foods, rng)
                if row is not None:
                    items.append(build_meal_item(row, GRAM_LIMITS[cat][0], cat))

        return items

    breakfast = build("breakfast")
    lunch = build("lunch")
    dinner = build("dinner")

    # 간식(유지/증량만)
    snack: List[MealItem] = []
    if float(ratios.get("snack", 0.0)) > 0:
        snack = build("snack")
        if not snack:
            # 심화 X: 2개만 최소 생성
            cat1 = "carb_low_gi" if force_low_gi else "carb"
            row1 = choose_food_row(cat1, nutrition_df, used_foods, rng)
            if row1 is not None:
                snack.append(build_meal_item(row1, GRAM_LIMITS[cat1][0], cat1))
            row2 = choose_food_row("protein", nutrition_df, used_foods, rng)
            if row2 is not None:
                snack.append(build_meal_item(row2, GRAM_LIMITS["protein"][0], "protein"))

    # --------------------------------------------------
    # 1) 끼니별 목표 비중 맞추기(스케일)
    # --------------------------------------------------
    def fit_meal(items: List[MealItem], meal_key: str):
        want = float(target_kcal) * float(ratios.get(meal_key, 0.0))
        if want <= 0:
            items[:] = []
            return
        cur = _sum_kcal(items)
        if cur <= 0:
            return

        scale = want / cur
        scale = _clamp(scale, meal_scale_lo, meal_scale_hi)
        items[:] = scale_meal_items(items, scale)

    fit_meal(breakfast, "breakfast")
    fit_meal(lunch, "lunch")
    fit_meal(dinner, "dinner")
    fit_meal(snack, "snack")

    # --------------------------------------------------
    # 2) 전체 합계 보정(목표와 합계 mismatch 최소화)
    #    ✅ 반복 + 상한 확장으로 고목표에서도 수렴
    # --------------------------------------------------
    def total_adjust(max_iter: int = 4):
        for _ in range(max_iter):
            all_items = breakfast + lunch + dinner + snack
            total_now = _sum_kcal(all_items)
            if total_now <= 0:
                return

            diff = total_now - target_kcal
            if abs(diff) < 30:
                return

            total_scale = float(target_kcal) / float(total_now)
            total_scale = _clamp(total_scale, total_scale_lo, total_scale_hi)

            breakfast[:] = scale_meal_items(breakfast, total_scale)
            lunch[:] = scale_meal_items(lunch, total_scale)
            dinner[:] = scale_meal_items(dinner, total_scale)
            snack[:] = scale_meal_items(snack, total_scale)

    total_adjust(max_iter=5)

    # 최종 집계
    all_items = breakfast + lunch + dinner + snack

    return DailyDietPlan(
        user_id=profile.user_id,
        date=date_value,
        target_kcal=round(float(target_kcal)),
        total_kcal=round(sum(float(i.calorie or 0.0) for i in all_items)),
        total_macro={
            "carb_g": round(sum(float(i.carb or 0.0) for i in all_items), 1),
            "protein_g": round(sum(float(i.protein or 0.0) for i in all_items), 1),
            "fat_g": round(sum(float(i.fat or 0.0) for i in all_items), 1),
        },
        meals={
            "breakfast": breakfast,
            "lunch": lunch,
            "dinner": dinner,
            "snack": snack,
        },
        breakfast=breakfast,
        lunch=lunch,
        dinner=dinner,
        snacks=snack,
    )
