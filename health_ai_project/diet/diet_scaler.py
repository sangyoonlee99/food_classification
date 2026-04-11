# diet/diet_scaler.py
from __future__ import annotations

from typing import List
from common.schemas import MealItem


def scale_meal_items(
    items: List[MealItem],
    scale: float,
    min_gram: float = 20.0,
) -> List[MealItem]:
    """
    MealItem 리스트를 동일 비율로 스케일링하되,
    '칼로리/탄단지'는 단순 곱셈 누적이 아니라
    기존 item의 (단위그램당 값) 기반으로 재계산한다.

    ✅ 효과
    - 여러 번 스케일링해도 kcal/탄단지 비율이 깨지지 않음
    - planner에서 끼니별 스케일 + 전체 스케일을 해도 안정적
    """
    scaled_items: List[MealItem] = []

    for item in items:
        scaled = item.copy(deep=True)

        base_g = float(item.portion_gram or 0.0)
        if base_g <= 0:
            # 안전 fallback: 그냥 곱셈
            new_g = max(float(item.portion_gram or 0.0) * float(scale), float(min_gram))
            scaled.portion_gram = new_g
            scaled.calorie = float(item.calorie or 0.0) * float(scale)
            scaled.carb = float(item.carb or 0.0) * float(scale)
            scaled.protein = float(item.protein or 0.0) * float(scale)
            scaled.fat = float(item.fat or 0.0) * float(scale)
            scaled_items.append(scaled)
            continue

        # g당 값(현재 portion 기준)
        kcal_per_g = float(item.calorie or 0.0) / base_g
        carb_per_g = float(item.carb or 0.0) / base_g
        protein_per_g = float(item.protein or 0.0) / base_g
        fat_per_g = float(item.fat or 0.0) / base_g

        new_g = max(base_g * float(scale), float(min_gram))

        scaled.portion_gram = round(new_g, 1)
        scaled.calorie = round(kcal_per_g * new_g, 1)
        scaled.carb = round(carb_per_g * new_g, 1)
        scaled.protein = round(protein_per_g * new_g, 1)
        scaled.fat = round(fat_per_g * new_g, 1)

        scaled_items.append(scaled)

    return scaled_items
