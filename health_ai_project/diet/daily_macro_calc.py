# diet/daily_macro_calc.py
# 순수 계산 전용 (테스트/검증/확장용)

def calculate_macro_grams(target_calorie: float, macros: dict) -> dict:
    KCAL = {"carb": 4, "protein": 4, "fat": 9}
    return {
        f"{k}_g": round(target_calorie * v / KCAL[k], 1)
        for k, v in macros.items()
    }
