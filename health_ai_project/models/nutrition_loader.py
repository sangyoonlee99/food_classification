import pandas as pd

class NutritionDB:
    def __init__(self, xlsx_path: str):
        self.df = pd.read_excel(xlsx_path)

        # 컬럼 정리
        self.df["음식명"] = self.df["음식명"].astype(str).str.strip()

    def summarize(self, food_names: list[str]):
        """
        음식명 리스트 → 영양 합산 결과
        없는 음식은 자동 skip
        """
        details = []

        total = {
            "calories": 0.0,
            "carb": 0.0,
            "protein": 0.0,
            "fat": 0.0,
        }

        for food in food_names:
            row = self.df[self.df["음식명"] == food]
            if row.empty:
                continue

            row = row.iloc[0]

            calories = float(row["에너지(kcal)"])
            carb = float(row["탄수화물(g)"])
            protein = float(row["단백질(g)"])
            fat = float(row["지방(g)"])

            details.append({
                "food_name": food,
                "calories": calories,
                "carb": carb,
                "protein": protein,
                "fat": fat,
            })

            total["calories"] += calories
            total["carb"] += carb
            total["protein"] += protein
            total["fat"] += fat

        return total, details
