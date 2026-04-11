# diet/diet_templates.py

DIET_TEMPLATES = {

    # =====================================================
    # 공통 식사 칼로리 비율 (현실 고정값)
    # =====================================================
    "_meal_kcal_ratio": {
        "breakfast": 0.30,
        "lunch": 0.40,
        "dinner": 0.30,
        "snack": 0.00,
    },

    # -----------------------------------------------------
    # 감량 (Weight Loss)
    # -----------------------------------------------------
    "weight_loss": {
        "calorie_factor": 0.80,
        "macros": {"carb": 0.40, "protein": 0.35, "fat": 0.25},

        # ✅ 규칙
        "rules": {
            "one_carb_per_meal": True,
        },

        "meals": {
            "breakfast": [
                {"category": "carb_low_gi", "portion": 1.0},   # 탄수 1개
                {"category": "protein", "portion": 1.2},
                {"category": "vegetable", "portion": 0.8},
            ],
            "lunch": [
                {"category": "carb", "portion": 1.1},          # 탄수 1개
                {"category": "protein", "portion": 1.3},
                {"category": "vegetable", "portion": 1.5},
            ],
            "dinner": [
                {"category": "carb_low_gi", "portion": 0.8},   # 탄수 1개
                {"category": "protein", "portion": 1.4},
                {"category": "vegetable", "portion": 1.8},
            ],
        },
    },

    # -----------------------------------------------------
    # 유지 (Maintenance)
    # -----------------------------------------------------
    "maintenance": {
        "calorie_factor": 1.00,
        "macros": {"carb": 0.45, "protein": 0.30, "fat": 0.25},

        "rules": {
            "one_carb_per_meal": True,
        },

        "meals": {
            "breakfast": [
                {"category": "carb", "portion": 1.0},
                {"category": "protein", "portion": 1.2},
                {"category": "vegetable", "portion": 0.8},
            ],
            "lunch": [
                {"category": "carb", "portion": 1.4},
                {"category": "protein", "portion": 1.4},
                {"category": "vegetable", "portion": 1.5},
            ],
            "dinner": [
                {"category": "carb_low_gi", "portion": 1.0},
                {"category": "protein", "portion": 1.3},
                {"category": "vegetable", "portion": 1.5},
            ],
        },
    },

    # -----------------------------------------------------
    # 벌크업 (Muscle Gain)
    # -----------------------------------------------------
    "muscle_gain": {
        "calorie_factor": 1.15,
        "macros": {"carb": 0.50, "protein": 0.30, "fat": 0.20},

        "rules": {
            "one_carb_per_meal": True,
        },

        "meals": {
            "breakfast": [
                {"category": "carb", "portion": 1.5},
                {"category": "protein", "portion": 1.6},
            ],
            "lunch": [
                {"category": "carb", "portion": 1.8},
                {"category": "protein", "portion": 1.6},
                {"category": "vegetable", "portion": 1.5},
            ],
            "dinner": [
                {"category": "carb", "portion": 1.5},
                {"category": "protein", "portion": 1.6},
                {"category": "vegetable", "portion": 1.8},
            ],
        },
    },

    # -----------------------------------------------------
    # 고령자 (Elderly)
    # -----------------------------------------------------
    "elderly": {
        "calorie_factor": 0.90,
        "macros": {"carb": 0.45, "protein": 0.30, "fat": 0.25},

        "rules": {
            "one_carb_per_meal": True,
        },

        "meals": {
            "breakfast": [
                {"category": "carb_low_gi", "portion": 0.8},
                {"category": "protein", "portion": 1.0},
                {"category": "vegetable", "portion": 0.8},
            ],
            "lunch": [
                {"category": "carb_low_gi", "portion": 1.2},
                {"category": "protein", "portion": 1.3},
                {"category": "vegetable", "portion": 1.5},
            ],
            "dinner": [
                {"category": "carb_low_gi", "portion": 0.8},
                {"category": "protein", "portion": 1.2},
                {"category": "vegetable", "portion": 1.6},
            ],
        },
    },

    # -----------------------------------------------------
    # 당뇨 (Diabetes)
    # -----------------------------------------------------
    "diabetes": {
        "calorie_factor": 0.95,
        "macros": {"carb": 0.35, "protein": 0.35, "fat": 0.30},

        "rules": {
            "one_carb_per_meal": True,
            "force_low_gi": True,
        },

        "meals": {
            "breakfast": [
                {"category": "carb_low_gi", "portion": 0.7},
                {"category": "protein", "portion": 1.3},
            ],
            "lunch": [
                {"category": "carb_low_gi", "portion": 1.0},
                {"category": "protein", "portion": 1.5},
                {"category": "vegetable", "portion": 1.5},
            ],
            "dinner": [
                {"category": "carb_low_gi", "portion": 0.8},
                {"category": "protein", "portion": 1.4},
                {"category": "vegetable", "portion": 1.8},
            ],
        },
    },

    # -----------------------------------------------------
    # 고혈압 (Hypertension)
    # -----------------------------------------------------
    "hypertension": {
        "calorie_factor": 0.95,
        "macros": {"carb": 0.45, "protein": 0.30, "fat": 0.25},
        "low_sodium": True,

        "rules": {
            "one_carb_per_meal": True,
            "force_low_gi": True,
        },

        "meals": {
            "breakfast": [
                {"category": "carb_low_gi", "portion": 0.8},
                {"category": "protein", "portion": 1.2},
            ],
            "lunch": [
                {"category": "carb_low_gi", "portion": 1.4},
                {"category": "protein", "portion": 1.5},
                {"category": "vegetable", "portion": 1.5},
            ],
            "dinner": [
                {"category": "carb_low_gi", "portion": 0.8},
                {"category": "protein", "portion": 1.4},
                {"category": "vegetable", "portion": 1.8},
            ],
        },
    },
}
