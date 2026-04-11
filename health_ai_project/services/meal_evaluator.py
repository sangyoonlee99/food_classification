# services/meal_evaluator.py
# 한 끼 영양 요약 → 점수/등급/해석 생성 (Rule-based)
# STEP 2 + STEP 3-1 (Context 확장)

from typing import Dict, Any, List


class MealEvaluator:
    """
    영양 합산 결과를 해석하여
    - 점수 (0~100)
    - 등급
    - 상태 플래그
    - 기본 조언 (STEP 2)
    - 사용자 맥락 조언 (STEP 3-1)
    생성
    """

    # ==================================================
    # STEP 2: 기본 영양 평가 (기존 유지)
    # ==================================================
    def evaluate(self, nutrition_summary: Dict[str, Any]) -> Dict[str, Any]:
        total = nutrition_summary.get("total", {})
        kcal = total.get("kcal", 0)
        carbs = total.get("carbs_g", 0)
        protein = total.get("protein_g", 0)
        fat = total.get("fat_g", 0)

        flags = {
            "low_protein": protein < 20,
            "high_carbs": carbs > 80,
            "low_fat": fat < 10,
            "low_calorie": kcal < 400,
            "high_calorie": kcal > 900,
        }

        score = self._calculate_score(flags)
        grade = self._grade(score)
        advice = self._generate_advice(flags)

        return {
            "meal_score": score,
            "grade": grade,
            "flags": flags,
            "advice": advice,
        }

    # ==================================================
    # STEP 3-1: 사용자 맥락 기반 평가 (추가)
    # ==================================================
    def evaluate_with_context(
        self,
        nutrition_summary: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, List[str]]:

        total = nutrition_summary.get("total", {})
        kcal = total.get("kcal", 0)
        carbs = total.get("carbs_g", 0)
        protein = total.get("protein_g", 0)

        contextual_advice: List[str] = []

        # 👴 고령자
        if context.get("elderly"):
            if protein < 25:
                contextual_advice.append(
                    "고령자 기준으로 단백질이 부족합니다. "
                    "생선, 달걀찜, 두부처럼 부드러운 단백질을 보완하세요."
                )
            else:
                contextual_advice.append(
                    "고령자에게 적절한 단백질 섭취가 이루어졌습니다."
                )

        # 🩸 당뇨
        if context.get("diabetes"):
            if carbs > 60:
                contextual_advice.append(
                    "탄수화물 섭취가 다소 많아 혈당 상승 가능성이 있습니다."
                )
            else:
                contextual_advice.append(
                    "탄수화물 조절이 잘 되어 혈당 부담이 적은 식사입니다."
                )

        # 🥗 다이어트
        diet = context.get("diet")
        if diet == "weight_loss":
            if kcal > 600:
                contextual_advice.append(
                    "체중 감량 목표 대비 열량이 높은 편입니다."
                )
            else:
                contextual_advice.append(
                    "체중 감량에 적절한 열량의 식사입니다."
                )

        # 💪 운동
        fitness = context.get("fitness")
        if fitness == "post_workout":
            if protein < 25:
                contextual_advice.append(
                    "운동 후 회복을 위해 단백질 보충이 필요합니다."
                )
            else:
                contextual_advice.append(
                    "운동 후 회복에 적절한 단백질 섭취가 이루어졌습니다."
                )

        return {
            "context_advice": contextual_advice
        }

    # ------------------------------------
    def _calculate_score(self, flags: Dict[str, bool]) -> int:
        score = 100

        penalties = {
            "low_protein": 20,
            "high_carbs": 15,
            "low_fat": 10,
            "low_calorie": 15,
            "high_calorie": 15,
        }

        for k, p in penalties.items():
            if flags.get(k):
                score -= p

        return max(score, 0)

    def _grade(self, score: int) -> str:
        if score >= 85:
            return "아주 좋음"
        if score >= 70:
            return "보통"
        if score >= 50:
            return "주의"
        return "개선 필요"

    def _generate_advice(self, flags: Dict[str, bool]) -> List[str]:
        advice = []

        if flags["low_protein"]:
            advice.append("단백질이 부족합니다. 달걀, 두부, 생선 등을 추가해 보세요.")
        if flags["high_carbs"]:
            advice.append("탄수화물 비중이 높아 혈당 변동 가능성이 있습니다.")
        if flags["low_fat"]:
            advice.append("지방이 너무 낮습니다. 견과류나 올리브유를 소량 추가해 보세요.")
        if flags["low_calorie"]:
            advice.append("전체 열량이 낮습니다. 식사량 보완이 필요합니다.")
        if flags["high_calorie"]:
            advice.append("열량이 높은 식사입니다. 다음 끼니는 가볍게 조절하세요.")

        if not advice:
            advice.append("영양 균형이 비교적 잘 맞는 식사입니다 👍")

        return advice
