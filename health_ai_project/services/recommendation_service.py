# services/recommendation_service.py

from typing import List, Dict
from common.schemas import NutritionSummary


class RecommendationService:
    """
    🔥 Replan 결과 기반 단일 추천 엔진
    """

    def build(
        self,
        *,
        diet_changed: bool,
        exercise_changed: bool,
        reason: str,
        nutrition: NutritionSummary,
    ) -> Dict[str, object]:

        diet_msgs: List[str] = []
        exercise_msgs: List[str] = []

        total = nutrition.total

        # -----------------------------
        # 식단 변경 설명
        # -----------------------------
        if diet_changed:
            if total.get("protein_g", 0) < 50:
                diet_msgs.append("단백질 보강 식단으로 조정되었습니다.")
            if total.get("kcal", 0) < 1400:
                diet_msgs.append("전체 섭취 열량을 낮춰 재구성했습니다.")

        # -----------------------------
        # 운동 변경 설명
        # -----------------------------
        if exercise_changed:
            exercise_msgs.append("회복 중심 운동으로 조정되었습니다.")

        # -----------------------------
        # 다음 행동 1줄
        # -----------------------------
        if diet_changed:
            next_action = "다음 식사에서 단백질 식품을 먼저 선택하세요."
        elif exercise_changed:
            next_action = "오늘은 가벼운 걷기 위주로 진행하세요."
        else:
            next_action = "현재 계획을 그대로 유지하세요."

        summary = self._build_summary(diet_changed, exercise_changed)

        return {
            "summary": summary,
            "reason": reason,
            "next_action": next_action,
            "diet_changes": diet_msgs,
            "exercise_changes": exercise_msgs,
        }

    # -----------------------------
    def _build_summary(self, diet_changed: bool, exercise_changed: bool) -> str:
        if diet_changed and exercise_changed:
            return "식단과 운동 계획이 상황에 맞게 조정되었습니다."
        if diet_changed:
            return "식단 계획이 조정되었습니다."
        if exercise_changed:
            return "운동 계획이 조정되었습니다."
        return "현재 계획을 잘 유지하고 있습니다."
