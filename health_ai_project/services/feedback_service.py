# services/feedback_service.py

import random
from typing import List
from common.schemas import NutritionSummary


class FeedbackService:
    """
    ✅ 단일 AI 판단 엔진 (최종)
    - 홈 화면
    - 리포트 화면
    - 주간 / 월간
    👉 모든 AI 요약 문구는 여기서만 생성
    """

    # ==================================================
    # 종합 상태 문구 (AI 톤의 핵심)
    # ==================================================
    STATE_MESSAGES = {
        "excellent": [
            "전반적인 생활 패턴이 매우 안정적입니다.",
            "식사와 운동의 균형이 잘 유지되고 있습니다.",
        ],
        "good": [
            "대체로 안정적인 흐름을 유지하고 있습니다.",
            "전반적인 관리 상태는 양호한 편입니다.",
        ],
        "warning": [
            "일부 생활 패턴에서 조정이 필요해 보입니다.",
            "식사 또는 운동 중 한쪽의 보완이 필요합니다.",
        ],
        "danger": [
            "생활 패턴 전반에 개선이 필요합니다.",
            "현재 상태가 반복되면 목표 달성이 어려울 수 있습니다.",
        ],
    }

    # ==================================================
    # 세부 피드백 문구
    # ==================================================
    DETAIL_MESSAGES = {
        "low_protein": "단백질 섭취가 부족합니다. 식사에 단백질원을 한 가지 추가해 보세요.",
        "low_fat": "지방 섭취가 매우 낮습니다. 견과류나 올리브유를 소량 추가해 보세요.",
        "low_calorie": "전체 섭취 열량이 낮은 편입니다. 식사량을 조금 늘려도 좋겠습니다.",
        "high_carbs": "탄수화물 비중이 높은 편입니다. 정제 탄수 섭취를 줄여보세요.",
        "low_exercise": "운동량이 다소 부족합니다. 주 2~3회, 20분 이상을 목표로 해보세요.",
        "good_exercise": "운동 습관은 비교적 잘 유지되고 있습니다.",
        "weight_down": "체중이 감소 추세에 있습니다. 현재 흐름을 잘 유지해 보세요.",
        "weight_up": "체중이 증가 추세에 있습니다. 섭취량과 활동량을 함께 점검해 보세요.",
        "weight_stall": "체중 변화가 정체 상태입니다. 작은 조정이 필요할 수 있습니다.",
        "no_record": "기록이 충분하지 않아 정확한 판단이 어렵습니다. 기록을 조금 더 남겨주세요.",
    }

    # ==================================================
    # 내부 유틸
    # ==================================================
    def _pick_state(self, score: int) -> str:
        if score >= 85:
            return "excellent"
        if score >= 70:
            return "good"
        if score >= 55:
            return "warning"
        return "danger"

    # ==================================================
    # ✅ 홈 / 리포트 공용 AI 판단 (최종 API)
    # ==================================================
    def weekly_feedback(
        self,
        score: int,
        nutrition: NutritionSummary,
        exercise_total_min: float | None = None,
        weight_diff: float | None = None,
        has_record: bool = True,
        max_messages: int = 3,
    ) -> List[str]:
        """
        ✔ 홈 / 리포트 동일 사용
        ✔ 식사 + 운동 + 체중 종합 판단
        """

        messages: List[str] = []

        # ------------------
        # 1️⃣ 종합 상태
        # ------------------
        state_key = self._pick_state(score)
        messages.append(random.choice(self.STATE_MESSAGES[state_key]))

        # ------------------
        # 2️⃣ 기록 부족 보호
        # ------------------
        if not has_record:
            messages.append(self.DETAIL_MESSAGES["no_record"])
            return messages[:max_messages]

        total = nutrition.total or {}

        # ------------------
        # 3️⃣ 식사 평가
        # ------------------
        if total.get("protein_g", 0) < 40:
            messages.append(self.DETAIL_MESSAGES["low_protein"])

        if total.get("fat_g", 0) < 10:
            messages.append(self.DETAIL_MESSAGES["low_fat"])

        if total.get("kcal", 0) < 1500:
            messages.append(self.DETAIL_MESSAGES["low_calorie"])

        if total.get("carbs_g", 0) > 300:
            messages.append(self.DETAIL_MESSAGES["high_carbs"])

        # ------------------
        # 4️⃣ 운동 평가
        # ------------------
        if exercise_total_min is not None:
            if exercise_total_min < 150:
                messages.append(self.DETAIL_MESSAGES["low_exercise"])
            else:
                messages.append(self.DETAIL_MESSAGES["good_exercise"])

        # ------------------
        # 5️⃣ 체중 평가
        # ------------------
        if weight_diff is not None:
            if abs(weight_diff) < 0.2:
                messages.append(self.DETAIL_MESSAGES["weight_stall"])
            elif weight_diff < 0:
                messages.append(self.DETAIL_MESSAGES["weight_down"])
            else:
                messages.append(self.DETAIL_MESSAGES["weight_up"])

        # ------------------
        # 6️⃣ 중복 제거 + 개수 제한
        # ------------------
        uniq: List[str] = []
        for m in messages:
            if m not in uniq:
                uniq.append(m)

        return uniq[:max_messages]
