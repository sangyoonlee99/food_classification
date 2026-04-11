# services/goal_message_service.py
# STEP 5-5: 다음 목표 1줄(anchor message) 생성 서비스

from __future__ import annotations

from common.schemas import (
    UserGoal,
    Event,
    DailyMealSummary,
    WeeklyMealSummary,
)


def _normalize_goal_type(goal_type: str | None) -> str:
    gt = (goal_type or "").strip().lower()
    if gt == "weight_gain":
        return "muscle_gain"
    if gt in ("maintain", "maintenance"):
        return "maintenance"
    if gt == "weight_loss":
        return "weight_loss"
    return "maintenance"


class GoalMessageService:
    """
    이벤트 + 목표 + 현재 상태를 기반으로
    '모든 추천의 기준이 되는 1줄 메시지' 생성

    ⚠️ 핵심 원칙
    - 계산은 GoalCalculatorService
    - 현실성 판단과 설명은 여기서 담당
    """

    MIN_KCAL = 1200
    MAX_KCAL = 3500

    def build_next_goal(
        self,
        goal: UserGoal,
        daily: DailyMealSummary | None = None,
        weekly: WeeklyMealSummary | None = None,
        event: Event | None = None,
    ) -> str:
        # ==================================================
        # 0️⃣ 목표 현실성 체크
        # ==================================================
        kcal = getattr(goal, "kcal_target", None)
        if kcal is not None:
            try:
                kcal_v = float(kcal)
            except Exception:
                kcal_v = None

            if kcal_v is not None:
                if kcal_v < self.MIN_KCAL:
                    return (
                        "현재 목표 설정은 매우 강도 높은 감량 계획입니다. "
                        "건강과 지속 가능성을 위해 목표 기간을 늘리거나 "
                        "중간 목표를 설정하는 것을 권장합니다."
                    )
                if kcal_v > self.MAX_KCAL:
                    return (
                        "현재 목표 설정은 섭취 열량이 크게 증가합니다. "
                        "체지방 증가나 소화 부담을 고려해 "
                        "조금 더 완만한 목표 설정을 추천합니다."
                    )

        gt = _normalize_goal_type(getattr(goal, "goal_type", None))

        # ==================================================
        # 1️⃣ 목표 유형 기반 기본 문장
        # ==================================================
        if gt == "weight_loss":
            base = "체중 감량 목표를 유지합니다"
        elif gt == "muscle_gain":
            base = "체중 증가 목표를 유지합니다"
        else:
            base = "현재 체중을 안정적으로 유지합니다"

        # ==================================================
        # 2️⃣ 이벤트 반영
        # ==================================================
        if event:
            et = getattr(event, "event_type", None)
            if et in ("dinner_drinking", "회식"):
                return f"{base}. 회식 이후 회복에 집중하는 주간입니다."
            if et in ("overeating", "과식"):
                return f"{base}. 과식 이후 균형 회복을 우선합니다."
            if et in ("travel", "여행"):
                return f"{base}. 여행 중 유지 가능한 루틴을 적용합니다."
            if et in ("sleep_deprivation", "수면부족"):
                return f"{base}. 컨디션 회복을 우선한 계획을 적용합니다."

        # ==================================================
        # 3️⃣ 주간 상태 반영
        # ==================================================
        if weekly:
            score = getattr(weekly, "weekly_score", None)
            if score is not None:
                try:
                    score_v = float(score)
                except Exception:
                    score_v = None

                if score_v is not None:
                    if score_v < 55:
                        return f"{base}. 이번 주는 식사·운동 리듬 회복이 핵심입니다."
                    if score_v >= 70:
                        return f"{base}. 현재 흐름을 그대로 이어가세요."

        # ==================================================
        # 4️⃣ 기본 메시지
        # ==================================================
        return f"{base}. 오늘 계획을 차분히 실천해 보세요."
