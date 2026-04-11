# events/replan_horizon_service.py
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

from events.horizon_engine import HorizonEngine, HorizonDecision


class ReplanHorizonService:
    """
    Step 8-A-13
    - Horizon은 기간/전략/일자별 미리보기 전용
    - actions 생성 ❌
    """

    def __init__(
        self,
        engine: Optional[HorizonEngine] = None,
    ):
        self.engine = engine or HorizonEngine()

    def build_horizon_adjustments(
        self,
        *,
        event_date: date,
        event_type: str,
        intensity: Optional[str],
        base_week_dates: List[date],
        user_settings: Dict[str, Any],
        actual_delta_kcal: Optional[float] = None,
        estimated_delta_kcal: float = 400.0,
    ) -> Dict[str, Any]:

        # -------------------------------------------------
        # 1️⃣ Horizon 판단 (기간·전략·일자별 조정)
        # -------------------------------------------------
        decision: HorizonDecision = self.engine.calculate(
            event_date=event_date,
            base_week_dates=base_week_dates,
            event_type=event_type,
            intensity=intensity,
            user_settings=user_settings,
            actual_delta_kcal=actual_delta_kcal,
            estimated_delta_kcal=estimated_delta_kcal,
        )

        # -------------------------------------------------
        # 2️⃣ 일자별 식단 / 운동 조정 정리 (미리보기용)
        # -------------------------------------------------
        days: List[Dict[str, Any]] = []
        all_days = sorted(
            set(decision.diet_day_adjustments.keys())
            | set(decision.exercise_day_adjustments.keys())
        )

        for d in all_days:
            days.append(
                {
                    "date": d.isoformat(),
                    "diet": decision.diet_day_adjustments.get(d, {}),
                    "exercise": decision.exercise_day_adjustments.get(d, {}),
                }
            )

        # -------------------------------------------------
        # 3️⃣ Horizon 결과 반환 (actions ❌)
        # -------------------------------------------------
        return {
            "strategy": decision.strategy,
            "horizon_days": decision.horizon_days,
            "show_user_view": decision.show_user_view,
            "days": days,
            "notes": decision.notes,
        }
