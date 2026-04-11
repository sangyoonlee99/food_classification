# scheduler/replan_scheduler.py
from __future__ import annotations

print("🔥🔥🔥 scheduler/replan_scheduler.py LOADED 🔥🔥🔥")

from datetime import date, timedelta
from typing import Dict, Any, Optional

from events.replan_orchestrator import ReplanOrchestrator
from services.engine_state_service import update_and_save_engine_state


class ReplanScheduler:
    """
    Step 8-A-14 (FINAL)
    - 자동 Replan 스케줄러
    - ENGINE STATE 단일 갱신 지점
    """

    def __init__(
        self,
        orchestrator: Optional[ReplanOrchestrator] = None,
    ):
        self.orchestrator = orchestrator or ReplanOrchestrator()

    # -------------------------------------------------
    # 1️⃣ 매일 자동 Replan
    # -------------------------------------------------
    def run_daily(
        self,
        *,
        user_id: bytes,
        today: date,
        user_settings: Dict[str, Any],
        event_flags: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        매일 1회 자동 실행 (평상시)
        """

        base_week_dates = [today + timedelta(days=i) for i in range(7)]

        # -------------------------------------------------
        # 1️⃣ Replan 실행 (이전 state 기준)
        # -------------------------------------------------
        # scheduler/replan_scheduler.py

        result = self.orchestrator.build(
            user_id=user_id,
            event_date=today,

            # ✅ 이벤트 없으면 None
            event_type=None,
            intensity=None,

            base_week_dates=base_week_dates,
            user_settings=user_settings,
            event_flags=event_flags,
            state=state,
        )

        # -------------------------------------------------
        # 2️⃣ 오늘 메트릭 정리 (운영 기준)
        # -------------------------------------------------
        today_metrics = {
            "weight_delta": state.get("weight_delta"),
            "kcal_delta": state.get("kcal_delta"),
            "signature": (
                result
                .get("messages", {})
                .get("ctx", {})
                .get("recommendation", {})
                .get("signature")
            ),
        }

        # -------------------------------------------------
        # 3️⃣ ENGINE STATE 갱신 + 저장 (🔥 단일 진실)
        # -------------------------------------------------
        next_state = update_and_save_engine_state(
            user_id=user_id,
            as_of_date=today,
            today_metrics=today_metrics,
        )

        return {
            **result,
            "engine_state": next_state,
        }

    # -------------------------------------------------
    # 2️⃣ 이벤트 기반 Replan
    # -------------------------------------------------
    def run_on_event(
        self,
        *,
        user_id: bytes,
        event_date: date,
        event_type: str,
        intensity: Optional[str],
        user_settings: Dict[str, Any],
        event_flags: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        명시적 이벤트 발생 시
        """

        base_week_dates = [event_date + timedelta(days=i) for i in range(7)]

        result = self.orchestrator.build(
            event_date=event_date,
            event_type=event_type,
            intensity=intensity,
            base_week_dates=base_week_dates,
            user_settings={**user_settings, "user_id": user_id},
            event_flags=event_flags,
            state=state,
        )

        today_metrics = {
            "weight_delta": state.get("weight_delta"),
            "kcal_delta": state.get("kcal_delta"),
            "signature": (
                result
                .get("messages", {})
                .get("ctx", {})
                .get("recommendation", {})
                .get("signature")
            ),
        }

        next_state = update_and_save_engine_state(
            user_id=user_id,
            as_of_date=event_date,
            today_metrics=today_metrics,
        )

        return {
            **result,
            "engine_state": next_state,
        }
