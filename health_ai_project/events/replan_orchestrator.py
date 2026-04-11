from __future__ import annotations

from datetime import datetime, date
from typing import Any, Dict, List, Optional

from events.replan_horizon_service import ReplanHorizonService
from routine.routine_engine import RoutineEngine
from events.replan_logger import ReplanLogger
from services.message_layer import build_replan_messages

# Execution / Policy
from services.recommendation_parser import parse_signature
from services.action_executor import execute_actions
from services.engine_cooldown_policy import record_execution
from services.engine_execution_policy import evaluate_execution_policy

# Exercise
from services.exercise_adapter import ExerciseAdapter
from exercise.exercise_planner import plan_daily_exercise
from exercise.exercise_adjuster import adjust_exercise


class ReplanOrchestrator:
    """
    FINAL (운영)
    """

    def __init__(
        self,
        horizon_service: Optional[ReplanHorizonService] = None,
        routine_engine: Optional[RoutineEngine] = None,
        logger: Optional[ReplanLogger] = None,
    ):
        self.horizon_service = horizon_service or ReplanHorizonService()
        self.routine_engine = routine_engine or RoutineEngine()
        self.logger = logger or ReplanLogger()
        self.exercise_adapter = ExerciseAdapter()

    def build(
        self,
        *,
        user_id: bytes,
        event_date: date,
        event_type: Optional[str],
        intensity: Optional[str],
        base_week_dates: List[date],
        user_settings: Dict[str, Any],
        event_flags: Dict[str, Any],
        state: Dict[str, Any],
        actual_delta_kcal: Optional[float] = None,
        estimated_delta_kcal: float = 400.0,
    ) -> Dict[str, Any]:

        # ==================================================
        # 🔥 0️⃣ 목표 변경 감지 → 엔진 상태 리셋 (핵심)
        # ==================================================
        if user_settings.get("goal_changed") is True:
            state = {}  # plateau / cooldown / signature 전부 초기화

        # ==================================================
        # 1️⃣ Horizon
        # ==================================================
        horizon = self.horizon_service.build_horizon_adjustments(
            event_date=event_date,
            event_type=event_type,
            intensity=intensity,
            base_week_dates=base_week_dates,
            user_settings=user_settings,
            actual_delta_kcal=actual_delta_kcal,
            estimated_delta_kcal=estimated_delta_kcal,
        )

        # ==================================================
        # 2️⃣ Exercise planning
        # ==================================================
        planned = plan_daily_exercise(user_settings)
        recommendations = planned.get("recommendations", [])

        adjusted = None
        if recommendations:
            adjusted = adjust_exercise(
                planned_exercise=recommendations[0],
                events=[{"type": event_type}] if event_type else [],
                plateau_days=state.get("plateau_days", 0),
            )

        daily_exercise = self.exercise_adapter.get_daily_plan(
            user_id=user_id,
            date=event_date,
        )

        state = {
            **state,
            "exercise_planned": planned,
            "exercise_adjusted": adjusted,
            "exercise_actual": (
                daily_exercise.dict()
                if daily_exercise and hasattr(daily_exercise, "dict")
                else None
            ),
        }

        # ==================================================
        # 3️⃣ Execution Policy (기존 시그니처 있을 때만)
        # ==================================================
        signature = state.get("recommendation_signature")
        execution = None

        if signature:
            policy = evaluate_execution_policy(
                state=state,
                signature=signature,
                today=event_date,
            )

            if policy["allowed"]:
                parsed = parse_signature(signature)
                execution = execute_actions(parsed, state)
                state = self.routine_engine.apply_execution(
                    execution=execution,
                    state=state,
                )

                if policy.get("cooldown_days"):
                    state = record_execution(
                        state=state,
                        signature=signature,
                        today=event_date,
                        cooldown_days=policy["cooldown_days"],
                    )
            else:
                execution = {
                    "skipped": True,
                    "reason": policy["reason"],
                }

        # ==================================================
        # 4️⃣ Actions (🔥 단일 진실 / 최신 목표 기준)
        # ==================================================
        actions = self.routine_engine.build_actions(
            user_id=user_id,
            target_date=event_date,
            event_flags=event_flags,
            state=state,
        )

        # ==================================================
        # 5️⃣ Result (🔥 B안 핵심 수정)
        # ==================================================

        # 🔥 horizon kcal delta → event_flags에 주입
        horizon_kcal_map = horizon.get("daily_kcal_delta", {}) if isinstance(horizon, dict) else {}

        actions["meal"]["event_flags"] = {
            **(actions.get("meal", {}).get("event_flags") or {}),
            "horizon_kcal_delta": horizon_kcal_map,
        }

        result = {
            "meta": {
                "generated_at": datetime.utcnow().isoformat(),
                "event_date": event_date.isoformat(),
            },
            "horizon": horizon,
            "execution": execution,
            "actions": actions,
            "recommendation_signature": actions.get("recommendation_signature"),
        }

