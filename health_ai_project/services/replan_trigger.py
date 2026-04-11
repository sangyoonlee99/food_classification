# services/replan_trigger.py

from __future__ import annotations
from services.engine_guard import can_trigger_replan
from services.event_logger import EventLogger
from events.replan_orchestrator import ReplanOrchestrator


def trigger_replan_if_needed(
    *,
    user_id: bytes,
    validation_result: dict,
):
    """
    STEP M-3-3 핵심 진입점
    """

    status = validation_result["status"]
    severity = validation_result["severity"]

    if status == "allowed":
        return  # 아무것도 안 함

    # 1️⃣ 이벤트 기록
    EventLogger().log_diet_event(
        user_id=user_id,
        event_type=f"diet_{status}",
        severity=severity,
        reasons=validation_result["reasons"],
    )

    # 2️⃣ replan 트리거 판단
    if severity in ("medium", "high") and can_trigger_replan(user_id):
        ReplanOrchestrator().run(user_id=user_id)
