# services/event_pipeline.py

from datetime import date

from services.event_service import insert_event_log
from services.engine_state_service import upsert_engine_state
from services.replan_run_service import insert_replan_run
from services.event_adjustment_service import EventAdjustmentService


def run_event_pipeline(
    user_id: bytes,
    event,
    diet_plan,
    exercise_plan,
    goal,
    engine_state: dict,
):
    """
    EVENT → ENGINE → REPLAN_RUN 전체 파이프라인
    """

    # 1️⃣ EVENT_LOG 저장
    insert_event_log(
        user_id=user_id,
        event_type=event.event_type,
        event_date=event.event_date,
        raw_flags=event.model_dump(),
    )

    # 2️⃣ ENGINE_STATE 갱신
    upsert_engine_state(
        user_id=user_id,
        as_of_date=date.today(),
        plateau_days=engine_state.get("plateau_days", 0),
        rolling_7d=engine_state.get("rolling_7d", {}),
        rolling_14d=engine_state.get("rolling_14d", {}),
    )

    # 3️⃣ 엔진 판단
    engine = EventAdjustmentService()
    result = engine.adjust_daily(
        event=event,
        diet_plan=diet_plan,
        exercise_plan=exercise_plan,
        goal=goal,
    )

    # 4️⃣ REPLAN_RUN 저장
    insert_replan_run(
        user_id=user_id,
        event_date=event.event_date,
        event_type=event.event_type,
        intensity=getattr(event, "intensity", None),
        horizon=getattr(result.get("diet"), "horizon", {}),
        actions={
            "diet": getattr(result.get("diet"), "actions", {}),
            "exercise": getattr(result.get("exercise"), "actions", {}),
        },
        cards={
            "feedback": result.get("feedback"),
            "daily_goal": result.get("daily_goal"),
        },
        status="applied",
        version="v1.0.0",
    )

    return result
