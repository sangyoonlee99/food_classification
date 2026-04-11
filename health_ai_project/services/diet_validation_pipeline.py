# services/diet_validation_pipeline.py

from services.diet_validator_runner import run_daily_diet_validation
from services.replan_trigger import trigger_replan_if_needed


def run_diet_validation_pipeline(
    *,
    user_id: bytes,
    plan,
    profile,
    goal,
):
    result = run_daily_diet_validation(
        plan=plan,
        profile=profile,
        goal=goal,
    )

    trigger_replan_if_needed(
        user_id=user_id,
        validation_result=result,
    )

    return result
