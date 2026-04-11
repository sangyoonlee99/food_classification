# services/diet_validator_runner.py

from __future__ import annotations
from datetime import date
from typing import Dict, Any

from diet.diet_validator import validate_daily_diet


def run_daily_diet_validation(
    *,
    plan,
    profile,
    goal,
) -> Dict[str, Any]:
    """
    STEP M-3 공용 진입점
    - validate_daily_diet 결과를 그대로 엔진/이벤트에 전달
    """

    result = validate_daily_diet(
        plan=plan,
        profile=profile,
        goal=goal,
    )

    return {
        "status": result["status"],          # allowed | over | under
        "severity": result["severity"],      # low | medium | high
        "reasons": result["reasons"],        # list[str]
        "metrics": result["metrics"],        # 수치 비교
        "is_valid": result["is_valid"],
    }
