# services/engine_cooldown_policy.py
from __future__ import annotations

from datetime import date
from typing import Dict, Any


def is_in_cooldown(
    *,
    state: Dict[str, Any],
    signature: str,
    today: date,
) -> bool:
    """
    동일 signature에 대해 cooldown 기간 내인지 판단
    """

    last = state.get("last_execution")
    if not last:
        return False

    if last.get("signature") != signature:
        return False

    try:
        executed_at = date.fromisoformat(last["executed_at"])
        cooldown_days = int(last.get("cooldown_days", 0))
    except Exception:
        return False

    return (today - executed_at).days < cooldown_days


def record_execution(
    *,
    state: Dict[str, Any],
    signature: str,
    today: date,
    cooldown_days: int,
) -> Dict[str, Any]:
    """
    실행 기록을 state에 반영
    """

    return {
        **state,
        "last_execution": {
            "signature": signature,
            "executed_at": today.isoformat(),
            "cooldown_days": int(cooldown_days),
        },
    }
