# services/engine_state_updater.py
from __future__ import annotations

from typing import Dict, Any
from datetime import date


def update_engine_state(
    *,
    prev_state: Dict[str, Any],
    today_metrics: Dict[str, Any],
    as_of_date: date,
) -> Dict[str, Any]:
    """
    ENGINE STATE 갱신 규칙 (Single Source of Truth)

    prev_state:
      {
        "plateau_days": int,
        "rolling_7d": {...},
        "rolling_14d": {...},
        "repeat_count": int,
        "last_signature": str,   # "UI signature" 기준 권장
        "last_update": "YYYY-MM-DD"
      }

    today_metrics:
      {
        "weight_delta": float | None,
        "kcal_delta": float | None,
        "signature": str | None,   # 🔥 UI signature ("diet:adjust" 등)
      }
    """

    prev_plateau = int(prev_state.get("plateau_days", 0) or 0)
    prev_7d = prev_state.get("rolling_7d") or {}
    prev_14d = prev_state.get("rolling_14d") or {}
    prev_repeat = int(prev_state.get("repeat_count", 0) or 0)
    prev_signature = prev_state.get("last_signature")

    weight_delta = today_metrics.get("weight_delta")
    today_signature = today_metrics.get("signature")

    # 1) plateau_days
    if weight_delta is None:
        plateau_days = prev_plateau
    elif abs(float(weight_delta)) < 0.1:
        plateau_days = prev_plateau + 1
    else:
        plateau_days = 0

    # 2) rolling
    rolling_7d = _update_rolling(prev=prev_7d, delta=weight_delta, window=7)
    rolling_14d = _update_rolling(prev=prev_14d, delta=weight_delta, window=14)

    # 3) repeat_count (UI signature 기준)
    if today_signature and today_signature == prev_signature:
        repeat_count = prev_repeat + 1
    else:
        repeat_count = 0

    return {
        "plateau_days": plateau_days,
        "rolling_7d": rolling_7d,
        "rolling_14d": rolling_14d,
        "repeat_count": repeat_count,
        "last_signature": today_signature,
        "last_update": as_of_date.isoformat(),
    }


def _update_rolling(*, prev: Dict[str, Any], delta: float | None, window: int) -> Dict[str, Any]:
    if delta is None:
        return prev or {}

    d = float(delta)
    if d < -0.2:
        trend = "better"
    elif d > 0.2:
        trend = "worse"
    else:
        trend = "flat"

    return {"window": window, "trend": trend}
