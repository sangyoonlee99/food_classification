from __future__ import annotations

from typing import Dict, Any, Tuple, Optional, Literal

RecLevel = Literal["keep", "micro", "macro"]


def evaluate_engine_state(
    state: Dict[str, Any]
) -> Tuple[RecLevel, Optional[Dict[str, Any]]]:
    """
    ENGINE_STATE 기반 추천 강도 / 배지 결정 (Single Source of Truth)

    state 예:
      {
        "plateau_days": 0~,
        "rolling_7d": {"trend": "better|flat|worse"},
        "rolling_14d": {"trend": "better|flat|worse"},
        "repeat_count": 0~,
      }
    """

    # -------------------------
    # 안전한 기본값
    # -------------------------
    if not state:
        return "keep", None

    plateau_days: int = int(state.get("plateau_days", 0) or 0)
    repeat_count: int = int(state.get("repeat_count", 0) or 0)

    rolling_7d: Dict[str, Any] = state.get("rolling_7d") or {}
    rolling_14d: Dict[str, Any] = state.get("rolling_14d") or {}

    trend_7d = rolling_7d.get("trend")
    trend_14d = rolling_14d.get("trend")

    # -------------------------
    # 0) 반복 과다 (UI 쿨다운)
    # -------------------------
    if repeat_count >= 2:
        return (
            "keep",
            {
                "code": "ENOUGH",
                "label": "이미 충분해요",
                "tone": "neutral",
            },
        )

    # -------------------------
    # 1) 추천 레벨 결정
    # -------------------------
    if plateau_days >= 14:
        rec_level: RecLevel = "macro"
    elif plateau_days >= 7:
        rec_level = "micro"
    else:
        rec_level = "keep"

    # -------------------------
    # 2) 배지 결정 (명확한 조건만)
    # -------------------------
    badge: Optional[Dict[str, Any]] = None

    if trend_14d == "worse":
        badge = {
            "code": "WARNING",
            "label": "조정이 필요해요",
            "tone": "warning",
        }
    elif trend_14d == "flat":
        badge = {
            "code": "STUCK",
            "label": "정체 상태예요",
            "tone": "neutral",
        }
    elif trend_7d == "better":
        badge = {
            "code": "GOOD",
            "label": "잘 진행 중이에요",
            "tone": "positive",
        }
    elif rec_level == "keep":
        badge = {
            "code": "KEEP",
            "label": "지금 흐름 좋아요",
            "tone": "neutral",
        }

    return rec_level, badge
