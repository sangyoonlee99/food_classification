from __future__ import annotations
from typing import Dict, Any, Optional

# -------------------------------------------------
# 이벤트 → 짧은 이유 문구 매핑
# -------------------------------------------------
EVENT_REASON_TEXT = {
    "overtime": "야근 일정이 있어 무리하지 않도록 했어요.",
    "dinner": "회식 일정이 있어 보수적으로 유지했어요.",
    "travel": "이동 일정이 많아 루틴 변경은 미뤘어요.",
    "sick": "컨디션 회복이 우선이에요.",
}

# -------------------------------------------------
# 내부: execution 결과 → 메시지 1개 변환
# -------------------------------------------------
def build_execution_message(
    execution: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:

    if not execution:
        return None

    event_type = execution.get("event_type")
    event_reason = EVENT_REASON_TEXT.get(event_type)

    # ===============================
    # 실행 스킵 (관찰 / 유지)
    # ===============================
    if execution.get("skipped"):
        reason = execution.get("reason")

        if reason == "cooldown":
            msg = {
                "title": "조금 더 지켜볼게요",
                "tone": "neutral",
                "badge": {"code": "COOLDOWN", "label": "관찰 중"},
            }
        elif reason == "repeated":
            msg = {
                "title": "이미 충분해요",
                "tone": "neutral",
                "badge": {"code": "ENOUGH", "label": "유지"},
            }
        else:
            msg = {
                "title": "오늘은 유지",
                "tone": "neutral",
            }

        if event_reason:
            msg["subtitle"] = event_reason

        return msg

    # ===============================
    # 실제 실행
    # ===============================
    level = execution.get("level", "micro")

    if level == "macro":
        msg = {
            "title": "조정이 필요해요",
            "tone": "warning",
            "badge": {"code": "ADJUST", "label": "전략 변경"},
        }
    else:
        msg = {
            "title": "가볍게 조정했어요",
            "tone": "positive",
            "badge": {"code": "MICRO", "label": "미세 조정"},
        }

    if event_reason:
        msg["subtitle"] = event_reason

    return msg


# -------------------------------------------------
# ✅ 외부(UI)에서 쓰는 공식 함수
# -------------------------------------------------
def get_today_ui_card(
    execution: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Home / Recommendation UI에서 사용하는 단일 카드 변환 함수
    """
    return build_execution_message(execution)
