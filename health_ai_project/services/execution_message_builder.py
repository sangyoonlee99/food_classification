from __future__ import annotations
from typing import Dict, Any


def build_execution_message(execution: Dict[str, Any] | None) -> Dict[str, Any] | None:
    """
    execution 결과를 UI 메시지로 변환
    """

    if not execution:
        return None

    # ----------------------------------
    # 실행 차단 (cooldown)
    # ----------------------------------
    if execution.get("skipped") and execution.get("reason") == "cooldown":
        return {
            "type": "info",
            "tone": "neutral",
            "title": "조정은 이미 반영 중이에요",
            "body": "최근에 조정이 적용되어 있어요. 며칠간은 현재 루틴을 유지해보세요.",
        }

    # ----------------------------------
    # 실제 실행됨
    # ----------------------------------
    if execution.get("applied"):
        level = execution.get("level", "micro")

        if level == "macro":
            return {
                "type": "action",
                "tone": "warning",
                "title": "루틴이 조정되었어요",
                "body": "정체가 길어져 식단이나 운동 구성이 일부 변경되었어요.",
            }

        return {
            "type": "action",
            "tone": "neutral",
            "title": "가벼운 조정이 적용됐어요",
            "body": "오늘은 부담 없이 작은 변화만 반영했어요.",
        }

    # ----------------------------------
    # fallback
    # ----------------------------------
    return None
