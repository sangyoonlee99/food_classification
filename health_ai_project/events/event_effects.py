# events/event_effects.py
# STEP 8-A-2 ~ 8-A-3 공용
# - EVENT_POLICY에서 "정적 정책"을 조회하여
# - Diet / Exercise replan 서비스가 이해할 수 있는
#   표준 dict 형태로 반환
# - 기간 분배, 가중치 계산은 HorizonEngine의 책임

from __future__ import annotations

from typing import Any, Dict

from events.event_policy import EVENT_POLICY
from events.event_types import EventType


def resolve_event_adjustments(event_type: EventType) -> Dict[str, Any]:
    """
    EVENT_POLICY에서 event_type에 해당하는 정책을 조회하여
    표준 형태로 반환한다.
    ⚠️ 주의:
    - 여기서는 '무엇을 조정할지'만 정의
    - '언제/며칠 동안/얼마나'는 HorizonEngine에서 처리
    """

    policy = EVENT_POLICY.get(event_type)

    if not policy:
        return {
            "diet": {},
            "exercise": {},
        }

    return {
        "diet": policy.get("diet", {}) or {},
        "exercise": policy.get("exercise", {}) or {},
    }
