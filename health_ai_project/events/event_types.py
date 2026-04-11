# events/event_types.py
from __future__ import annotations

from enum import Enum
from typing import Any


class EventType(str, Enum):
    # ✅ 표준 식별자(대문자): 내부 정책/엔진에서 사용
    DINNER_OUT = "DINNER_OUT"           # 회식/외식
    OVERTIME = "OVERTIME"               # 야근
    MEETING = "MEETING"                 # 모임
    SLEEP_DEBT = "SLEEP_DEBT"           # 수면부족
    EXTRA_EXERCISE = "EXTRA_EXERCISE"   # 운동량 증가
    TRAVEL = "TRAVEL"                   # 여행

    # ✅ legacy/세부 타입(기존 코드 호환)
    DINNER_DRINKING = "dinner_drinking"
    RESTAURANT_MEAL = "restaurant_meal"
    CELEBRATION = "celebration"
    OVEREATING = "overeating"
    LATE_MEAL = "late_meal"
    SLEEP_DEBT_LEGACY = "sleep_debt"

    @classmethod
    def from_any(cls, v: Any) -> "EventType":
        if isinstance(v, cls):
            return v
        if v is None:
            raise ValueError("event_type is required")

        s = str(v).strip()

        # 1) enum name (예: "DINNER_OUT")
        try:
            return cls[s]
        except KeyError:
            pass

        # 2) enum value (예: "dinner_drinking")
        for item in cls:
            if item.value == s:
                return item

        # 3) 별칭 매핑(한글/자주 쓰는 케이스)
        alias_map = {
            # dinner out
            "DINNEROUT": cls.DINNER_OUT,
            "DINNER_OUT": cls.DINNER_OUT,
            "dinner_out": cls.DINNER_OUT,
            "dining_out": cls.DINNER_OUT,
            "회식": cls.DINNER_OUT,
            "외식": cls.DINNER_OUT,

            # overtime
            "OVERTIME": cls.OVERTIME,
            "overtime": cls.OVERTIME,
            "야근": cls.OVERTIME,
            "late_meal": cls.OVERTIME,  # 야근/늦은식사 성격을 야근으로 묶고 싶으면 유지

            # meeting
            "MEETING": cls.MEETING,
            "meeting": cls.MEETING,
            "모임": cls.MEETING,

            # sleep debt
            "SLEEP_DEBT": cls.SLEEP_DEBT,
            "sleep_debt": cls.SLEEP_DEBT,
            "수면부족": cls.SLEEP_DEBT,

            # extra exercise / travel
            "EXTRA_EXERCISE": cls.EXTRA_EXERCISE,
            "extra_exercise": cls.EXTRA_EXERCISE,
            "TRAVEL": cls.TRAVEL,
            "travel": cls.TRAVEL,

            # legacy passthrough
            "dinner_drinking": cls.DINNER_DRINKING,
            "restaurant_meal": cls.RESTAURANT_MEAL,
            "celebration": cls.CELEBRATION,
            "overeating": cls.OVEREATING,
        }

        key = s.replace(" ", "")
        if key in alias_map:
            return alias_map[key]

        raise ValueError(f"Unsupported event_type: {v}")
