# events/event_service.py
from __future__ import annotations

import json
import uuid
from datetime import date
from typing import Any, Dict, List, Optional

import oracledb
from infra.db_server import get_db_conn
from events.event_types import EventType


def _read_lob(v):
    if v is None:
        return None
    if hasattr(v, "read"):
        try:
            v = v.read()
        except Exception:
            v = str(v)
    if isinstance(v, (bytes, bytearray)):
        try:
            v = v.decode("utf-8")
        except Exception:
            v = v.decode("utf-8", errors="ignore")
    return v


class EventService:
    """
    ✅ 단일 진실: event_log
    - weekly_meal_plan / routine_engine / replan_orchestrator 공용
    """

    def insert_event(
        self,
        *,
        user_id: bytes,
        event_date: date,
        event_type: str,
        severity: str,
        note: Optional[str] = None,
        raw_flags: Optional[Dict[str, Any]] = None,
    ) -> None:
        with get_db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO event_log (
                    event_id,
                    user_id,
                    event_date,
                    event_type,
                    severity,
                    note,
                    raw_flags
                ) VALUES (
                    :event_id,
                    :user_id,
                    :event_date,
                    :event_type,
                    :severity,
                    :note,
                    :raw_flags
                )
                """,
                {
                    "event_id": uuid.uuid4().bytes,
                    "user_id": oracledb.Binary(user_id),
                    "event_date": event_date,
                    "event_type": event_type,
                    "severity": severity,
                    "note": note,
                    "raw_flags": json.dumps(raw_flags or {}, ensure_ascii=False),
                },
            )
            conn.commit()

    def load_events_for_date(
        self,
        *,
        user_id: bytes,
        target_date: date,
    ) -> List[Dict[str, Any]]:
        with get_db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT event_type, severity, note, raw_flags
                FROM event_log
                WHERE user_id = :u
                  AND TRUNC(event_date) = TRUNC(:d)
                ORDER BY created_at
                """,
                {"u": oracledb.Binary(user_id), "d": target_date},
            )
            rows = cur.fetchall() or []

        events: List[Dict[str, Any]] = []
        for et, sev, note, raw in rows:
            s = _read_lob(raw)
            try:
                raw_flags = json.loads(s) if s else {}
            except Exception:
                raw_flags = {}

            # ✅ 표준화(핵심)
            std = EventType.from_any(et).value

            events.append(
                {
                    "event_type": et,                # 원문(한글/레거시)
                    "event_type_std": std,           # 표준(대문자)
                    "severity": sev,
                    "note": note,
                    "raw_flags": raw_flags,
                }
            )
        return events

    def build_event_flags(
        self,
        *,
        user_id: bytes,
        target_date: date,
    ) -> Dict[str, Any]:
        events = self.load_events_for_date(user_id=user_id, target_date=target_date)

        flags = {
            "events": events,
            "social": False,
            "overtime": False,
            "travel": False,
            "meeting": False,
            "sleep_debt": False,
        }

        for ev in events:
            std = str(ev.get("event_type_std") or "").strip()

            if std == EventType.DINNER_OUT.value:
                flags["social"] = True
            elif std == EventType.OVERTIME.value:
                flags["overtime"] = True
            elif std == EventType.TRAVEL.value:
                flags["travel"] = True
            elif std == EventType.MEETING.value:
                flags["meeting"] = True
            elif std == EventType.SLEEP_DEBT.value:
                flags["sleep_debt"] = True

        return flags
