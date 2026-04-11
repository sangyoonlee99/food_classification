# services/event_service.py
from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List, Optional, Any

import oracledb
from infra.db_server import get_db_conn
from events.event_types import EventType



def load_events_for_period(
    user_id: bytes,
    *,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    days: int = 14,
) -> List[Dict[str, Any]]:
    """
    ✅ legacy 호환 API
    - recommendation_layer 등에서 사용
    - 반환 형태: [{"type": <std>, "severity": <sev>, "raw": {...}}]
    """
    # 기본: 최근 N일
    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(days=days)

    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT event_type, severity
            FROM event_log
            WHERE user_id = :u
              AND TRUNC(event_date) BETWEEN TRUNC(:s) AND TRUNC(:e)
            ORDER BY created_at
            """,
            {
                "u": oracledb.Binary(user_id),
                "s": start_date,
                "e": end_date,
            },
        )
        rows = cur.fetchall() or []

    out: List[Dict[str, Any]] = []
    for et, sev in rows:
        try:
            std = EventType.from_any(et).value
        except Exception:
            std = str(et)

        out.append({"type": std, "severity": sev})
    return out


def has_protective_event(events: List[Dict[str, Any]]) -> bool:
    """
    ✅ legacy 호환
    - 보호 이벤트(예: 회식/여행) 있으면 엔진 억제/완화 등에서 사용
    """
    protective = {EventType.DINNER_OUT.value, EventType.TRAVEL.value}
    for e in events or []:
        t = str(e.get("type") or "").strip()
        if t in protective:
            return True
    return False

class EventService:
    """
    🔥 system 이벤트 생성/중복 방지용 서비스
    (record_meal.py에서 사용)
    """

    def exists_event(
        self,
        *,
        user_id: bytes,
        event_date: date,
        event_type: str,
        source: str,
    ) -> bool:
        with get_db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT COUNT(*)
                FROM event_log
                WHERE user_id = :u
                  AND TRUNC(event_date) = TRUNC(:d)
                  AND event_type = :t
                  AND source = :s
                """,
                {
                    "u": oracledb.Binary(user_id),
                    "d": event_date,
                    "t": event_type,
                    "s": source,
                },
            )
            return (cur.fetchone()[0] or 0) > 0

    def create_event(
        self,
        *,
        user_id: bytes,
        event_date: date,
        event_type: str,
        severity: Optional[str] = None,
        note: Optional[str] = None,
        source: str = "system",
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
                    source,
                    created_at
                )
                VALUES (
                    SYS_GUID(),
                    :u,
                    :d,
                    :t,
                    :sev,
                    :note,
                    :src,
                    SYSDATE
                )
                """,
                {
                    "u": oracledb.Binary(user_id),
                    "d": event_date,
                    "t": event_type,
                    "sev": severity,
                    "note": note,
                    "src": source,
                },
            )
            conn.commit()



# (선택) 기존 코드가 EventService를 services.*에서 찾는 경우 대비
__all__ = ["load_events_for_period", "has_protective_event", "EventService"]
