# infra/repositories/log_repository.py

from __future__ import annotations
from datetime import date
from typing import Dict, Any
from infra.db_server import get_db_conn


def load_event_flags(
    *,
    user_id: bytes,
    target_date: date,
) -> Dict[str, Any]:
    """
    event_log → RoutineEngine용 event_flags
    """
    flags: Dict[str, Any] = {}

    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT event_type, severity
            FROM event_log
            WHERE user_id = :user_id
              AND event_date = :d
            """,
            {"user_id": user_id, "d": target_date},
        )
        rows = cur.fetchall()

    for event_type, severity in rows:
        # 예: dining_out → True
        flags[event_type] = {
            "severity": severity or "normal"
        }

    return flags
