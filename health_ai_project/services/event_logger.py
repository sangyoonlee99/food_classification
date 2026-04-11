# services/event_logger.py

from __future__ import annotations
import uuid
from datetime import date
from infra.db_server import get_db_conn


class EventLogger:
    def log_diet_event(
        self,
        *,
        user_id: bytes,
        event_type: str,
        severity: str,
        reasons: list[str],
    ):
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
                    note
                )
                VALUES (
                    :event_id,
                    :user_id,
                    :event_date,
                    :event_type,
                    :severity,
                    :note
                )
                """,
                {
                    "event_id": uuid.uuid4().bytes,
                    "user_id": user_id,
                    "event_date": date.today(),
                    "event_type": event_type,
                    "severity": severity,
                    "note": "; ".join(reasons),
                },
            )
            conn.commit()
