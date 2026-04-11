# services/body_service.py
from __future__ import annotations
from datetime import datetime, date
import uuid
from infra.db_server import get_db_conn


class BodyService:
    @staticmethod
    def record_weight(*, user_id: bytes, weight_kg: float, measured_at: datetime):
        with get_db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO body_log (
                    body_log_id,
                    user_id,
                    measured_at,
                    weight_kg,
                    source
                ) VALUES (
                    :id,
                    :u,
                    :d,
                    :w,
                    'manual'
                )
                """,
                {
                    "id": uuid.uuid4().bytes,
                    "u": user_id,
                    "d": measured_at,
                    "w": weight_kg,
                },
            )
            conn.commit()

    @staticmethod
    def get_latest_weight(*, user_id: bytes):
        with get_db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT weight_kg
                FROM body_log
                WHERE user_id = :u
                ORDER BY measured_at DESC
                FETCH FIRST 1 ROWS ONLY
                """,
                {"u": user_id},
            )
            row = cur.fetchone()
            return float(row[0]) if row else None
