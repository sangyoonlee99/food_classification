# services/engine_state_service.py
from __future__ import annotations

import json
from datetime import date
from typing import Dict, Any

from infra.db_server import get_db_conn
from services.engine_state_updater import update_engine_state


def _clob_to_str(val):
    """
    Oracle CLOB → str 안전 변환
    """
    if val is None:
        return None
    if hasattr(val, "read"):
        return val.read()
    return val


def update_and_save_engine_state(
    *,
    user_id: bytes,
    as_of_date: date,
    today_metrics: Dict[str, Any],
) -> Dict[str, Any]:

    prev_state = _load_engine_state(user_id)

    next_state = update_engine_state(
        prev_state=prev_state,
        today_metrics=today_metrics,
        as_of_date=as_of_date,
    )

    _upsert_engine_state(
        user_id=user_id,
        as_of_date=as_of_date,
        state=next_state,
    )

    return next_state


def _load_engine_state(user_id: bytes) -> Dict[str, Any]:
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT
                plateau_days,
                rolling_7d,
                rolling_14d,
                as_of_date
            FROM engine_state
            WHERE user_id = :user_id
        """, {"user_id": user_id})

        row = cur.fetchone()

    if not row:
        return {}

    rolling_7d = _clob_to_str(row[1])
    rolling_14d = _clob_to_str(row[2])

    return {
        "plateau_days": row[0] or 0,
        "rolling_7d": json.loads(rolling_7d) if rolling_7d else {},
        "rolling_14d": json.loads(rolling_14d) if rolling_14d else {},
        "last_update": row[3].isoformat() if row[3] else None,
    }


def _upsert_engine_state(
    *,
    user_id: bytes,
    as_of_date: date,
    state: Dict[str, Any],
) -> None:
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            MERGE INTO engine_state e
            USING dual
            ON (e.user_id = :user_id)
            WHEN MATCHED THEN UPDATE SET
                as_of_date     = :as_of_date,
                plateau_days  = :plateau_days,
                rolling_7d    = :rolling_7d,
                rolling_14d   = :rolling_14d,
                updated_at    = CURRENT_TIMESTAMP
            WHEN NOT MATCHED THEN INSERT
                (user_id, as_of_date, plateau_days, rolling_7d, rolling_14d)
            VALUES
                (:user_id, :as_of_date, :plateau_days, :rolling_7d, :rolling_14d)
        """, {
            "user_id": user_id,
            "as_of_date": as_of_date,
            "plateau_days": int(state.get("plateau_days", 0)),
            "rolling_7d": json.dumps(state.get("rolling_7d", {}), ensure_ascii=False),
            "rolling_14d": json.dumps(state.get("rolling_14d", {}), ensure_ascii=False),
        })
        conn.commit()
