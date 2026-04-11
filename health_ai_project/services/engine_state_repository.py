# services/engine_state_repository.py
from __future__ import annotations

import json
from datetime import date
from typing import Dict, Any


SAVE_ENGINE_STATE_SQL = """
MERGE INTO engine_state s
USING (
    SELECT
        :user_id       AS user_id,
        :as_of_date    AS as_of_date,
        :plateau_days  AS plateau_days,
        :rolling_7d    AS rolling_7d,
        :rolling_14d   AS rolling_14d
    FROM dual
) src
ON (s.user_id = src.user_id)
WHEN MATCHED THEN
    UPDATE SET
        s.as_of_date   = src.as_of_date,
        s.plateau_days = src.plateau_days,
        s.rolling_7d   = src.rolling_7d,
        s.rolling_14d  = src.rolling_14d,
        s.updated_at   = CURRENT_TIMESTAMP
WHEN NOT MATCHED THEN
    INSERT (
        user_id,
        as_of_date,
        plateau_days,
        rolling_7d,
        rolling_14d,
        updated_at
    )
    VALUES (
        src.user_id,
        src.as_of_date,
        src.plateau_days,
        src.rolling_7d,
        src.rolling_14d,
        CURRENT_TIMESTAMP
    )
"""


def save_engine_state(
    conn,
    *,
    user_id: bytes,
    as_of_date: date,
    state: Dict[str, Any],
) -> None:
    """
    ENGINE STATE 저장 (단일 진실 소스)

    state 예:
    {
        "plateau_days": int,
        "rolling_7d": {...},
        "rolling_14d": {...}
    }
    """

    plateau_days = int(state.get("plateau_days", 0))
    rolling_7d = state.get("rolling_7d") or {}
    rolling_14d = state.get("rolling_14d") or {}

    cur = conn.cursor()
    cur.execute(
        SAVE_ENGINE_STATE_SQL,
        {
            "user_id": user_id,
            "as_of_date": as_of_date,
            "plateau_days": plateau_days,
            "rolling_7d": json.dumps(rolling_7d, ensure_ascii=False),
            "rolling_14d": json.dumps(rolling_14d, ensure_ascii=False),
        },
    )

    conn.commit()
