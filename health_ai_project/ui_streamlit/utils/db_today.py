from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Dict, Optional

from infra.db_server import get_db_conn


# ==================================================
# Utils
# ==================================================
def _day_range(d: date) -> tuple[datetime, datetime]:
    start_dt = datetime(d.year, d.month, d.day)
    end_dt = start_dt + timedelta(days=1)
    return start_dt, end_dt


def _table_exists(conn, table_name: str) -> bool:
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM user_tables WHERE table_name = :table_name",
        {"table_name": table_name.upper()},
    )
    return cur.fetchone() is not None


# ==================================================
# Today Counts
# ==================================================
def fetch_today_counts(user_id: bytes, d: date) -> Dict[str, int]:
    start_dt, end_dt = _day_range(d)

    with get_db_conn() as conn:
        cur = conn.cursor()

        def _count(sql: str) -> int:
            cur.execute(
                sql,
                {
                    "user_id": user_id,
                    "start_dt": start_dt,
                    "end_dt": end_dt,
                },
            )
            return int(cur.fetchone()[0] or 0)

        meal = (
            _count(
                """
                SELECT COUNT(*)
                FROM meal_record
                WHERE user_id = :user_id
                  AND eaten_at >= :start_dt
                  AND eaten_at <  :end_dt
                """
            )
            if _table_exists(conn, "MEAL_RECORD")
            else 0
        )

        exercise = (
            _count(
                """
                SELECT COUNT(*)
                FROM exercise_record
                WHERE user_id = :user_id
                  AND performed_at >= :start_dt
                  AND performed_at <  :end_dt
                """
            )
            if _table_exists(conn, "EXERCISE_RECORD")
            else 0
        )

        event = (
            _count(
                """
                SELECT COUNT(*)
                FROM event_log
                WHERE user_id = :user_id
                  AND event_date >= :start_dt
                  AND event_date <  :end_dt
                """
            )
            if _table_exists(conn, "EVENT_LOG")
            else 0
        )

    return {"meal": meal, "exercise": exercise, "event": event}


# ==================================================
# Latest Event
# ==================================================
def fetch_latest_event(user_id: bytes, d: date) -> Optional[dict]:
    start_dt, end_dt = _day_range(d)

    with get_db_conn() as conn:
        if not _table_exists(conn, "EVENT_LOG"):
            return None

        cur = conn.cursor()
        cur.execute(
            """
            SELECT event_type
            FROM (
                SELECT event_type
                FROM event_log
                WHERE user_id = :user_id
                  AND event_date >= :start_dt
                  AND event_date <  :end_dt
                ORDER BY created_at DESC
            )
            WHERE ROWNUM = 1
            """,
            {
                "user_id": user_id,
                "start_dt": start_dt,
                "end_dt": end_dt,
            },
        )

        row = cur.fetchone()
        return {"event_type": row[0]} if row else None


# ==================================================
# Today Meal Summary
# ==================================================
def fetch_today_meal_summary(user_id: bytes, d: date) -> Dict[str, float]:
    start_dt, end_dt = _day_range(d)

    with get_db_conn() as conn:
        if not _table_exists(conn, "MEAL_RECORD"):
            return dict(meal_cnt=0, kcal=0, carb=0, protein=0, fat=0)

        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                COUNT(*),
                NVL(SUM(kcal), 0),
                NVL(SUM(carb_g), 0),
                NVL(SUM(protein_g), 0),
                NVL(SUM(fat_g), 0)
            FROM meal_record
            WHERE user_id = :user_id
              AND eaten_at >= :start_dt
              AND eaten_at <  :end_dt
            """,
            {
                "user_id": user_id,
                "start_dt": start_dt,
                "end_dt": end_dt,
            },
        )

        cnt, kcal, carb, protein, fat = cur.fetchone()

    return {
        "meal_cnt": int(cnt or 0),
        "kcal": float(kcal),
        "carb": float(carb),
        "protein": float(protein),
        "fat": float(fat),
    }
