# services/engine_guard.py

from datetime import date
from infra.db_server import get_db_conn


def can_trigger_replan(user_id: bytes) -> bool:
    """
    하루 1회 replan 제한
    """
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT as_of_date
            FROM engine_state
            WHERE user_id = :user_id
            """,
            {"user_id": user_id},
        )
        row = cur.fetchone()

    if not row:
        return True

    last_date = row[0]
    return last_date != date.today()
