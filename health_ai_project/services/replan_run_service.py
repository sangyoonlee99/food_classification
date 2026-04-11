# services/replan_run_service.py

import uuid
import json
from datetime import datetime, date
from infra.db_server import get_db_conn


def insert_replan_run(
    user_id: bytes,
    event_date: date | None,
    event_type: str | None,
    intensity: str | None,
    horizon: dict,
    actions: dict,
    cards: dict,
    status: str = "applied",
    version: str = "v1.0.0",
):
    """
    Replan 실행 결과 전체 스냅샷 저장
    """
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO replan_run (
                replan_id,
                user_id,
                generated_at,
                event_date,
                event_type,
                intensity,
                horizon,
                actions,
                status,
                cards,
                version
            ) VALUES (
                :replan_id,
                :user_id,
                :generated_at,
                :event_date,
                :event_type,
                :intensity,
                :horizon,
                :actions,
                :status,
                :cards,
                :version
            )
        """, {
            "replan_id": uuid.uuid4().bytes,
            "user_id": user_id,
            "generated_at": datetime.now(),
            "event_date": event_date,
            "event_type": event_type,
            "intensity": intensity,
            "horizon": json.dumps(horizon, ensure_ascii=False),
            "actions": json.dumps(actions, ensure_ascii=False),
            "status": status,
            "cards": json.dumps(cards, ensure_ascii=False),
            "version": version,
        })
        conn.commit()

