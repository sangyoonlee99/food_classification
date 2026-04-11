from __future__ import annotations

from datetime import datetime
from typing import Dict, Any
import uuid

from infra.db_server import get_db_conn


def insert_recommendation_history(
    *,
    user_id: bytes,
    signature: str,
    level: str,
    status: str,
    ui_state: str,
    variant: str = "A",
    source: str = "engine",
    created_at: datetime | None = None,
):
    """
    🔹 추천 히스토리 단건 저장
    (repeat / suppress / AB test 기반)
    """

    with get_db_conn() as conn:
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO recommendation_history (
                rec_id,
                user_id,
                signature,
                rec_level,
                status,
                ui_state,
                variant,
                source,
                created_at
            )
            VALUES (
                :rec_id,
                :user_id,
                :signature,
                :rec_level,
                :status,
                :ui_state,
                :variant,
                :source,
                :created_at
            )
        """, {
            "rec_id": uuid.uuid4().bytes,
            "user_id": user_id,
            "signature": signature,
            "rec_level": level,  # 👈 여기
            "status": status,
            "ui_state": ui_state,
            "variant": variant,
            "source": source,
            "created_at": created_at or datetime.now(),
        })

        conn.commit()
