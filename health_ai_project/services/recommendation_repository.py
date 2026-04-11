# services/recommendation_repository.py
from __future__ import annotations

from typing import Dict, Any, Optional
from datetime import datetime
from uuid import uuid4

from infra.db_server import get_db_conn


def save_recommendation_history(
    *,
    user_id: bytes,
    recommendation_result: Dict[str, Any],
    variant: str = "R-5",
    source: str = "engine",
) -> None:
    """
    recommendation_layer 결과를 recommendation_history 테이블에 1 row 저장
    """
    ctx = recommendation_result.get("ctx", {}) or {}
    rec = ctx.get("recommendation", {}) or {}

    status = recommendation_result.get("status")
    signature = rec.get("signature")
    rec_level = rec.get("level")
    ui_state = rec.get("state")

    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
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
            """,
            {
                "rec_id": uuid4().bytes,   # RAW(16)
                "user_id": user_id,        # RAW(16)
                "signature": signature,
                "rec_level": rec_level,
                "status": status,
                "ui_state": ui_state,
                "variant": variant,
                "source": source,
                "created_at": datetime.now(),
            },
        )
        conn.commit()


def load_last_recommendation(*, user_id: bytes) -> Optional[Dict[str, Any]]:
    """
    recommendation_history에서 user_id 기준 최신 1건 + 연속 반복 count 계산
    반환:
      {"signature": "...", "count": <연속반복>, "created_at": datetime, "variant": "...", "source": "..."}
    """
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT signature, created_at, variant, source
              FROM recommendation_history
             WHERE user_id = :user_id
             ORDER BY created_at DESC
            """,
            {"user_id": user_id},
        )
        rows = cur.fetchmany(20)  # 최근 20개 내에서 연속반복만 계산(충분)

    if not rows:
        return None

    last_sig, last_created, last_variant, last_source = rows[0]

    # 연속 반복 count 계산: 최신 signature와 같은게 몇 개 연속으로 이어지는지
    cnt = 0
    for sig, *_ in rows:
        if sig == last_sig:
            cnt += 1
        else:
            break

    return {
        "signature": last_sig,
        "count": cnt - 1 if cnt > 0 else 0,  # layer에서는 "이전까지 몇 번 반복" 형태가 편함
        "created_at": last_created,
        "variant": last_variant,
        "source": last_source,
    }
