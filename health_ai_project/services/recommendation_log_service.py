from __future__ import annotations

from typing import List, Optional
from datetime import date

from infra.db_server import get_db_conn


# =================================================
# A-7-9-1️⃣ 오늘 동일 signature 존재 여부 확인
# =================================================
def _exists_today_recommendation(
    *,
    user_id: bytes,
    signature: str,
    rec_date: Optional[date] = None,
) -> bool:
    """
    같은 사용자 + 같은 날짜 + 같은 signature
    추천 로그가 이미 있는지 확인
    """

    sql = """
    SELECT 1
    FROM recommendation_log
    WHERE user_id = :user_id
      AND signature = :signature
      AND rec_date = COALESCE(:rec_date, TRUNC(SYSDATE))
      AND ROWNUM = 1
    """

    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            sql,
            {
                "user_id": user_id,
                "signature": signature,
                "rec_date": rec_date,
            },
        )
        return cur.fetchone() is not None


# =================================================
# 1️⃣ 추천 로그 저장 (중복 방지 포함)
# =================================================
def insert_recommendation_log(
    *,
    user_id: bytes,
    signature: str,
    level: str,
    status: str,
    mode: str = "normal",
    source: str = "engine",
    rec_date: Optional[date] = None,
) -> None:
    """
    추천 결과를 DB에 기록

    ✔ A-7-9 적용:
      - 같은 날 + 같은 signature 이미 있으면 INSERT 스킵
    """

    # -----------------------------
    # A-7-9: 중복 방지
    # -----------------------------
    if _exists_today_recommendation(
        user_id=user_id,
        signature=signature,
        rec_date=rec_date,
    ):
        return  # 🔕 조용히 스킵 (UX/엔진 흐름 방해 없음)

    sql = """
    INSERT INTO recommendation_log (
        log_id,
        user_id,
        rec_date,
        signature,
        rec_level,
        rec_mode,
        status,
        source
    )
    VALUES (
        NULL,
        :user_id,
        COALESCE(:rec_date, TRUNC(SYSDATE)),
        :signature,
        :rec_level,
        :rec_mode,
        :status,
        :source
    )
    """

    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            sql,
            {
                "user_id": user_id,
                "rec_date": rec_date,
                "signature": signature,
                "rec_level": level,
                "rec_mode": mode,
                "status": status,
                "source": source,
            },
        )
        conn.commit()


# =================================================
# 2️⃣ 최근 추천 signature 조회
# =================================================
def get_recent_recommendation_signatures(
    *,
    user_id: bytes,
    days: int = 7,
    limit: int = 5,
) -> List[str]:
    """
    최근 N일간 사용자에게 제공된 추천 signature 목록
    (반복 판단용)
    """

    sql = """
    SELECT signature
    FROM (
        SELECT signature
        FROM recommendation_log
        WHERE user_id = :user_id
          AND rec_date >= TRUNC(SYSDATE) - :days
        ORDER BY created_at DESC
    )
    WHERE ROWNUM <= :limit
    """

    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            sql,
            {
                "user_id": user_id,
                "days": days,
                "limit": limit,
            },
        )
        rows = cur.fetchall()

    return [r[0] for r in rows]
