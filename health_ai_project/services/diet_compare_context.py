# services/diet_compare_context.py
from __future__ import annotations

from datetime import date
from typing import Dict, Any

from infra.db_server import get_db_conn


def build_diet_compare_context(*, user_id: bytes, target_date: date) -> Dict[str, Any]:
    """
    STEP M-3-1
    - AI 가이드(목표) vs 실제 섭취를
      diet_validator 입력용 단일 context로 정규화
    """

    # =========================
    # 1) 목표 (AI 가이드 기준)
    # =========================
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                kcal_target,
                macro_target
            FROM user_goal
            WHERE user_id = :user_id
              AND is_active = 'Y'
            """,
            {"user_id": user_id},
        )
        row = cur.fetchone()

    if not row:
        raise RuntimeError("Active user_goal not found")

    kcal_target, macro_json = row

    macro_target = {}
    if macro_json:
        if hasattr(macro_json, "read"):  # CLOB
            macro_json = macro_json.read()
        import json
        macro_target = json.loads(macro_json)

    # =========================
    # 2) 실제 섭취 (오늘 요약)
    # =========================
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                total_kcal,
                carb_g,
                protein_g,
                fat_g
            FROM daily_meal_summary
            WHERE user_id = :user_id
              AND summary_date = :d
            """,
            {"user_id": user_id, "d": target_date},
        )
        row = cur.fetchone()

    if not row:
        # 기록 없으면 0으로 판단 (❌ 예외 아님)
        total_kcal = carb_g = protein_g = fat_g = 0.0
    else:
        total_kcal, carb_g, protein_g, fat_g = row

    # =========================
    # 3) 🔥 Validator 입력 포맷 (고정)
    # =========================
    context = {
        "date": target_date.isoformat(),
        "target": {
            "kcal": float(kcal_target),
            "macro": macro_target,  # ratio / grams 포함 가능
        },
        "actual": {
            "kcal": float(total_kcal),
            "macro": {
                "carb": float(carb_g or 0),
                "protein": float(protein_g or 0),
                "fat": float(fat_g or 0),
            },
        },
    }

    return context
