# services/user_goal_service.py
from __future__ import annotations

from datetime import date
import json
import uuid
from typing import Optional, Dict, Any

from infra.db_server import get_db_conn


def build_macro_target_from_kcal(kcal: int) -> dict:
    carb_ratio = 0.5
    protein_ratio = 0.25
    fat_ratio = 0.25

    return {
        "carb_g": int((kcal * carb_ratio) / 4),
        "protein_g": int((kcal * protein_ratio) / 4),
        "fat_g": int((kcal * fat_ratio) / 9),
    }


def upsert_active_user_goal(
    *,
    user_id: bytes,
    goal_type: str,
    kcal_target: int,
    input_macro_target: Optional[Dict[str, Any]] = None,
) -> None:
    # 1) macro_target 확정 (무조건 값 있게)
    macro_target = input_macro_target or {}
    if not macro_target:
        macro_target = build_macro_target_from_kcal(int(kcal_target))

    # 2) 키 통일 (혹시 carbs_g로 들어오면 carb_g로 정규화)
    if "carbs_g" in macro_target and "carb_g" not in macro_target:
        macro_target["carb_g"] = macro_target.pop("carbs_g")

    macro_json = json.dumps(macro_target, ensure_ascii=False)

    with get_db_conn() as conn:
        cur = conn.cursor()

        # 3) 기존 active goal 있으면 update, 없으면 insert
        cur.execute(
            """
            SELECT goal_id
            FROM user_goal
            WHERE user_id = :u
              AND is_active = 'Y'
            """,
            {"u": user_id},
        )
        row = cur.fetchone()

        if row:
            cur.execute(
                """
                UPDATE user_goal
                SET
                    goal_type   = :goal_type,
                    kcal_target = :kcal,
                    macro_target = :macro
                WHERE user_id = :u
                  AND is_active = 'Y'
                """,
                {"goal_type": goal_type, "kcal": int(kcal_target), "macro": macro_json, "u": user_id},
            )
        else:
            cur.execute(
                """
                INSERT INTO user_goal (
                    goal_id, user_id, goal_type,
                    start_date, kcal_target, macro_target, is_active
                )
                VALUES (
                    :goal_id, :u, :goal_type,
                    :start_date, :kcal, :macro, 'Y'
                )
                """,
                {
                    "goal_id": uuid.uuid4().bytes,
                    "u": user_id,
                    "goal_type": goal_type,
                    "start_date": date.today(),
                    "kcal": int(kcal_target),
                    "macro": macro_json,
                },
            )

        conn.commit()
