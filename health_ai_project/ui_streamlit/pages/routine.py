# ui_streamlit/pages/routine.py
from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st

# ==================================================
# Path
# ==================================================
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ==================================================
# Imports
# ==================================================
from ui_streamlit.utils.auth import require_login
from ui_streamlit.components import (
    app_container_start,
    app_container_end,
    card_start,
    card_end,
    muted_text,
    spacer,
    bottom_nav,
)
from infra.db_server import get_db_conn
from services.recommendation_history_loader import load_recommendation_state
from services.daily_exercise_guide_service import DailyExerciseGuideService


# ==================================================
# Utils
# ==================================================
def _ensure_bytes16(v) -> bytes:
    if isinstance(v, (bytes, bytearray)):
        return bytes(v)
    if isinstance(v, str):
        return bytes.fromhex(v.replace("0x", "").strip())
    raise TypeError("invalid user_id")


def _get_active_goal_kcal(user_id: bytes) -> int | None:
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT kcal_target
            FROM user_goal
            WHERE user_id = :u
              AND is_active = 'Y'
            """,
            {"u": user_id},
        )
        r = cur.fetchone()
    return int(r[0]) if r and r[0] is not None else None


def _sum_today_kcal(user_id: bytes) -> int:
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT NVL(SUM(kcal),0)
            FROM meal_record
            WHERE user_id = :u
              AND TRUNC(eaten_at) = TRUNC(SYSDATE)
            """,
            {"u": user_id},
        )
        r = cur.fetchone()
    return int(r[0] or 0)


# ==================================================
# Auth
# ==================================================
require_login()
user_id_raw = _ensure_bytes16(st.session_state["user_id"])

# ==================================================
# Recommendation state (유지용)
# ==================================================
rec_state = load_recommendation_state(user_id=user_id_raw) or {}
ctx = (rec_state.get("ctx") or {}).get("recommendation", {}) or {}

# ==================================================
# Today summary
# ==================================================
goal_kcal = _get_active_goal_kcal(user_id_raw)
kcal_today = _sum_today_kcal(user_id_raw)

exercise_guide = DailyExerciseGuideService().build_today_guide(
    user_id=user_id_raw
) or {}

ex_today = int(exercise_guide.get("today_minutes", 0))
ex_target = int(exercise_guide.get("target_minutes", 0))
ex_message = exercise_guide.get("message", "오늘은 가벼운 활동을 추천드려요.")
pref = exercise_guide.get("preferred", {})

# --------------------------------------------------
# 식사 가이드
# --------------------------------------------------
if goal_kcal:
    margin = max(int(goal_kcal * 0.1), 150)
    if kcal_today < goal_kcal - margin:
        meal_guide = "섭취량이 부족합니다. 남은 끼니를 균형 있게 보강해 주세요."
    elif kcal_today > goal_kcal + margin:
        meal_guide = "섭취량이 높은 편입니다. 남은 끼니는 가볍게 조절하세요."
    else:
        meal_guide = "오늘 식사 흐름은 안정적입니다."
else:
    meal_guide = "오늘 식사는 현재 흐름을 유지하는 것이 좋겠습니다."

# ==================================================
# UI
# ==================================================
app_container_start()

st.markdown("## 📋 오늘의 루틴")
spacer(6)

# ==================================================
# 🧭 오늘의 실행 요약
# ==================================================
card_start("🧭 오늘의 실행 루틴")

st.markdown(
    f"""
- 🍽️ **식사**: {meal_guide}
- 🏃 **운동**: {ex_message}
"""
)

if pref.get("cardio") or pref.get("strength"):
    muted_text(
        "선호 운동: "
        + " · ".join(
            [p for p in [pref.get("cardio"), pref.get("strength")] if p]
        )
    )

card_end()
spacer(10)

# ==================================================
# 🍽️ 오늘 식사 루틴
# ==================================================
card_start("🍽️ 오늘 식사 루틴")

st.markdown(f"- 목표 섭취 열량: **{goal_kcal or '-'} kcal**")
st.markdown(f"- 오늘 섭취 열량: **{kcal_today} kcal**")
st.markdown(f"- 가이드: {meal_guide}")

card_end()
spacer(10)

# ==================================================
# 🏃 오늘 운동 루틴
# ==================================================
card_start("🏃 오늘 운동 루틴")

st.markdown(f"- 목표 운동 시간: **{ex_target} 분**")
st.markdown(f"- 오늘 운동 시간: **{ex_today} 분**")
st.markdown(f"- 가이드: {ex_message}")

if pref.get("cardio") or pref.get("strength"):
    muted_text(
        "추천 운동: "
        + " · ".join(
            [p for p in [pref.get("cardio"), pref.get("strength")] if p]
        )
    )

card_end()
spacer(10)

# ==================================================
# 📅 주간 식단표
# ==================================================
card_start("📅 주간 식단표")
muted_text("이벤트·식사·운동 기록을 반영해 주간 계획을 자동 조정합니다.")

btn_col, _ = st.columns([4, 8])
with btn_col:
    if st.button("📅 주간 식단표 열기"):
        st.switch_page("pages/weekly_meal_plan.py")

card_end()

spacer(14)
app_container_end()
bottom_nav(active="routine")
