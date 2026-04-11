from __future__ import annotations

import sys
from pathlib import Path
from datetime import date
import uuid
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
    spacer,
    bottom_nav,
)
from infra.db_server import get_db_conn

# ==================================================
# Utils
# ==================================================
def _ensure_bytes16(v) -> bytes:
    if isinstance(v, (bytes, bytearray)):
        return bytes(v)
    if isinstance(v, str):
        return bytes.fromhex(v.replace("0x", "").strip())
    raise TypeError("invalid user_id")

def _gen_raw16() -> bytes:
    return uuid.uuid4().bytes

def calc_weight_gap(current: float | None, target: float | None):
    if current is None or target is None:
        return None

    diff = round(target - current, 1)

    if abs(diff) < 0.2:
        return ("🎯 목표 체중 도달", "±0.0 kg", "현재 체중을 유지하세요")

    if diff > 0:
        return ("목표 체중까지", f"+{diff} kg", "체중 증가가 필요해요")

    return ("목표 체중까지", f"{diff} kg", "체중 감량이 필요해요")

# ==================================================
# Auth
# ==================================================
require_login()
user_id = st.session_state.get("user_id")
if not user_id:
    st.switch_page("pages/login.py")
    st.stop()

user_id_raw = _ensure_bytes16(user_id)

# ==================================================
# UI START
# ==================================================
app_container_start()
st.markdown("## ⚖️ 체중 기록")
spacer(8)

# ==================================================
# 📅 날짜 선택 (가장 먼저)
# ==================================================
selected_date: date = st.date_input(
    "📅 기록 날짜",
    value=st.session_state.get("selected_weight_date", date.today()),
    help="과거 날짜도 기록 및 수정할 수 있습니다.",
)
st.session_state["selected_weight_date"] = selected_date

spacer(6)

# ==================================================
# Load weight for selected date
# ==================================================
body_log_id = None
loaded_weight = None

with get_db_conn() as conn:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT body_log_id, weight_kg
        FROM body_log
        WHERE user_id = :u
          AND TRUNC(measured_at) = :d
        """,
        {"u": user_id_raw, "d": selected_date},
    )
    row = cur.fetchone()

if row:
    body_log_id, loaded_weight = row
    default_weight = float(loaded_weight)
    mode = "update"

    # ✅ 기록된 체중 명확히 표시
    st.success(
        f"📌 {selected_date}에 기록된 체중: **{default_weight:.1f} kg**"
    )
else:
    default_weight = 70.0
    mode = "insert"
    st.info(
        f"ℹ️ {selected_date}에는 아직 체중 기록이 없습니다."
    )

spacer(8)

# ==================================================
# Load goal
# ==================================================
with get_db_conn() as conn:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT target_weight_kg
        FROM user_goal
        WHERE user_id = :u
          AND is_active = 'Y'
        """,
        {"u": user_id_raw},
    )
    goal_row = cur.fetchone()

target_weight = float(goal_row[0]) if goal_row else None
gap = calc_weight_gap(default_weight, target_weight)

# ==================================================
# 목표 요약
# ==================================================
if gap:
    title, value, desc = gap
    st.markdown(
        f"""
        <div style="border:1px solid #e5e7eb;border-radius:12px;padding:14px;margin-bottom:12px;">
            <b>{title}</b><br>
            <span style="font-size:18px;font-weight:700;">{value}</span><br>
            <span style="font-size:13px;color:#666">{desc}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==================================================
# 체중 입력
# ==================================================
weight = st.number_input(
    "체중 (kg)",
    min_value=30.0,
    max_value=200.0,
    step=0.1,
    value=round(default_weight, 1),
)

spacer(12)

# ==================================================
# Save
# ==================================================
if st.button("💾 저장", use_container_width=True):
    with get_db_conn() as conn:
        cur = conn.cursor()

        if mode == "insert":
            cur.execute(
                """
                INSERT INTO body_log (
                    body_log_id,
                    user_id,
                    measured_at,
                    weight_kg,
                    source
                )
                VALUES (
                    :id,
                    :u,
                    :d,
                    :w,
                    'manual'
                )
                """,
                {
                    "id": _gen_raw16(),
                    "u": user_id_raw,
                    "d": selected_date,
                    "w": weight,
                },
            )
        else:
            cur.execute(
                """
                UPDATE body_log
                SET weight_kg = :w
                WHERE body_log_id = :id
                """,
                {
                    "w": weight,
                    "id": body_log_id,
                },
            )

        conn.commit()

    st.success(f"✅ {selected_date} 체중이 저장되었습니다.")
    st.rerun()

spacer(20)
app_container_end()

# ==================================================
# Bottom Nav
# ==================================================
bottom_nav(active="record")
