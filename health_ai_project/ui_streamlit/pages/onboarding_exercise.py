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
from ui_streamlit.components import (
    app_container_start,
    app_container_end,
    card_start,
    card_end,
    spacer,
    primary_button,
)
from infra.db_server import get_db_conn

# ==================================================
# Session Guard
# ==================================================
user_id = st.session_state.get("user_id")
if not user_id:
    st.error("로그인이 필요합니다.")
    st.stop()

# ==================================================
# Constants
# ==================================================
CARDIO_OPTIONS = ["걷기", "조깅", "러닝", "러닝머신", "자전거"]
STRENGTH_OPTIONS = ["홈트레이닝", "헬스", "크로스핏"]

# ==================================================
# Header
# ==================================================
app_container_start()
st.markdown("### 🏃 운동 선호 설정")
st.caption("선호하는 운동을 선택하면, 이후 운동 가이드와 추천에 반영됩니다.")
spacer(12)

# ==================================================
# Load existing preferences (if any)
# ==================================================
saved_cardio = []
saved_strength = []

with get_db_conn() as conn:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT preferred_cardio, preferred_strength
        FROM user_profile
        WHERE user_id = :user_id
        """,
        {"user_id": user_id},
    )
    row = cur.fetchone()

if row:
    if row[0]:
        saved_cardio = [x.strip() for x in row[0].split(",") if x.strip()]
    if row[1]:
        saved_strength = [x.strip() for x in row[1].split(",") if x.strip()]

# ==================================================
# UI
# ==================================================
card_start("유산소 운동 (선호)")
cardio_selected = st.multiselect(
    "유산소 운동을 선택하세요",
    CARDIO_OPTIONS,
    default=saved_cardio,
)
card_end()

spacer(10)

card_start("근력 운동 (선호)")
strength_selected = st.multiselect(
    "근력 운동을 선택하세요",
    STRENGTH_OPTIONS,
    default=saved_strength,
)
card_end()

spacer(16)

# ==================================================
# Save Action
# ==================================================
if primary_button("💾 저장하고 계속"):
    cardio_csv = ",".join(cardio_selected) if cardio_selected else None
    strength_csv = ",".join(strength_selected) if strength_selected else None

    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE user_profile
            SET
                preferred_cardio   = :cardio,
                preferred_strength = :strength
            WHERE user_id = :user_id
            """,
            {
                "cardio": cardio_csv,
                "strength": strength_csv,
                "user_id": user_id,
            },
        )
        conn.commit()

    st.success("✅ 운동 선호가 저장되었습니다.")
    st.switch_page("pages/home.py")

app_container_end()
