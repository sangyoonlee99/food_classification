# ui_streamlit/pages/record.py
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
# Auth
# ==================================================
from ui_streamlit.utils.auth import require_login
require_login()

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
    bottom_nav,
)

# ==================================================
# UI
# ==================================================
app_container_start()

st.markdown("## 📝 기록하기")
st.caption("오늘의 상태를 기록해 주세요.")
spacer(12)

# ==================================================
# 🍽️ 식사 기록
# ==================================================
card_start("🍽️ 식사 기록")
st.markdown("오늘 섭취한 식사를 기록합니다.")
spacer(4)
if primary_button("식사 기록하기"):
    st.switch_page("pages/record_meal.py")
card_end()

spacer(10)

# ==================================================
# 🏃 운동 기록
# ==================================================
card_start("🏃 운동 기록")
st.markdown("오늘 수행한 운동을 기록합니다.")
spacer(4)
if primary_button("운동 기록하기"):
    st.switch_page("pages/record_exercise.py")
card_end()

spacer(10)

# ==================================================
# ⚖️ 체중 기록
# ==================================================
card_start("⚖️ 체중 기록")
st.markdown("오늘의 체중을 기록합니다.")
spacer(4)
if primary_button("체중 기록하기"):
    st.switch_page("pages/record_weight.py")
card_end()

spacer(10)

# ==================================================
# 🧩 이벤트 기록
# ==================================================
card_start("🧩 이벤트 기록")
st.markdown("회식·야근·여행 등 일상 이벤트를 기록합니다.")
spacer(4)
if primary_button("이벤트 기록하기"):
    st.switch_page("pages/record_event.py")
card_end()

spacer(24)

app_container_end()
bottom_nav(active="record")
