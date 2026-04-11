from __future__ import annotations

import sys
from pathlib import Path
from datetime import date
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
    spacer,
)
from infra.repositories.user_repository import UserRepository

# ==================================================
# Auth
# ==================================================
require_login()
user_id = st.session_state["user_id"]
repo = UserRepository()

# ==================================================
# ✅ 이미 프로필 있으면 goal_setup으로
# ==================================================
if repo.has_profile(user_id):
    st.switch_page("pages/goal_setup.py")
    st.stop()

# ==================================================
# UI
# ==================================================
app_container_start()
st.markdown("### 🧩 기본 정보 설정")
st.caption("맞춤 운동·식단 추천을 위해 꼭 필요한 정보만 입력해주세요.")
spacer(12)

# ------------------------------
# 기본 정보
# ------------------------------
card_start("👤 기본 정보")

sex_ui = st.selectbox("성별", ["선택해주세요", "남성", "여성"])

birth_year = st.number_input(
    "출생년도",
    min_value=1930,
    max_value=date.today().year,
    value=1995,
)

height_cm = st.number_input(
    "키 (cm)",
    min_value=120.0,
    max_value=220.0,
    value=170.0,
)

weight_kg = st.number_input(
    "현재 체중 (kg)",
    min_value=30.0,
    max_value=200.0,
    value=70.0,
)

activity_ui = st.selectbox(
    "평소 활동 수준",
    ["선택해주세요", "낮음", "보통", "높음"],
)

card_end()
spacer(10)

# ------------------------------
# 건강 정보
# ------------------------------
card_start("🩺 건강 정보 (선택)")

conditions_ui = st.multiselect(
    "건강 유의사항",
    ["해당없음", "당뇨", "고혈압"],
)

card_end()
spacer(10)

# ------------------------------
# 🔥 선호 운동 (핵심)
# ------------------------------
card_start("🏃 선호 운동 (추천 기준에 사용됩니다)")

CARDIO_OPTIONS = ["걷기", "조깅", "러닝", "러닝머신", "자전거", "수영"]
STRENGTH_OPTIONS = ["홈트레이닝", "헬스", "크로스핏", "근력운동"]

preferred_cardio = st.multiselect(
    "선호 유산소 운동",
    options=CARDIO_OPTIONS,
)

preferred_strength = st.multiselect(
    "선호 근력 운동",
    options=STRENGTH_OPTIONS,
)


st.caption(
    "선택한 운동을 기준으로 운동 가이드와 루틴이 우선 추천됩니다."
)

card_end()
spacer(14)

# ==================================================
# Normalize
# ==================================================
sex = "male" if sex_ui == "남성" else "female" if sex_ui == "여성" else None

activity_level = {
    "낮음": "low",
    "보통": "medium",
    "높음": "high",
}.get(activity_ui)

conditions = [] if "해당없음" in conditions_ui else conditions_ui

# ==================================================
# Save → goal_setup
# ==================================================
if st.button("저장하고 시작하기", use_container_width=True):
    if not sex or not activity_level:
        st.warning("성별과 활동 수준을 선택해주세요.")
        st.stop()

    repo.save_user_profile(
        user_id=user_id,
        sex=sex,
        birth_year=int(birth_year),
        height_cm=float(height_cm),
        weight_kg=float(weight_kg),
        activity_level=activity_level,
        conditions=conditions,
        preferences={},  # 🔥 legacy 필드 (미사용)
        preferred_cardio=preferred_cardio,
        preferred_strength=preferred_strength,
    )

    st.switch_page("pages/goal_setup.py")
    st.stop()

app_container_end()
