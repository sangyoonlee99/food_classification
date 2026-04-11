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
from ui_streamlit.utils.auth import login
from infra.repositories.user_repository import UserRepository

repo = UserRepository()

# ==================================================
# UI
# ==================================================
st.markdown("### 로그인")

email = st.text_input("이메일", placeholder="you@example.com")
password = st.text_input("비밀번호", type="password")

col1, col2 = st.columns(2)

with col1:
    if st.button("로그인", use_container_width=True):
        ok = login(email, password)

        if not ok:
            st.session_state.clear()
            st.error("이메일 또는 비밀번호가 올바르지 않습니다.")
            st.stop()

        # =============================================
        # ✅ 로그인 성공 이후 분기 (핵심)
        # =============================================
        user_id = st.session_state.get("user_id")

        if not user_id:
            st.session_state.clear()
            st.error("세션 오류가 발생했습니다. 다시 로그인해주세요.")
            st.stop()

        # 🔥 최초 가입자 → 무조건 온보딩
        if not repo.has_profile(user_id):
            st.switch_page("pages/onboarding.py")
        else:
            st.switch_page("pages/home.py")

with col2:
    if st.button("회원가입", use_container_width=True):
        st.switch_page("pages/register.py")
