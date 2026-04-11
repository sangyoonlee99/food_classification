from __future__ import annotations

import sys
from pathlib import Path
import uuid
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from infra.db_server import get_db_conn
from ui_streamlit.utils.auth import hash_password

st.markdown("### 회원가입")

email = st.text_input("이메일")
password = st.text_input("비밀번호", type="password")
nickname = st.text_input("이름 또는 별명 (선택)")  # ✅ 추가

if st.button("가입하기"):
    if not email or not password:
        st.warning("이메일과 비밀번호를 입력하세요.")
        st.stop()

    user_id = uuid.uuid4().bytes  # RAW(16)
    pw_hash = hash_password(password)

    try:
        with get_db_conn() as conn:
            cur = conn.cursor()

            # users
            cur.execute("""
                INSERT INTO users (user_id, email, password_hash, status)
                VALUES (:user_id, :email, :pw, 'active')
            """, {
                "user_id": user_id,
                "email": email.lower(),
                "pw": pw_hash,
            })

            # user_profile (nickname 포함)
            cur.execute("""
                INSERT INTO user_profile (user_id, nickname)
                VALUES (:user_id, :nickname)
            """, {
                "user_id": user_id,
                "nickname": nickname if nickname else None,
            })

            # user_settings
            cur.execute("""
                INSERT INTO user_settings (user_id)
                VALUES (:user_id)
            """, {
                "user_id": user_id,
            })

            conn.commit()   # ✅ 확정 커밋

        st.success("회원가입 완료! 로그인 화면으로 이동합니다.")
        st.switch_page("pages/login.py")

    except Exception as e:
        st.error(f"회원가입 실패: {e}")
