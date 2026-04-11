from __future__ import annotations

import sys
from pathlib import Path
import uuid
import streamlit as st
import hashlib
from passlib.hash import bcrypt

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from infra.db_server import get_db_conn

st.title("회원가입")

email = st.text_input("이메일")
password = st.text_input("비밀번호", type="password")

def hash_password(pw: str) -> str:
    return bcrypt.hash(pw)

if st.button("가입하기"):
    if not email or not password:
        st.warning("이메일과 비밀번호를 입력하세요.")
        st.stop()

    try:
        with get_db_conn() as conn:
            cur = conn.cursor()

            # 🔍 이메일 중복 체크
            cur.execute(
                "SELECT COUNT(*) FROM users WHERE email = :email",
                {"email": email},
            )
            if cur.fetchone()[0] > 0:
                st.error("이미 가입된 이메일입니다.")
                st.stop()

            user_id = uuid.uuid4().bytes
            password_hash = hash_password(password)

            # users
            cur.execute(
                """
                INSERT INTO users (user_id, email, password_hash, status, created_at)
                VALUES (:user_id, :email, :pw, 'active', SYSTIMESTAMP)
                """,
                {
                    "user_id": user_id,
                    "email": email,
                    "pw": password_hash,
                },
            )

            # user_profile
            cur.execute(
                "INSERT INTO user_profile (user_id) VALUES (:user_id)",
                {"user_id": user_id},
            )

            # user_settings
            cur.execute(
                "INSERT INTO user_settings (user_id) VALUES (:user_id)",
                {"user_id": user_id},
            )

            conn.commit()

        st.success("회원가입이 완료되었습니다. 로그인해주세요.")
        st.switch_page("pages/login.py")

    except Exception as e:
        st.error(f"회원가입 실패: {e}")
