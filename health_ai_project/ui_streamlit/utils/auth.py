from __future__ import annotations

import os
import re
import hashlib
from typing import Optional, Dict, Any

import streamlit as st
from infra.db_server import get_db_conn


# ==================================================
# Password hashing (PBKDF2)
# ==================================================
def hash_password(password: str, *, iterations: int = 200_000) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, salt_hex, hash_hex = stored.split("$", 3)
        if algo != "pbkdf2_sha256":
            return False

        dk = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode(),
            bytes.fromhex(salt_hex),
            int(iters),
        )
        return dk.hex() == hash_hex
    except Exception:
        return False


def is_valid_email(email: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email or ""))


# ==================================================
# DB helpers
# ==================================================
def _fetch_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                u.user_id,
                u.email,
                u.password_hash,
                u.status,
                p.nickname
            FROM users u
            LEFT JOIN user_profile p ON p.user_id = u.user_id
            WHERE LOWER(u.email) = :email
            """,
            {"email": email.lower()},
        )
        row = cur.fetchone()

    if not row:
        return None

    return {
        "user_id": row[0],          # RAW(16)
        "email": row[1],
        "password_hash": row[2],
        "status": row[3],
        "nickname": row[4],
    }


def _user_exists(user_id) -> bool:
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM users WHERE user_id = :user_id",
            {"user_id": user_id},
        )
        return cur.fetchone()[0] > 0


# ==================================================
# Auth API
# ==================================================
def login(email: str, password: str) -> bool:
    """
    ✔️ 로그인 성공 시에만 세션 생성
    ✔️ 기존 세션 완전 제거
    ✔️ 온보딩 플래그 초기화
    """
    user = _fetch_user_by_email(email)
    if not user:
        return False

    if user["status"] != "active":
        return False

    if not verify_password(password, user["password_hash"]):
        return False

    # 🔥 세션 완전 초기화 (잔존 방지)
    st.session_state.clear()

    st.session_state["is_logged_in"] = True
    st.session_state["user_id"] = user["user_id"]
    st.session_state["user_email"] = user["email"]
    st.session_state["user_nickname"] = user.get("nickname")

    return True


def logout():
    """
    ✔️ 로그아웃은 항상 login 페이지로
    """
    st.session_state.clear()
    st.switch_page("pages/login.py")


def require_login():
    """
    ✔️ 모든 보호 페이지에서 사용
    ✔️ 세션 + DB 이중 검증
    """
    user_id = st.session_state.get("user_id")

    # 1️⃣ 세션 자체가 없으면 즉시 로그인 페이지
    if not user_id:
        st.switch_page("pages/login.py")

    # 2️⃣ DB에 사용자 없으면 (삭제 / 초기화 등)
    if not _user_exists(user_id):
        st.session_state.clear()
        st.switch_page("pages/login.py")
