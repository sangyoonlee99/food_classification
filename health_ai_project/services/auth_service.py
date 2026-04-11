# services/auth_service.py
print("🔥🔥🔥 services/auth_service.py LOADED 🔥🔥🔥")

from datetime import datetime
from typing import Optional

from infra.db_server import get_db_conn
from services.auth_utils import (
    generate_user_id,
    hash_password,
    verify_password,
    create_access_token,
)
from services.auth_schemas import UserCreate, UserLogin, AuthResult


class AuthService:
    """
    Auth Service (DB 기반)
    - 회원가입
    - 로그인
    - 로그아웃 (JWT 무상태)
    """

    def register(self, data: UserCreate) -> AuthResult:
        conn = get_db_conn()
        cur = conn.cursor()

        # 이메일 중복 체크
        cur.execute(
            """
            SELECT 1
            FROM users
            WHERE email = :email
            """,
            {"email": data.email},
        )
        if cur.fetchone():
            raise ValueError("이미 존재하는 사용자입니다.")

        user_id = generate_user_id()          # str (uuid)
        user_id_raw = bytes.fromhex(user_id.replace("-", ""))  # RAW(16)

        cur.execute(
            """
            INSERT INTO users (
                user_id,
                email,
                password_hash,
                status,
                created_at
            ) VALUES (
                :user_id,
                :email,
                :password_hash,
                'active',
                CURRENT_TIMESTAMP
            )
            """,
            {
                "user_id": user_id_raw,
                "email": data.email,
                "password_hash": hash_password(data.password),
            },
        )

        conn.commit()

        access_token = create_access_token(user_id)

        return AuthResult(
            user_id=user_id_raw,
            access_token=access_token,
        )

    def login(self, data: UserLogin) -> AuthResult:
        conn = get_db_conn()
        cur = conn.cursor()

        cur.execute(
            """
            SELECT user_id, password_hash
            FROM users
            WHERE email = :email
              AND status = 'active'
            """,
            {"email": data.email},
        )

        row = cur.fetchone()
        if not row:
            raise ValueError("사용자를 찾을 수 없습니다.")

        user_id_raw, password_hash = row

        if not verify_password(data.password, password_hash):
            raise ValueError("비밀번호가 올바르지 않습니다.")

        # JWT에는 string user_id 사용
        user_id_str = user_id_raw.hex()

        access_token = create_access_token(user_id_str)

        # last_login_at 업데이트
        cur.execute(
            """
            UPDATE users
            SET last_login_at = CURRENT_TIMESTAMP
            WHERE user_id = :user_id
            """,
            {"user_id": user_id_raw},
        )
        conn.commit()

        return AuthResult(
            user_id=user_id_raw,
            access_token=access_token,
        )

    def logout(self, user_id: str) -> bool:
        # JWT 무상태 → 서버 처리 없음
        return True
