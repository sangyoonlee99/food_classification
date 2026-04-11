# services/auth_utils.py

import bcrypt
import jwt
import uuid
from datetime import datetime, timedelta

# ⚠️ STEP C(DB)에서 .env로 분리 예정
SECRET_KEY = "CHANGE_ME_LATER"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 12


def generate_user_id() -> str:
    """
    내부 user_id 생성
    """
    return str(uuid.uuid4())


def hash_password(password: str) -> str:
    """
    비밀번호 해시
    """
    return bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt()
    ).decode("utf-8")


def verify_password(password: str, hashed_password: str) -> bool:
    """
    비밀번호 검증
    """
    return bcrypt.checkpw(
        password.encode("utf-8"),
        hashed_password.encode("utf-8")
    )


def create_access_token(user_id: str) -> str:
    """
    JWT Access Token 생성
    """
    payload = {
        "sub": user_id,
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS),
    }

    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
