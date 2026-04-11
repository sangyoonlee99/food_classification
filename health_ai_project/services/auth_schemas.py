# services/auth_schemas.py

from pydantic import BaseModel, EmailStr


class UserCreate(BaseModel):
    """
    회원가입 요청 스키마
    """
    email: EmailStr
    password: str


class UserLogin(BaseModel):
    """
    로그인 요청 스키마
    """
    email: EmailStr
    password: str


class AuthResult(BaseModel):
    """
    인증 결과 (응답)
    """
    user_id: str
    access_token: str
