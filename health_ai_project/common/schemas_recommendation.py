# common/schemas_recommendation.py
from __future__ import annotations

from typing import Literal, Optional, List
from pydantic import BaseModel


class RecommendationAction(BaseModel):
    """
    사용자에게 제시되는 '단일 행동 단위'
    UI / UX / 로그 / A/B 테스트 기준 단위
    """

    kind: Literal[
        "diet",        # 식단 관련 행동
        "exercise",    # 운동 관련 행동
        "routine",     # 구조 변경 (macro)
        "info",        # 안내 / 유지
    ]

    level: Literal[
        "micro",       # 소폭 조정
        "macro",       # 구조 변경
        "keep",        # 유지
    ]

    message: str                   # 사용자에게 보여줄 문장
    value: Optional[int | str] = None


class RecommendationBlock(BaseModel):
    """
    today / tomorrow 단위 블록
    """

    type: Literal["today", "tomorrow"]
    title: str
    actions: List[RecommendationAction]


class RecommendationResult(BaseModel):
    """
    Recommendation Layer 최종 결과 계약
    """

    status: Literal["applied", "partial", "failed"]
    recommendations: List[RecommendationBlock]
