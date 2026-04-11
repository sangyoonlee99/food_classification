# services/recommendation_history_loader.py
from __future__ import annotations

from typing import Dict, Any

from services.recommendation_repository import load_last_recommendation


def load_recommendation_state(*, user_id: bytes) -> Dict[str, Any]:
    """
    - DB에서 이전 추천(signature)과 반복 카운트를 읽어서
    - recommendation_layer.generate_recommendation(state=...)에 넣어줄 state 생성
    """
    last = load_last_recommendation(user_id=user_id)

    if not last:
        return {
            "last_signature": "keep",
            "repeat_count": 0,
        }

    return {
        "last_signature": last.get("signature") or "keep",
        "repeat_count": int(last.get("count", 0) or 0),
    }
