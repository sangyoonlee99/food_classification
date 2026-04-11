from __future__ import annotations
from typing import Optional, Dict, Any

from services.recommendation_parser import parse_signature

# =================================================
# SQL: 오늘 최신 추천 signature 1건
# =================================================
UI_CARD_SQL = """
SELECT signature
FROM (
    SELECT signature
    FROM recommendation_log
    WHERE user_id  = :user_id
      AND rec_date = TRUNC(SYSDATE)
    ORDER BY created_at DESC
)
WHERE ROWNUM = 1
"""

# =================================================
# UI 카드 프리셋 (문구 최소화 / 톤 고정)
# =================================================
UI_CARD_PRESETS = {
    "STUCK": {
        "tone": "neutral",
        "title": "체중 정체가 감지됐어요",
        "body": "최근 변화가 적어요. 오늘은 가볍게 조정해볼게요.",
    },
    "OVER": {
        "tone": "warning",
        "title": "오늘 섭취가 많았어요",
        "body": "조금만 활동량을 늘려보는 건 어때요?",
    },
    "ENOUGH": {
        "tone": "neutral",
        "title": "잘 유지하고 있어요",
        "body": "오늘은 현재 흐름을 유지해도 괜찮아요.",
    },
    "GOOD": {
        "tone": "positive",
        "title": "아주 좋아요!",
        "body": "오늘 목표를 잘 달성했어요 👍",
    },
}

# =================================================
# signature → badge 코드 매핑
# =================================================
def signature_to_badge(signature: str) -> str:
    """
    signature 문자열을 UI badge 코드로 변환
    """
    if not signature:
        return "ENOUGH"

    if "diet:-" in signature:
        return "STUCK"

    if "diet:+" in signature:
        return "OVER"

    if signature == "keep":
        return "ENOUGH"

    return "GOOD"


# =================================================
# Home / Dashboard 용 단일 UI 카드
# =================================================
def get_today_ui_card(
    conn,
    user_id: bytes,
) -> Optional[Dict[str, Any]]:
    """
    오늘의 추천 결과를 UI 카드 1장으로 반환

    Contract:
    {
        title: str,
        body: str,
        tone: str,
        badge: str,
        signature: str,
        actions: list
    }
    """
    cur = conn.cursor()
    cur.execute(UI_CARD_SQL, {"user_id": user_id})
    row = cur.fetchone()

    if not row:
        return None

    signature: str = row[0] or ""
    badge = signature_to_badge(signature)

    preset = UI_CARD_PRESETS.get(badge, UI_CARD_PRESETS["ENOUGH"])
    actions = parse_signature(signature)

    return {
        "title": preset["title"],
        "body": preset["body"],
        "tone": preset["tone"],
        "badge": badge,          # UI 표시용 (문자열)
        "signature": signature,  # 디버그 / 추적용
        "actions": actions,      # 상세 조정 내역
    }
