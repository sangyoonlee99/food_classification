# services/engine_execution_policy.py
from __future__ import annotations

from typing import Dict, Any
from datetime import date

from services.engine_cooldown_policy import is_in_cooldown


def _classify_signature(signature: str) -> str:
    """
    signature 포맷 혼재 대응:
      - 실행형: "diet:-200,cardio:+10" / "diet:-300" / "keep"
      - UI형: "diet:adjust" / "diet:menu" / "exercise:adjust" / "exercise:routine" / "keep"

    return: "none" | "micro" | "macro"
    """
    if not signature or signature == "keep":
        return "none"

    # macro 키워드(운영에서 강제 분류)
    if signature.endswith("menu") or signature.endswith("routine"):
        return "macro"
    if "menu_change" in signature or "routine_change" in signature:
        return "macro"

    # 실행형(숫자 조정)은 micro로 취급
    # diet:-200, cardio:+10, strength:+10 등
    if "diet:" in signature or "cardio:" in signature or "strength:" in signature:
        return "micro"

    return "none"


def evaluate_execution_policy(
    *,
    state: Dict[str, Any],
    signature: str,
    today: date,
) -> Dict[str, Any]:
    """
    실행 가능 여부 판단 (Single Source of Truth)

    return:
      { allowed: bool, reason: str, cooldown_days: int }
    """

    plateau_days = int(state.get("plateau_days", 0) or 0)

    # 1) cooldown 최우선
    if is_in_cooldown(state=state, signature=signature, today=today):
        return {"allowed": False, "reason": "cooldown_active", "cooldown_days": 0}

    kind = _classify_signature(signature)

    # 2) micro/macro는 plateau 조건
    if kind == "micro":
        if plateau_days < 7:
            return {"allowed": False, "reason": "plateau_too_short", "cooldown_days": 0}
        return {"allowed": True, "reason": "micro_allowed", "cooldown_days": 1}

    if kind == "macro":
        if plateau_days < 14:
            return {"allowed": False, "reason": "macro_not_ready", "cooldown_days": 0}
        return {"allowed": True, "reason": "macro_allowed", "cooldown_days": 3}

    # 3) 기본(keep/없음)
    return {"allowed": True, "reason": "default_allowed", "cooldown_days": 0}
