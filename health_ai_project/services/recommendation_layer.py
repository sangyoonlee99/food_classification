from __future__ import annotations

from typing import Dict, Any, List, Literal, TypedDict


from services.engine_state_policy import evaluate_engine_state
from services.execution_ui_mapper import build_execution_message
from services.explanation.meal_explanation_builder import build_meal_explanation  # ✅ 추가


RecType = Literal["today", "tomorrow"]
RecKind = Literal["diet", "exercise", "routine", "info"]
RecLevel = Literal["micro", "macro", "keep"]


class RecommendationAction(TypedDict, total=False):
    text: str
    kind: RecKind
    level: RecLevel
    icon: str
    theme: str
    cta: str


class RecommendationBlock(TypedDict, total=False):
    type: RecType
    title: str
    actions: List[RecommendationAction]


SIGNATURE_MESSAGES: Dict[str, Dict[str, Any]] = {
    "keep": {
        "text": "오늘은 현재 계획을 유지하면서 몸 상태를 지켜보는 게 좋아 보여요.",
        "kind": "info",
        "level": "keep",
    },
    "diet:adjust": {
        "text": "최근 흐름을 보면 식사량을 살짝만 조정하는 게 좋아 보여요.",
        "kind": "diet",
        "level": "micro",
    },
    "exercise:adjust": {
        "text": "가벼운 활동량 조정으로 충분히 흐름을 잡을 수 있어 보여요.",
        "kind": "exercise",
        "level": "micro",
    },
    "diet:menu": {
        "text": "식단 구성을 한 번 바꿔보는 게 정체를 푸는 데 도움이 될 수 있어요.",
        "kind": "routine",
        "level": "macro",
    },
    "exercise:routine": {
        "text": "운동 루틴에 변화를 주는 게 필요한 시점이에요.",
        "kind": "routine",
        "level": "macro",
    },
}

SIGNATURE_UI_META: Dict[str, Dict[str, str]] = {
    "keep": {"icon": "👀", "theme": "neutral", "cta": "keep"},
    "diet:adjust": {"icon": "🥗", "theme": "positive", "cta": "adjust"},
    "exercise:adjust": {"icon": "🏃", "theme": "positive", "cta": "adjust"},
    "diet:menu": {"icon": "🍱", "theme": "warning", "cta": "change"},
    "exercise:routine": {"icon": "💪", "theme": "warning", "cta": "change"},
}


def _build_signature_from_cards(
    meal_card: Dict[str, Any] | None,
    exercise_card: Dict[str, Any] | None,
) -> tuple[str, RecLevel]:
    parts: List[str] = []
    level: RecLevel = "keep"

    if meal_card:
        actions = meal_card.get("actions", {}) or {}
        if actions.get("change_menu"):
            parts.append("diet:menu")
            level = "macro"
        elif actions.get("adjust_grams"):
            parts.append("diet:adjust")
            level = "micro"

    if exercise_card:
        actions = exercise_card.get("actions", {}) or {}
        if actions.get("change_routine"):
            parts.append("exercise:routine")
            level = "macro"
        elif actions.get("adjust_minutes"):
            if level != "macro":
                level = "micro"
            parts.append("exercise:adjust")

    return ("|".join(parts) if parts else "keep"), level


def _as_info_action(text: str) -> RecommendationAction:
    return {
        "text": text,
        "kind": "info",
        "level": "keep",
        "icon": "📌",
        "theme": "neutral",
        "cta": "info",
    }


def build_recommendations(
    message_result: Dict[str, Any],
    *,
    has_protective_event: bool = False,
    has_illness_event: bool = False,
) -> Dict[str, Any]:

    cards = message_result.get("cards") or []
    state_payload = message_result.get("state", {}) or {}
    ctx_payload = message_result.get("ctx", {}) or {}

    # 🔴 카드가 없으면 keep 카드 생성
    if not cards:
        cards = [{"type": "meal", "actions": {}}]

    meal_card = next((c for c in cards if c.get("type") == "meal"), None)
    exercise_card = next((c for c in cards if c.get("type") == "exercise"), None)
    horizon_card = next((c for c in cards if c.get("type") == "horizon"), None)

    status = "applied" if (meal_card or exercise_card) else "partial"
    signature, rec_level = _build_signature_from_cards(meal_card, exercise_card)

    engine_rec_level, engine_badge = evaluate_engine_state(state_payload)

    if has_protective_event or has_illness_event:
        rec_level = "keep"
        signature = "keep"
    elif engine_rec_level != "keep":
        rec_level = engine_rec_level

    prev_ctx = ctx_payload.get("prev_recommendation", {}) or {}
    prev_sig = prev_ctx.get("signature", "keep")
    prev_count = int(prev_ctx.get("count", 0) or 0)

    repeated = prev_sig == signature
    repeat_count = (prev_count + 1) if repeated else 0
    suppressed = repeat_count >= 2

    recommendations: List[RecommendationBlock] = []

    execution = ctx_payload.get("execution")
    exec_msg = build_execution_message(execution)
    if exec_msg:
        recommendations.append({
            "type": "today",
            "title": exec_msg.get("title", "오늘은 유지"),
            "actions": [_as_info_action(exec_msg.get("body", ""))],
        })

    if suppressed:
        action: RecommendationAction = {
            "text": "비슷한 추천이 반복되고 있어요. 지금은 현재 루틴을 유지해보세요.",
            "kind": "info",
            "level": "keep",
            "icon": "🛑",
            "theme": "neutral",
            "cta": "keep",
        }
    else:
        base = SIGNATURE_MESSAGES.get(signature, SIGNATURE_MESSAGES["keep"])
        ui = SIGNATURE_UI_META.get(signature, SIGNATURE_UI_META["keep"])
        action = {**base, **ui}

    recommendations.append({
        "type": "today",
        "title": "오늘의 추천",
        "actions": [action],
    })

    if horizon_card:
        bullets = horizon_card.get("bullets") or []
        if bullets:
            recommendations.append({
                "type": "tomorrow",
                "title": "내일을 위한 한 줄 가이드",
                "actions": [_as_info_action(str(bullets[0]))],
            })

    badge = engine_badge
    ui_state = "normal"

    if has_illness_event:
        badge = {"code": "RECOVERY", "label": "회복이 우선이에요", "tone": "soft"}
        ui_state = "recovery"
    elif has_protective_event:
        badge = {"code": "UNDERSTOOD", "label": "상황을 고려했어요", "tone": "soft"}
        ui_state = "enough"
    elif suppressed:
        badge = {"code": "ENOUGH", "label": "이미 충분해요", "tone": "neutral"}
        ui_state = "enough"

    # ==================================================
    # ✅ STEP 2 핵심: 식단 설명 ctx에 주입
    # ==================================================
    meal_explanation = build_meal_explanation(
        meal_actions=(meal_card or {}).get("actions") or {}
    )

    return {
        "status": status,
        "recommendations": recommendations,
        "ctx": {
            "recommendation": {
                "signature": signature,
                "level": rec_level,
                "repeat_count": repeat_count,
                "suppressed": suppressed,
                "state": ui_state,
                "badge": badge,
                "explanation": {
                    "meal": meal_explanation,
                },
            },
            "meta": {"variant": "R-5", "source": "engine"},
        },
    }


def generate_recommendation(
    *,
    user_id: bytes,
    replan_result: dict,
    state: dict,
) -> Dict[str, Any]:

    message_result = replan_result.get("messages") or {}

    message_result = {
        **message_result,
        "state": state or {},
        "ctx": {
            "execution": replan_result.get("execution"),
            "prev_recommendation": {
                "signature": (state or {}).get("last_signature", "keep"),
                "count": (state or {}).get("repeat_count", 0),
            },
        },
    }

    try:
        events = load_events_for_period(user_id=user_id)
        has_protective = has_protective_event(events)
        has_illness = any(e.get("type") == "illness" for e in events)
    except Exception:
        has_protective = False
        has_illness = False

    return build_recommendations(
        message_result,
        has_protective_event=has_protective,
        has_illness_event=has_illness,
    )
