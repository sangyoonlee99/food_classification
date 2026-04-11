from __future__ import annotations

import sys
from pathlib import Path
from datetime import date
import streamlit as st

# ==================================================
# Path
# ==================================================
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ==================================================
# Imports
# ==================================================
from infra.repositories.user_repository import UserRepository
from infra.db_server import get_db_conn

from services.recommendation_layer import generate_recommendation
from services.recommendation_repository import save_recommendation_history
from services.recommendation_history_loader import load_recommendation_state
from scheduler.replan_scheduler import ReplanScheduler

from ui_streamlit.components import bottom_nav


# ==================================================
# Utils
# ==================================================
def _lob_to_str(x):
    if x is None:
        return None
    if hasattr(x, "read"):
        try:
            return x.read()
        except Exception:
            return str(x)
    return x


# ==================================================
# Event flags (today)
# ==================================================
def load_today_event_flags(*, user_id: bytes) -> dict:
    sql = """
    SELECT event_type, severity, note
    FROM event_log
    WHERE user_id = :user_id
      AND event_date = TRUNC(SYSDATE)
    """
    flags: dict = {}
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute(sql, {"user_id": user_id})
        for event_type, severity, note in cur.fetchall():
            flags[event_type] = {
                "severity": severity,
                "note": _lob_to_str(note),
            }
    return flags


# ==================================================
# Session / Repo
# ==================================================
repo = UserRepository()

user_id: bytes | None = st.session_state.get("user_id")
if not user_id:
    st.error("로그인이 필요합니다.")
    st.stop()

repo.ensure_user_initialized(user_id=user_id)

if not repo.has_profile(user_id):
    st.switch_page("pages/onboarding.py")

# ==================================================
# Header
# ==================================================
st.markdown("## 🧠 오늘의 건강 추천")
st.caption("AI가 오늘의 식단 · 운동 · 루틴을 종합 분석해 추천합니다.")

# ==================================================
# First-time Guide
# ==================================================
if not st.session_state.get("seen_guide_recommendation"):
    with st.container(border=True):
        st.markdown("### 👋 처음 오셨나요?")
        st.markdown(
            """
            이 화면에서는  
            **당신의 목표 · 최근 기록 · 이벤트**를 종합해서  
            오늘 필요한 **식단·운동 조정**을 추천해드려요.
            """
        )
        st.markdown(
            "- **[🚀 오늘의 추천 실행]** 버튼을 눌러보세요.\n"
            "- 추천은 하루에 여러 번 실행해도 괜찮아요."
        )

        if st.button("알겠습니다 👍"):
            st.session_state["seen_guide_recommendation"] = True
            st.rerun()

# ==================================================
# Run button
# ==================================================
user_settings = repo.get_user_settings(user_id=user_id)

if "running" not in st.session_state:
    st.session_state["running"] = False

run = st.button(
    "🚀 오늘의 추천 실행",
    disabled=st.session_state["running"],
)

# ==================================================
# Run recommendation (ENGINE CORE)
# ==================================================
if run:
    st.session_state["running"] = True
    try:
        with st.spinner("AI가 오늘의 추천을 생성 중입니다..."):
            state = load_recommendation_state(user_id=user_id) or {}
            event_flags = load_today_event_flags(user_id=user_id)

            scheduler = ReplanScheduler()
            replan_result = scheduler.run_daily(
                user_id=user_id,
                today=date.today(),
                user_settings=user_settings,
                event_flags=event_flags,
                state=state,
            )

            recommendation = generate_recommendation(
                user_id=user_id,
                replan_result=replan_result,
                state=state,
            )

            save_recommendation_history(
                user_id=user_id,
                recommendation_result=recommendation,
                variant="R-7",
                source="streamlit",
            )

        st.session_state["last_recommendation"] = recommendation
        st.success("✅ 오늘의 추천이 생성되었습니다")
        st.rerun()
    finally:
        st.session_state["running"] = False

# ==================================================
# Render recommendation cards
# ==================================================
recommendation = st.session_state.get("last_recommendation")
if not recommendation:
    recommendation = load_recommendation_state(user_id=user_id) or {}

cards = recommendation.get("recommendations", [])
ctx = (recommendation.get("ctx") or {}).get("recommendation", {}) or {}

badge = ctx.get("badge") or {}
repeat_count = int(ctx.get("repeat_count", 0))
suppressed = bool(ctx.get("suppressed"))
ui_state = ctx.get("state", "normal")

if not cards:
    st.info(
        "아직 오늘의 추천이 없습니다.\n\n"
        "👉 **[🚀 오늘의 추천 실행]** 버튼을 눌러 시작해 보세요."
    )
else:
    # ==================================================
    # 📌 오늘의 기준선 (UI ONLY)
    # ==================================================
    if ui_state in ("enough", "normal", "keep"):
        meal_line = "목표 칼로리 기준 유지"
        exercise_line = "평소 강도 유지 (무리 X)"
    elif ui_state in ("adjust", "light"):
        meal_line = "평소보다 약간 가볍게"
        exercise_line = "유산소 10~15분 추가 권장"
    elif ui_state in ("warning", "over"):
        meal_line = "오늘은 섭취량 조절 필요"
        exercise_line = "가벼운 활동 위주 권장"
    else:
        meal_line = "오늘 컨디션에 맞춰 유연하게"
        exercise_line = "무리하지 않는 선에서 유지"

    with st.expander("📌 오늘의 기준", expanded=False):
        st.markdown(f"- 🍽️ **식사**: {meal_line}")
        st.markdown(f"- 🏃 **운동**: {exercise_line}")

    # 🔖 상단 배지
    if badge:
        tone = badge.get("tone", "neutral")
        label = badge.get("label", "")
        if tone == "warning":
            st.warning(f"🔔 {label}")
        elif tone == "soft":
            st.info(f"💙 {label}")
        else:
            st.caption(f"ℹ️ {label}")

    # 🔁 반복 안내
    if repeat_count > 0:
        st.caption(f"🔁 비슷한 추천 {repeat_count}회째")

    # 카드 분리
    today_cards = [c for c in cards if c.get("type") == "today"]
    tomorrow_cards = [c for c in cards if c.get("type") == "tomorrow"]

    # =========================
    # 📌 오늘의 추천
    # =========================
    if today_cards:
        st.subheader("📌 오늘의 추천")
        for card in today_cards:
            with st.container(border=True):
                st.markdown(f"### {card.get('title','')}")
                for action in card.get("actions", []):
                    icon = action.get("icon", "•")
                    text = action.get("text", "")
                    theme = action.get("theme", "neutral")
                    if theme == "warning":
                        st.error(f"{icon} {text}")
                    elif theme == "positive":
                        st.success(f"{icon} {text}")
                    else:
                        st.markdown(f"{icon} {text}")

    # =========================
    # 🌙 내일 가이드
    # =========================
    if tomorrow_cards:
        st.subheader("🌙 내일을 위한 한 줄 가이드")
        for card in tomorrow_cards:
            with st.container(border=True):
                st.markdown(f"**{card.get('title','')}**")
                for action in card.get("actions", []):
                    st.info(f"➡️ {action.get('text','')}")

    if suppressed:
        st.caption("🛑 비슷한 추천이 반복되어 현재는 유지 모드로 안내 중입니다.")

# ==================================================
# Bottom Nav
# ==================================================
bottom_nav(active="recommendation")
