from __future__ import annotations

import streamlit as st
from typing import Optional
from datetime import date, timedelta
from infra.db_server import get_db_conn


def selectable_item(label: str, key: str):
    return st.button(label, key=key, use_container_width=True)


# ==================================================
# Layout Helpers
# ==================================================
def app_container_start():
    st.markdown('<div class="app-container">', unsafe_allow_html=True)


def app_container_end():
    st.markdown("</div>", unsafe_allow_html=True)


# ==================================================
# Card Components
# ==================================================
def card_start(title: Optional[str] = None, right_text: Optional[str] = None):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if title:
        if right_text:
            st.markdown(
                f"""
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div class="card-title">{title}</div>
                    <div style="color:#888; font-size:14px;">{right_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f'<div class="card-title">{title}</div>', unsafe_allow_html=True)


def card_end():
    st.markdown("</div>", unsafe_allow_html=True)


# ==================================================
# Section / Text
# ==================================================
def section_header(text: str):
    st.markdown(
        f"""
        <div style="margin:18px 0 10px 0; font-weight:700; font-size:15px;">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def muted_text(text: str):
    st.markdown(
        f'<div style="color:#777; font-size:13px;">{text}</div>',
        unsafe_allow_html=True,
    )


# ==================================================
# Buttons
# ==================================================
def primary_button(label: str, key: Optional[str] = None) -> bool:
    return st.button(label, key=key)


def secondary_button(label: str, key: Optional[str] = None) -> bool:
    st.markdown(
        """
        <style>
        .secondary-btn button {
            width: 100%;
            height: 44px;
            border-radius: 22px;
            background-color: #e9e9e9;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    return st.button(label, key=key)


# ==================================================
# Macro / Progress UI
# ==================================================
def macro_bar(carb: float, protein: float, fat: float):
    total = carb + protein + fat
    if total <= 0:
        muted_text("영양 정보가 아직 없습니다.")
        return

    st.markdown(
        f"""
        <div style="margin-top:8px;">
            <div style="display:flex; height:14px; border-radius:7px; overflow:hidden;">
                <div style="flex:{carb/total}; background:#f6c344;"></div>
                <div style="flex:{protein/total}; background:#6fbf73;"></div>
                <div style="flex:{fat/total}; background:#f08a8a;"></div>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:12px; margin-top:6px;">
                <div>탄 {round(carb)}g</div>
                <div>단 {round(protein)}g</div>
                <div>지 {round(fat)}g</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def score_circle(score: int):
    st.markdown(
        f"""
        <div style="
            width:90px;
            height:90px;
            border-radius:50%;
            background:#111;
            color:#fff;
            display:flex;
            align-items:center;
            justify-content:center;
            font-size:26px;
            font-weight:700;
        ">
            {score}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==================================================
# Spacer / Skeleton
# ==================================================
def spacer(h: int = 16):
    st.markdown(f'<div style="height:{h}px;"></div>', unsafe_allow_html=True)


def skeleton(height: int = 16, radius: int = 8):
    st.markdown(
        f"""
        <div style="
            height:{height}px;
            border-radius:{radius}px;
            background: linear-gradient(
                90deg,
                #f0f0f0 25%,
                #e6e6e6 37%,
                #f0f0f0 63%
            );
            background-size: 400% 100%;
            animation: skeleton-loading 1.4s ease infinite;
            margin-bottom: 8px;
        "></div>
        """,
        unsafe_allow_html=True,
    )


# ==================================================
# bottom_nav (✅ 중복 ID 완전 해결)
# ==================================================
def bottom_nav(active: str):
    """
    active:
      home | routine | report | record | settings
      record_meal / record_event / record_exercise / record_weight → record
    """
    st.markdown("---")
    cols = st.columns(5)

    record_actives = {
        "record",
        "record_meal",
        "record_event",
        "record_exercise",
        "record_weight",
    }

    with cols[0]:
        if st.button(
            "홈",
            key="bottom_nav_home",
            use_container_width=True,
            type="primary" if active == "home" else "secondary",
        ):
            st.switch_page("pages/home.py")

    with cols[1]:
        if st.button(
            "루틴",
            key="bottom_nav_routine",
            use_container_width=True,
            type="primary" if active == "routine" else "secondary",
        ):
            st.switch_page("pages/routine.py")

    with cols[2]:
        if st.button(
            "리포트",
            key="bottom_nav_report",
            use_container_width=True,
            type="primary" if active == "report" else "secondary",
        ):
            st.switch_page("pages/report.py")

    with cols[3]:
        if st.button(
            "기록",
            key="bottom_nav_record",
            use_container_width=True,
            type="primary" if active in record_actives else "secondary",
        ):
            st.switch_page("pages/record.py")

    with cols[4]:
        if st.button(
            "설정",
            key="bottom_nav_settings",
            use_container_width=True,
            type="primary" if active == "settings" else "secondary",
        ):
            st.switch_page("pages/settings.py")


# ==================================================
# Exercise Cards (기존 유지)
# ==================================================
def exercise_today_card(user_id: bytes):
    today = date.today()

    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                total_minutes,
                cardio_minutes,
                strength_minutes,
                intensity_level
            FROM daily_exercise_summary
            WHERE user_id = :u
              AND summary_date = :d
            """,
            {"u": user_id, "d": today},
        )
        row = cur.fetchone()

    card_start("🏃 오늘의 운동")
    if not row:
        muted_text("아직 기록된 운동이 없습니다.")
        card_end()
        return

    total, cardio, strength, intensity = row
    st.metric("총 운동 시간", f"{total} 분")
    muted_text(f"유산소 {cardio}분 · 근력 {strength}분 · 강도 {intensity}")
    card_end()


def exercise_weekly_card(user_id: bytes):
    today = date.today()
    start = today - timedelta(days=6)

    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                SUM(total_minutes),
                SUM(cardio_minutes),
                SUM(strength_minutes)
            FROM daily_exercise_summary
            WHERE user_id = :u
              AND summary_date BETWEEN :s AND :e
            """,
            {"u": user_id, "s": start, "e": today},
        )
        row = cur.fetchone()

    card_start("📅 주간 운동 요약")
    if not row or not row[0]:
        muted_text("최근 7일간 기록된 운동이 없습니다.")
        card_end()
        return

    total, cardio, strength = map(lambda x: int(x or 0), row)
    st.markdown(f"- **총 운동 시간**: `{total} 분`")
    st.markdown(f"- **유산소 / 근력**: `{cardio} 분 / {strength} 분`")

    if total >= 300:
        st.success("🔥 아주 훌륭해요! 주간 목표를 충분히 달성했어요.")
    elif total >= 150:
        st.info("👍 잘하고 있어요. 조금만 더 꾸준히 가볼까요?")
    else:
        st.warning("⚠️ 운동량이 부족해요. 가벼운 활동부터 늘려보세요.")

    card_end()
