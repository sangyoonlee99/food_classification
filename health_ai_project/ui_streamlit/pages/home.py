from __future__ import annotations

import sys
from pathlib import Path
from datetime import date, timedelta, datetime
import streamlit as st

from ui_streamlit.theme import apply_theme
apply_theme()

# ==================================================
# Path
# ==================================================
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ==================================================
# Imports
# ==================================================
from ui_streamlit.utils.auth import require_login, logout
from ui_streamlit.components import (
    app_container_start,
    app_container_end,
    spacer,
    bottom_nav,
)
from infra.db_server import get_db_conn
from services.daily_exercise_guide_service import DailyExerciseGuideService

from services.feedback_service import FeedbackService
from common.schemas import NutritionSummary


# ==================================================
# Utils
# ==================================================
def _ensure_bytes16(v) -> bytes:
    if isinstance(v, (bytes, bytearray)):
        return bytes(v)
    return bytes.fromhex(v.replace("0x", "").strip())

def _to_int(x, default=0):
    try:
        return int(float(x or 0))
    except Exception:
        return default

def calc_weight_gap(current: float | None, target: float | None):
    if current is None or target is None:
        return None
    diff = round(target - current, 1)
    if abs(diff) < 0.2:
        return ("🎯 목표 체중 도달", "±0.0 kg", "현재 체중을 유지하세요")
    if diff > 0:
        return ("목표 체중까지", f"+{diff} kg", "체중 증가가 필요해요")
    return ("목표 체중까지", f"{diff} kg", "체중 감량이 필요해요")

def get_week_range_monday(d: date):
    """
    월요일 시작 ~ 일요일 종료 (캘린더 주)
    """
    start = d - timedelta(days=d.weekday())  # Monday
    end = start + timedelta(days=6)          # Sunday
    return start, end


# ==================================================
# Macro explanation
# ==================================================
def macro_explain(goal_type: str):
    if goal_type in ("muscle_gain", "bulk"):
        return [
            ("탄수화물", "45%", "훈련 에너지 확보"),
            ("단백질", "30%", "근합성·보존"),
            ("지방", "25%", "호르몬 유지"),
        ], "근육 증가 목표에 맞춰 단백질 비중을 높인 구성입니다."
    if goal_type in ("weight_loss", "cut"):
        return [
            ("탄수화물", "40%", "과잉 섭취 방지"),
            ("단백질", "35%", "근손실 방지"),
            ("지방", "25%", "필수 지방"),
        ], "감량 시 근손실을 최소화하기 위한 구성입니다."
    return [
        ("탄수화물", "45%", "에너지 공급"),
        ("단백질", "30%", "근육 유지"),
        ("지방", "25%", "균형 유지"),
    ], "균형 유지를 우선한 구성입니다."

GOAL_KR = {
    "muscle_gain": "근육 증가(벌크)",
    "bulk": "근육 증가(벌크)",
    "weight_gain": "체중 증가",
    "weight_loss": "체중 감량",
    "cut": "감량",
    "maintenance": "유지",
}

# ==================================================
# Auth
# ==================================================
require_login()
user_id = _ensure_bytes16(st.session_state.get("user_id"))
today = date.today()

# ==================================================
# Header
# ==================================================
app_container_start()

h1, h2 = st.columns([5, 1])
with h1:
    st.markdown(f"### 👋 {st.session_state.get('user_nickname') or st.session_state.get('user_email')} 님")
with h2:
    if st.button("로그아웃", key="logout_home"):
        logout()
        st.stop()

spacer(8)

# ==================================================
# 🎯 오늘 목표
# ==================================================
with get_db_conn() as conn:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT goal_type, target_date, kcal_target, target_weight_kg
        FROM user_goal
        WHERE user_id=:u AND is_active='Y'
        """,
        {"u": user_id},
    )
    goal = cur.fetchone()

with get_db_conn() as conn:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT weight_kg
        FROM (
            SELECT weight_kg
            FROM body_log
            WHERE user_id=:u
            ORDER BY measured_at DESC
        )
        WHERE ROWNUM = 1
        """,
        {"u": user_id},
    )
    row = cur.fetchone()
    current_weight = float(row[0]) if row else None

if goal:
    goal_type, target_date, kcal_target, target_weight_kg = goal
    d_day = (target_date.date() - today).days if isinstance(target_date, datetime) else "-"
    macro_rows, macro_msg = macro_explain(goal_type)
    has_weight = current_weight is not None and target_weight_kg is not None
    gap = calc_weight_gap(current_weight, target_weight_kg) if has_weight else None

    if has_weight:
        weight_html = f"""
        <b>{gap[0]}</b><br>
        <span style="font-size:16px;font-weight:700;">
            {gap[1]}
        </span><br>
        <span style="font-size:12px;color:#666">
            {gap[2]}
        </span>
        """
    else:
        weight_html = """
        <b>목표 체중</b><br>
        <span style="font-size:13px;color:#999">
            체중 기록을 입력해 주세요
        </span>
        """

    st.markdown(
        f"""
        <div style="display:flex;gap:20px;">
          <div>
            <b>목표 유형</b><br>{GOAL_KR.get(goal_type, goal_type)}
          </div>

          <div style="border-left:1px solid #ddd;padding-left:14px;">
            <b>하루 섭취 열량</b><br>{_to_int(kcal_target):,} kcal
          </div>

          <div style="border-left:1px solid #ddd;padding-left:14px;">
            <b>남은 기간</b><br>D-{d_day}
          </div>

          <div style="border-left:1px solid #ddd;padding-left:14px;">
            {weight_html}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    spacer(8)

    spacer(8)

    # 🔹 매크로
    cols = st.columns(3)
    for i, (k, v, d) in enumerate(macro_rows):
        cols[i].markdown(
            f"""
                <div style="border-right:1px solid #eee;padding-right:10px;">
                  <b>{k} {v}</b><br>
                  <span style="font-size:12px;color:#666">{d}</span>
                </div>
                """,
            unsafe_allow_html=True,
        )

    st.caption(f"💡 {macro_msg}")

spacer(8)


# ==================================================
# 🍽️ 오늘 식사 (보정 반영 – home 전용)
# ==================================================

# 1️⃣ 오늘 섭취 kcal
with get_db_conn() as conn:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT NVL(SUM(kcal),0)
        FROM meal_record
        WHERE user_id=:u AND TRUNC(eaten_at)=TRUNC(SYSDATE)
        """,
        {"u": user_id},
    )
    eaten = _to_int(cur.fetchone()[0])

# 2️⃣ 오늘 "보정된 목표 kcal" (DB 테이블 없이 계산)
# 기준: user_goal.kcal_target + 전날 섭취 초과분 (상한 300)

# 전날 섭취 kcal
with get_db_conn() as conn:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT NVL(SUM(kcal),0)
        FROM meal_record
        WHERE user_id = :u
          AND TRUNC(eaten_at) = TRUNC(SYSDATE) - 1
        """,
        {"u": user_id},
    )
    eaten_yesterday = _to_int(cur.fetchone()[0])

base_target_kcal = _to_int(kcal_target)

# 전날 초과분 계산
excess_yesterday = max(eaten_yesterday - base_target_kcal, 0)

# 🔥 보정 상한 300 kcal
recover_kcal = min(excess_yesterday, 300)

effective_kcal_target = max(
    base_target_kcal - recover_kcal,
    1200,  # 하한
)

# 3️⃣ 초과 / 남음 계산
diff_kcal = eaten - effective_kcal_target

if diff_kcal > 0:
    status_label = "초과"
    status_value = diff_kcal
elif diff_kcal < 0:
    status_label = "남음"
    status_value = abs(diff_kcal)
else:
    status_label = "달성"
    status_value = 0

with st.container(border=True):
    st.markdown("### 🍽️ 오늘 식사")
    st.markdown(
        f"""
        <div style="display:flex;gap:20px;">
          <div><b>목표</b><br>{effective_kcal_target:,} kcal</div>
          <div style="border-left:1px solid #ddd;padding-left:14px;">
            <b>섭취</b><br>{eaten:,} kcal
          </div>
          <div style="border-left:1px solid #ddd;padding-left:14px;">
            <b>{status_label}</b><br>{status_value:,} kcal
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


spacer(8)

# ==================================================
# 🏃 오늘 운동
# ==================================================
# 운동 가이드 로드
guide = DailyExerciseGuideService().build_today_guide(user_id=user_id)

# 오늘 운동 합계 (exercise_record 기준)
with get_db_conn() as conn:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT NVL(SUM(minutes),0)
        FROM exercise_record
        WHERE user_id = :u
          AND TRUNC(performed_at) = TRUNC(SYSDATE)
        """,
        {"u": user_id},
    )
    today_ex_minutes = int(cur.fetchone()[0] or 0)

target_minutes = guide["target_minutes"]

with st.container(border=True):
    st.markdown("### 🏃 오늘 운동")
    st.markdown(f"목표 {target_minutes}분 | 현재 {today_ex_minutes}분")
    st.caption(guide.get("message", ""))


spacer(8)

# ==================================================
# 📊 이번 주 요약 (월~일 기준)
# ==================================================
week_start, week_end = get_week_range_monday(today)

with get_db_conn() as conn:
    cur = conn.cursor()

    # ✅ 주간 평균 섭취 kcal (월~일)
    cur.execute(
        """
        SELECT AVG(total_kcal) FROM (
            SELECT TRUNC(eaten_at) AS d, SUM(kcal) AS total_kcal
            FROM meal_record
            WHERE user_id = :u
              AND TRUNC(eaten_at) BETWEEN :ws AND :we
            GROUP BY TRUNC(eaten_at)
        )
        """,
        {
            "u": user_id,
            "ws": week_start,
            "we": week_end,
        },
    )
    avg_kcal = _to_int(cur.fetchone()[0])

    # ✅ 주간 총 운동 시간 (월~일)
    cur.execute(
        """
        SELECT NVL(SUM(minutes),0)
        FROM exercise_record
        WHERE user_id = :u
          AND TRUNC(performed_at) BETWEEN :ws AND :we
        """,
        {
            "u": user_id,
            "ws": week_start,
            "we": week_end,
        },
    )
    total_ex = _to_int(cur.fetchone()[0])

with st.container(border=True):
    st.markdown("### 📊 이번 주 요약")
    st.caption(f"{week_start} ~ {week_end}")

    st.markdown(
        f"""
        <div style="display:flex;gap:20px;">
          <div>
            <b>평균 섭취</b><br>
            {avg_kcal:,} kcal / 일
          </div>

          <div style="border-left:1px solid #ddd;padding-left:14px;">
            <b>평균 운동</b><br>
            {round(total_ex/7,1)} 분 / 일
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(f"총 운동 시간 {total_ex}분")

spacer(8)

# ==================================================
# 📈 이번 주 AI 리포트 요약
# ==================================================
week_start, week_end = get_week_range_monday(today)

with get_db_conn() as conn:
    cur = conn.cursor()

    # 주간 식사 요약
    cur.execute(
        """
        SELECT
            NVL(SUM(kcal),0),
            NVL(SUM(protein_g),0),
            NVL(SUM(fat_g),0),
            NVL(SUM(carb_g),0),
            COUNT(DISTINCT TRUNC(eaten_at))
        FROM meal_record
        WHERE user_id = :u
          AND TRUNC(eaten_at) BETWEEN :ws AND :we
        """,
        {"u": user_id, "ws": week_start, "we": week_end},
    )
    kcal, protein_g, fat_g, carbs_g, meal_days = cur.fetchone()

    # 주간 운동
    cur.execute(
        """
        SELECT NVL(SUM(minutes),0)
        FROM exercise_record
        WHERE user_id = :u
          AND TRUNC(performed_at) BETWEEN :ws AND :we
        """,
        {"u": user_id, "ws": week_start, "we": week_end},
    )
    ex_total_min = _to_int(cur.fetchone()[0])

    # 체중 변화
    cur.execute(
        """
        SELECT weight_kg
        FROM (
            SELECT weight_kg
            FROM body_log
            WHERE user_id=:u
              AND TRUNC(measured_at) BETWEEN :ws AND :we
            ORDER BY measured_at
        )
        """,
        {"u": user_id, "ws": week_start, "we": week_end},
    )
    weights = [float(r[0]) for r in cur.fetchall()]

# ---- NutritionSummary 생성
nutrition = NutritionSummary(
    total={
        "kcal": kcal,
        "protein_g": protein_g,
        "fat_g": fat_g,
        "carbs_g": carbs_g,
    }
)

# ---- 종합 점수 계산 (리포트와 동일 기준)
avg_kcal = kcal / meal_days if meal_days else 0

if avg_kcal <= 0:
    diet_score = 50
elif 1800 <= avg_kcal <= 2200:
    diet_score = 80
elif avg_kcal < 1800:
    diet_score = 65
else:
    diet_score = 55

if ex_total_min <= 0:
    exercise_score = 40
elif ex_total_min < 150:
    exercise_score = 65
else:
    exercise_score = 80

weight_score = 60
weight_diff = None
if len(weights) >= 2:
    weight_diff = weights[-1] - weights[0]
    if abs(weight_diff) < 0.2:
        weight_score = 70
    elif weight_diff < 0:
        weight_score = 80
    else:
        weight_score = 55

score = int(
    diet_score * 0.5 +
    exercise_score * 0.3 +
    weight_score * 0.2
)

feedback_service = FeedbackService()
ai_messages = feedback_service.weekly_feedback(
    score=score,
    nutrition=nutrition,
    exercise_total_min=ex_total_min,
    weight_diff=weight_diff,
    has_record=meal_days > 0,
    max_messages=2,  # 홈은 2줄만
)

with st.container(border=True):
    st.markdown("### 🧑‍🏫 오늘의 Ara 코치 조언")
    for msg in ai_messages:
        st.markdown(f"- 👨‍⚕️ {msg}")

spacer(8)
app_container_end()
bottom_nav(active="home")
