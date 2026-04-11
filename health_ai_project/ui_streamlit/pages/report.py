from __future__ import annotations

import sys
from pathlib import Path
from datetime import date, timedelta
import calendar

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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
from ui_streamlit.utils.auth import require_login
from ui_streamlit.components import (
    app_container_start,
    app_container_end,
    card_start,
    card_end,
    muted_text,
    spacer,
    bottom_nav,
)
from infra.db_server import get_db_conn
from infra.repositories.user_repository import UserRepository

from services.feedback_service import FeedbackService
from common.schemas import NutritionSummary

# ==================================================
# Matplotlib Korean Font (auto-detect)
# ==================================================
def _set_korean_font():
    try:
        from matplotlib import font_manager

        installed = {f.name for f in font_manager.fontManager.ttflist}
        candidates = [
            "Malgun Gothic",
            "NanumGothic",
            "AppleGothic",
            "Noto Sans CJK KR",
            "Noto Sans KR",
            "Pretendard",
        ]
        chosen = None
        for c in candidates:
            if c in installed:
                chosen = c
                break
        if chosen:
            plt.rcParams["font.family"] = chosen
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

_set_korean_font()

# ==================================================
# Utils
# ==================================================
def _ensure_bytes16(v) -> bytes:
    if isinstance(v, (bytes, bytearray)):
        b = bytes(v)
        if len(b) != 16:
            raise ValueError(f"RAW(16) must be 16 bytes. got len={len(b)}")
        return b
    if isinstance(v, str):
        s = v.strip()
        if s.lower().startswith("0x"):
            s = s[2:]
        b = bytes.fromhex(s)
        if len(b) != 16:
            raise ValueError(f"RAW(16) hex must be 32 chars. got len={len(b)}")
        return b
    raise TypeError(f"cannot coerce to RAW(16): {type(v)}")

def _query_df(sql: str, params: dict, columns: list[str]) -> pd.DataFrame:
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall() or []
    return pd.DataFrame(rows, columns=columns)

def _week_range(d: date):
    start = d - timedelta(days=d.weekday())
    end = start + timedelta(days=6)
    return start, end

def _month_range(d: date):
    start = date(d.year, d.month, 1)
    last_day = calendar.monthrange(d.year, d.month)[1]
    end = date(d.year, d.month, last_day)
    return start, end

def _load_goal(user_id_raw: bytes) -> dict:
    kcal_target = 2000
    macro_target = {}
    goal_type = "maintenance"
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT kcal_target, macro_target, goal_type
            FROM user_goal
            WHERE user_id = :u
              AND is_active = 'Y'
            """,
            {"u": user_id_raw},
        )
        row = cur.fetchone()
    if row:
        kcal_target = int(row[0] or 2000)
        goal_type = row[2] or "maintenance"
    return {"kcal_target": kcal_target, "macro_target": macro_target, "goal_type": goal_type}

def _fetch_meal_daily(user_id_raw: bytes, start: date, end: date) -> pd.DataFrame:
    sql = """
        SELECT
            TRUNC(eaten_at) AS day,
            NVL(SUM(kcal), 0) AS total_kcal,
            NVL(SUM(carb_g), 0) AS carb_g,
            NVL(SUM(protein_g), 0) AS protein_g,
            NVL(SUM(fat_g), 0) AS fat_g
        FROM meal_record
        WHERE user_id = :u
          AND TRUNC(eaten_at) BETWEEN TRUNC(:s) AND TRUNC(:e)
        GROUP BY TRUNC(eaten_at)
        ORDER BY TRUNC(eaten_at)
    """
    df = _query_df(sql, {"u": user_id_raw, "s": start, "e": end}, ["day", "total_kcal", "carb_g", "protein_g", "fat_g"])
    if df.empty:
        return df
    df["day"] = pd.to_datetime(df["day"]).dt.date
    for c in ["total_kcal", "carb_g", "protein_g", "fat_g"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df

def _fetch_exercise_daily_safe(user_id_raw: bytes, start: date, end: date) -> pd.DataFrame:
    candidates = [
        (
            """
            SELECT
                TRUNC(exercised_at) AS day,
                NVL(SUM(duration_min), 0) AS total_min,
                NVL(SUM(CASE WHEN LOWER(exercise_type) IN ('cardio','유산소') THEN duration_min ELSE 0 END), 0) AS cardio_min,
                NVL(SUM(CASE WHEN LOWER(exercise_type) IN ('strength','근력') THEN duration_min ELSE 0 END), 0) AS strength_min
            FROM exercise_log
            WHERE user_id = :u
              AND TRUNC(exercised_at) BETWEEN TRUNC(:s) AND TRUNC(:e)
            GROUP BY TRUNC(exercised_at)
            ORDER BY TRUNC(exercised_at)
            """,
            ["day", "total_min", "cardio_min", "strength_min"],
        ),
        (
            """
            SELECT
                TRUNC(performed_at) AS day,
                NVL(SUM(minutes), 0) AS total_min,
                NVL(SUM(CASE WHEN LOWER(exercise_type) IN ('cardio','유산소') THEN minutes ELSE 0 END), 0) AS cardio_min,
                NVL(SUM(CASE WHEN LOWER(exercise_type) IN ('strength','근력') THEN minutes ELSE 0 END), 0) AS strength_min
            FROM exercise_log
            WHERE user_id = :u
              AND TRUNC(performed_at) BETWEEN TRUNC(:s) AND TRUNC(:e)
            GROUP BY TRUNC(performed_at)
            ORDER BY TRUNC(performed_at)
            """,
            ["day", "total_min", "cardio_min", "strength_min"],
        ),
        (
            """
            SELECT
                TRUNC(performed_at) AS day,
                NVL(SUM(minutes), 0) AS total_min,
                NVL(SUM(CASE WHEN LOWER(exercise_type) IN ('cardio','유산소') THEN minutes ELSE 0 END), 0) AS cardio_min,
                NVL(SUM(CASE WHEN LOWER(exercise_type) IN ('strength','근력') THEN minutes ELSE 0 END), 0) AS strength_min
            FROM exercise_record
            WHERE user_id = :u
              AND TRUNC(performed_at) BETWEEN TRUNC(:s) AND TRUNC(:e)
            GROUP BY TRUNC(performed_at)
            ORDER BY TRUNC(performed_at)
            """,
            ["day", "total_min", "cardio_min", "strength_min"],
        ),
    ]

    for sql, cols in candidates:
        try:
            df = _query_df(sql, {"u": user_id_raw, "s": start, "e": end}, cols)
            if df.empty:
                continue  # ✅ 여기 핵심
            df["day"] = pd.to_datetime(df["day"]).dt.date
            for c in ["total_min", "cardio_min", "strength_min"]:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            return df
        except Exception:
            continue

    return pd.DataFrame(columns=["day", "total_min", "cardio_min", "strength_min"])


def _badge_kcal(diff: float) -> str:
    if diff > 200:
        return f"🔶 {diff:+.0f} kcal"
    if diff < -200:
        return f"🔷 {diff:+.0f} kcal"
    return f"✅ {diff:+.0f} kcal"

# def _ai_summary(weekly_diff_total: float, avg_kcal: float, kcal_target: float, ex_total_min: float) -> tuple[str, list[str]]:
#     if avg_kcal <= 0:
#         return "이번 기간은 섭취 기록이 부족해 상태 판단이 제한적입니다.", [
#             "하루 1회만이라도 식사 기록을 남겨 주세요."
#         ]
#
#     if weekly_diff_total > 200:
#         state = "이번 주는 섭취량이 목표보다 높은 편입니다."
#         tip1 = "점심/저녁 탄수(밥·면·빵) 비중을 소폭 줄이면 초과 kcal를 빠르게 해소할 수 있어요."
#     elif weekly_diff_total < -200:
#         state = "이번 주는 섭취량이 목표보다 낮은 편입니다."
#         tip1 = "단백질(닭가슴살·두부·계란) 1회 추가로 전체 밸런스를 맞춰보세요."
#     else:
#         state = "이번 주는 섭취 패턴이 목표 범위에서 비교적 안정적입니다."
#         tip1 = "현재 패턴을 유지하면서 단백질을 꾸준히 챙기면 점수 안정에 도움이 됩니다."
#
#     if ex_total_min <= 0:
#         tip2 = "현재 운동 기록이 없습니다. 주 2~3회, 20~30분만 추가해도 다음 주 점수와 체감이 달라질 수 있어요."
#     elif ex_total_min < 150:
#         tip2 = "운동량이 약간 부족해요. 유산소 1회만 늘리면 균형이 좋아집니다."
#     else:
#         tip2 = "운동량은 충분한 편입니다. 중강도 비중을 소폭 높이면 효과가 더 좋아질 수 있어요."
#
#     return state, [tip1, tip2]
#
# def _monthly_ai_2lines(df_meal: pd.DataFrame) -> tuple[str, str]:
#     if df_meal is None or df_meal.empty or df_meal.shape[0] <= 0:
#         return ("이번 달은 기록이 부족해 패턴 판단이 어렵습니다.", "최소 3~4일만 기록해도 월간 패턴이 훨씬 선명해져요.")
#
#     std = float(df_meal["total_kcal"].std()) if df_meal.shape[0] >= 2 else 0.0
#     if std > 400:
#         return ("이번 달은 섭취 변동이 큰 편입니다.", "특정 날짜에 섭취 편차가 커서, 주간 단위로 한 번씩만 정리해도 관리가 쉬워집니다.")
#     return ("이번 달은 섭취 패턴이 비교적 일정했습니다.", "이 흐름을 유지하면서 단백질을 꾸준히 챙기면 체감 성과가 좋아질 가능성이 큽니다.")
#
# # ==================================================
# ✅ Weight (body_log)
# ==================================================
def _fetch_weight_daily(user_id_raw: bytes, start: date, end: date) -> pd.DataFrame:
    sql = """
        SELECT
            day,
            weight_kg
        FROM (
            SELECT
                TRUNC(measured_at) AS day,
                weight_kg,
                ROW_NUMBER() OVER (
                    PARTITION BY TRUNC(measured_at)
                    ORDER BY NVL(created_at, CAST(measured_at AS TIMESTAMP)) DESC
                ) rn
            FROM body_log
            WHERE user_id = :u
              AND TRUNC(measured_at) BETWEEN TRUNC(:s) AND TRUNC(:e)
        )
        WHERE rn = 1
        ORDER BY day
    """
    df = _query_df(sql, {"u": user_id_raw, "s": start, "e": end}, ["day", "weight_kg"])
    if df.empty:
        return df
    df["day"] = pd.to_datetime(df["day"]).dt.date
    df["weight_kg"] = pd.to_numeric(df["weight_kg"], errors="coerce")
    return df

def _weight_summary_text(df: pd.DataFrame) -> str:
    if df is None or df.shape[0] < 2:
        return "체중 기록이 충분하지 않습니다."

    start_w = float(df.iloc[0]["weight_kg"])
    end_w = float(df.iloc[-1]["weight_kg"])
    diff = end_w - start_w

    if abs(diff) < 0.2:
        return "최근 체중이 정체 상태입니다."
    if diff < 0:
        return f"이번 기간 {abs(diff):.1f}kg 감소했습니다."
    return f"이번 기간 {diff:.1f}kg 증가했습니다."

# ==================================================
# Plots
# ==================================================
def _plot_weight_line(df: pd.DataFrame, title: str):
    fig = plt.figure(figsize=(5.2, 2.4))
    if df is None or df.empty:
        plt.title(title)
        plt.text(0.5, 0.5, "데이터 없음", ha="center", va="center")
        plt.axis("off")
        return fig

    x = [d.strftime("%m-%d") for d in df["day"]]
    y = df["weight_kg"].tolist()

    plt.plot(x, y, marker="o")
    plt.ylabel("kg")
    plt.title(title)
    plt.xticks(rotation=35, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    return fig

def _plot_kcal_bar(df_daily: pd.DataFrame, title: str):
    fig = plt.figure(figsize=(5.6, 2.6))
    if df_daily is None or df_daily.empty:
        plt.title(title)
        plt.text(0.5, 0.5, "데이터 없음", ha="center", va="center")
        plt.axis("off")
        return fig

    x = [d.strftime("%m-%d") for d in df_daily["day"]]
    y = df_daily["total_kcal"].tolist()

    plt.bar(x, y)
    plt.ylabel("kcal")
    plt.title(title)
    plt.xticks(rotation=35, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    return fig

def _plot_macro_pie(carb_g: float, protein_g: float, fat_g: float, title: str):
    fig = plt.figure(figsize=(3.8, 3.8))
    total = float(carb_g + protein_g + fat_g)
    if total <= 0:
        plt.title(title)
        plt.text(0.5, 0.5, "데이터 없음", ha="center", va="center")
        plt.axis("off")
        return fig

    plt.pie(
        [carb_g, protein_g, fat_g],
        labels=["탄", "단", "지"],
        autopct="%1.0f%%",
        startangle=90,
        textprops={"fontsize": 9},
    )
    plt.title(title)
    plt.tight_layout()
    return fig

def _plot_exercise_bar(df_ex: pd.DataFrame, title: str):
    fig = plt.figure(figsize=(5.6, 2.4))
    if df_ex is None or df_ex.empty:
        plt.title(title)
        plt.text(0.5, 0.5, "데이터 없음", ha="center", va="center")
        plt.axis("off")
        return fig

    x = [d.strftime("%m-%d") for d in df_ex["day"]]
    y = df_ex["total_min"].tolist()

    plt.bar(x, y)
    plt.ylabel("분")
    plt.title(title)
    plt.xticks(rotation=35, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    return fig

def _count_kcal_days(df_meal: pd.DataFrame, kcal_target: float) -> tuple[int, int, int]:
    over_days = under_days = ok_days = 0
    if df_meal is None or df_meal.empty:
        return 0, 0, 0
    for v in df_meal["total_kcal"].tolist():
        d = float(v) - float(kcal_target)
        if d > 200:
            over_days += 1
        elif d < -200:
            under_days += 1
        else:
            ok_days += 1
    return over_days, under_days, ok_days

# ==================================================
# Auth
# ==================================================
require_login()
user_id = st.session_state.get("user_id")
if not user_id:
    st.switch_page("pages/login.py")
    st.stop()

user_id_raw = _ensure_bytes16(user_id)
repo = UserRepository()

# ==================================================
# UI
# ==================================================
app_container_start()
st.markdown("## 📊 리포트")
muted_text("주간/월간 기준으로 식사·운동 패턴을 한 눈에 요약합니다.")
spacer(6)

today = date.today()
week_start, week_end = _week_range(today)
month_start, month_end = _month_range(today)

goal = _load_goal(user_id_raw)
kcal_target = float(goal["kcal_target"] or 2000)

tab_week, tab_month = st.tabs(["📆 주간", "🗓️ 월간"])

# ==================================================
# WEEKLY
# ==================================================
with tab_week:
    selected_week_date = st.date_input(
        "📅 기준 날짜 선택",
        value=today,
        key="report_week_date"
    )

    week_start, week_end = _week_range(selected_week_date)
    st.caption(f"{week_start} ~ {week_end}")

    df_meal = _fetch_meal_daily(user_id_raw, week_start, week_end)
    df_ex = _fetch_exercise_daily_safe(user_id_raw, week_start, week_end)
    df_weight = _fetch_weight_daily(user_id_raw, week_start, week_end)

    total_kcal = float(df_meal["total_kcal"].sum()) if not df_meal.empty else 0.0
    days_count = int(df_meal.shape[0]) if not df_meal.empty else 0
    avg_kcal = (total_kcal / days_count) if days_count else 0.0

    weekly_target_total = float(kcal_target) * 7.0
    diff_total = total_kcal - weekly_target_total

    over_days, under_days, ok_days = _count_kcal_days(df_meal, kcal_target)

    ex_total_min = float(df_ex["total_min"].sum()) if not df_ex.empty else 0.0
    ex_days = int(df_ex.shape[0]) if not df_ex.empty else 0
    avg_ex_min = ex_total_min / 7

    carb_g = float(df_meal["carb_g"].sum()) if not df_meal.empty else 0.0
    protein_g = float(df_meal["protein_g"].sum()) if not df_meal.empty else 0.0
    fat_g = float(df_meal["fat_g"].sum()) if not df_meal.empty else 0.0


    sub_r, sub_g = st.tabs(["📝 리포트", "📊 그래프"])

    with sub_r:
        card_start("주간 요약", right_text="핵심")
        st.markdown(f"🔥 **총 섭취 kcal**: `{int(total_kcal)} kcal`")
        st.markdown(f"🎯 **목표 대비(7일)**: `{_badge_kcal(diff_total)}` (목표 `{int(weekly_target_total)} kcal`)")
        st.markdown(f"🏃 **총 운동 시간**: `{int(ex_total_min)} 분`")
        st.markdown(f"📌 **기록 일수**: `{days_count} 일` · **일 평균 섭취**: `{int(avg_kcal)} kcal`")
        card_end()
        spacer(6)

        card_start("① 체중 요약", right_text="주간")
        st.markdown(f"- {_weight_summary_text(df_weight)}")
        card_end()
        spacer(6)

        card_start("② 식사 요약", right_text="7일")
        st.markdown("**요약 지표**")
        st.markdown(f"- 평균 섭취: `{int(avg_kcal)} kcal` / 목표 `{int(kcal_target)} kcal`")
        st.markdown(f"- 과식(>+200): `{over_days}일` · 부족(<-200): `{under_days}일` · 적정(±200): `{ok_days}일`")
        st.markdown("**탄/단/지 총합(7일)**")
        st.markdown(f"- 🍚 탄: `{int(carb_g)} g`")
        st.markdown(f"- 🥩 단: `{int(protein_g)} g`")
        st.markdown(f"- 🥑 지: `{int(fat_g)} g`")
        card_end()
        spacer(6)

        card_start("③ 운동 요약", right_text="평균 포함")
        st.markdown(f"- 🏃 **총 운동 시간**: `{int(ex_total_min)} 분`")
        st.markdown(f"- 📊 **일 평균 운동**: `{int(avg_ex_min)} 분 / 일`")
        card_end()
        spacer(6)

        # ============================================
        # 🤖 실제 AI 판단 (FeedbackService 연동)
        # ============================================
        feedback_service = FeedbackService()

        # ✔ NutritionSummary 객체 생성 (FeedbackService 규격)
        nutrition = NutritionSummary(
            total={
                "kcal": total_kcal,
                "protein_g": protein_g,
                "fat_g": fat_g,
                "carbs_g": carb_g,
            }
        )

        # =========================
        # 🧠 종합 점수 계산
        # =========================

        # 1️⃣ 식사 점수 (기존)
        if avg_kcal <= 0:
            diet_score = 50
        elif kcal_target * 0.9 <= avg_kcal <= kcal_target * 1.1:
            diet_score = 80
        elif avg_kcal < kcal_target * 0.9:
            diet_score = 65
        else:
            diet_score = 55

        # 2️⃣ 운동 점수
        if ex_total_min <= 0:
            exercise_score = 40
        elif ex_total_min < 150:
            exercise_score = 65
        else:
            exercise_score = 80

        # 3️⃣ 체중 점수 (정체/감소/증가)
        weight_score = 60
        if df_weight is not None and df_weight.shape[0] >= 2:
            diff = df_weight.iloc[-1]["weight_kg"] - df_weight.iloc[0]["weight_kg"]
            if abs(diff) < 0.2:
                weight_score = 70
            elif diff < 0:
                weight_score = 80
            else:
                weight_score = 55

        # 4️⃣ 종합 점수 (가중치)
        score = int(
            diet_score * 0.5 +
            exercise_score * 0.3 +
            weight_score * 0.2
        )

        ai_messages = feedback_service.weekly_feedback(
            score=score,
            nutrition=nutrition,
            exercise_total_min=ex_total_min,
            weight_diff=(
                df_weight.iloc[-1]["weight_kg"] - df_weight.iloc[0]["weight_kg"]
                if df_weight is not None and df_weight.shape[0] >= 2
                else None
            ),
            has_record=days_count > 0,
            max_messages=3,  # 리포트는 3줄
        )

        card_start("④ 🧑‍🏫 Ara 코치 판단 요약", right_text="AI 분석")
        for msg in ai_messages:
            st.markdown(f"- 👨‍⚕️ {msg}")
        card_end()

    with sub_g:
        card_start("① 주간 체중 변화", right_text="그래프")
        st.pyplot(_plot_weight_line(df_weight, "주간 체중 변화"), use_container_width=True)
        card_end()
        spacer(6)

        card_start("② 일자별 섭취 kcal", right_text="그래프")
        st.pyplot(_plot_kcal_bar(df_meal, "일자별 섭취 kcal"), use_container_width=True)
        st.caption("※ 일자별 총 섭취 kcal만 표시 (시간 단위로 쪼개지지 않음)")
        card_end()
        spacer(6)

        card_start("③ 탄·단·지 비율", right_text="그래프")
        st.pyplot(_plot_macro_pie(carb_g, protein_g, fat_g, "탄·단·지 비율(%)"), use_container_width=True)
        card_end()
        spacer(6)

        card_start("④ 운동 시간 분포", right_text="그래프")
        if df_ex.empty:
            muted_text("운동 데이터가 없거나 집계할 수 없습니다. 운동 기록 후 다시 확인해 주세요.")
        st.pyplot(_plot_exercise_bar(df_ex, "일자별 운동 시간(분)"), use_container_width=True)
        st.caption("※ 하루 총 운동 시간 기준")
        card_end()

# ==================================================
# MONTHLY
# ==================================================
with tab_month:
    feedback_service = FeedbackService()
    col1, col2 = st.columns(2)

    with col1:
        selected_year = st.selectbox(
            "연도",
            list(range(today.year, today.year - 5, -1)),
            index=0,
            key="report_year"
        )

    with col2:
        selected_month = st.selectbox(
            "월",
            list(range(1, 13)),
            index=today.month - 1,
            key="report_month"
        )

    selected_month_date = date(selected_year, selected_month, 1)
    month_start, month_end = _month_range(selected_month_date)
    st.caption(f"{month_start} ~ {month_end}")

    df_meal = _fetch_meal_daily(user_id_raw, month_start, month_end)
    df_ex = _fetch_exercise_daily_safe(user_id_raw, month_start, month_end)
    df_weight = _fetch_weight_daily(user_id_raw, month_start, month_end)

    total_kcal = float(df_meal["total_kcal"].sum()) if not df_meal.empty else 0.0
    days_count = int(df_meal.shape[0]) if not df_meal.empty else 0
    avg_kcal = (total_kcal / days_count) if days_count else 0.0

    ex_total_min = float(df_ex["total_min"].sum()) if not df_ex.empty else 0.0
    ex_days = int(df_ex.shape[0]) if not df_ex.empty else 0
    days_in_month = (month_end - month_start).days + 1
    avg_ex_min = ex_total_min / days_in_month

    carb_g = float(df_meal["carb_g"].sum()) if not df_meal.empty else 0.0
    protein_g = float(df_meal["protein_g"].sum()) if not df_meal.empty else 0.0
    fat_g = float(df_meal["fat_g"].sum()) if not df_meal.empty else 0.0

    over_days, under_days, ok_days = _count_kcal_days(df_meal, kcal_target)

    sub_r, sub_g = st.tabs(["📝 리포트", "📊 그래프"])

    with sub_r:
        card_start("월간 요약", right_text="핵심")
        st.markdown(f"🔥 **월간 총 섭취**: `{int(total_kcal)} kcal`")
        st.markdown(f"🗓️ **기록 일수**: `{days_count} 일`")
        st.markdown(f"📊 **일 평균 섭취**: `{int(avg_kcal)} kcal`")
        st.markdown(f"🏃 **총 운동 시간**: `{int(ex_total_min)} 분`")
        st.markdown(f"📊 **일 평균 운동**: `{int(avg_ex_min)} 분 / 일`")
        card_end()
        spacer(6)

        card_start("① 체중 요약", right_text="월간")
        st.markdown(f"- {_weight_summary_text(df_weight)}")
        card_end()
        spacer(6)

        card_start("② 식사 요약", right_text="월간")
        st.markdown("**요약 지표**")
        st.markdown(f"- 평균 섭취: `{int(avg_kcal)} kcal` / 목표 `{int(kcal_target)} kcal`")
        st.markdown(f"- 과식(>+200): `{over_days}일` · 부족(<-200): `{under_days}일` · 적정(±200): `{ok_days}일`")
        st.markdown("**탄/단/지 총합(월간)**")
        st.markdown(f"- 🍚 탄: `{int(carb_g)} g`")
        st.markdown(f"- 🥩 단: `{int(protein_g)} g`")
        st.markdown(f"- 🥑 지: `{int(fat_g)} g`")
        card_end()
        spacer(6)

        # card_start("③ AI 종합 판단", right_text="2줄 요약")
        # l1, l2 = _monthly_ai_2lines(df_meal)
        # st.markdown(l1)
        # st.markdown(f"- ✅ {l2}")
        # card_end()

    nutrition = NutritionSummary(
        total={
            "kcal": total_kcal,
            "protein_g": protein_g,
            "fat_g": fat_g,
            "carbs_g": carb_g,
        }
    )

    # =========================
    # 🧠 종합 점수 계산
    # =========================

    # 1️⃣ 식사 점수 (기존)
    if avg_kcal <= 0:
        diet_score = 50
    elif kcal_target * 0.9 <= avg_kcal <= kcal_target * 1.1:
        diet_score = 80
    elif avg_kcal < kcal_target * 0.9:
        diet_score = 65
    else:
        diet_score = 55

    # 2️⃣ 운동 점수
    if ex_total_min <= 0:
        exercise_score = 40
    elif ex_total_min < 150:
        exercise_score = 65
    else:
        exercise_score = 80

    # 3️⃣ 체중 점수 (정체/감소/증가)
    weight_score = 60
    if df_weight is not None and df_weight.shape[0] >= 2:
        diff = df_weight.iloc[-1]["weight_kg"] - df_weight.iloc[0]["weight_kg"]
        if abs(diff) < 0.2:
            weight_score = 70
        elif diff < 0:
            weight_score = 80
        else:
            weight_score = 55

    # 4️⃣ 종합 점수 (가중치)
    score = int(
        diet_score * 0.5 +
        exercise_score * 0.3 +
        weight_score * 0.2
    )

    ai_messages = feedback_service.weekly_feedback(
        score=score,
        nutrition=nutrition,
        exercise_total_min=ex_total_min,
        weight_diff=(
            df_weight.iloc[-1]["weight_kg"] - df_weight.iloc[0]["weight_kg"]
            if df_weight is not None and df_weight.shape[0] >= 2
            else None
        ),
        has_record=days_count > 0,
        max_messages=2,  # 월간은 2줄 추천
    )

    card_start("③ 🧑‍🏫 Ara 코치 종합 판단", right_text="AI 분석")
    for msg in ai_messages[:2]:
        st.markdown(f"- 👨‍⚕️ {msg}")
    card_end()

    with sub_g:
        card_start("① 월간 체중 변화", right_text="그래프")
        st.pyplot(_plot_weight_line(df_weight, "월간 체중 변화"), use_container_width=True)
        card_end()
        spacer(6)

        card_start("② 월간 섭취 추이", right_text="그래프")
        st.pyplot(_plot_kcal_bar(df_meal, "월간 일자별 섭취 kcal"), use_container_width=True)
        card_end()
        spacer(6)

        card_start("③ 월간 탄·단·지 비율", right_text="그래프")
        st.pyplot(_plot_macro_pie(carb_g, protein_g, fat_g, "월간 탄·단·지 비율(%)"), use_container_width=True)
        card_end()
        spacer(6)

        card_start("④ 월간 운동 추이", right_text="그래프")
        if df_ex.empty:
            muted_text("운동 데이터가 없거나 집계할 수 없습니다. 운동 기록 후 다시 확인해 주세요.")
        st.pyplot(_plot_exercise_bar(df_ex, "월간 일자별 운동 시간(분)"), use_container_width=True)
        st.caption("※ 하루 총 운동 시간 기준")
        card_end()

spacer(10)
app_container_end()
bottom_nav(active="report")
