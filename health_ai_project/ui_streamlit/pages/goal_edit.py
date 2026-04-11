# ui_streamlit/pages/goal_edit.py
from __future__ import annotations

import sys
from pathlib import Path
from datetime import date, datetime
import json
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
from ui_streamlit.utils.auth import require_login
from ui_streamlit.components import (
    app_container_start,
    app_container_end,
    card_start,
    card_end,
    spacer,
    primary_button,
    muted_text,
    bottom_nav,
)
from infra.db_server import get_db_conn
from services.goal_calculator_service import GoalCalculatorService

# (선택) 목표 안내 문구 서비스가 있으면 사용 (없어도 페이지 동작해야 함)
try:
    from services.goal_message_service import GoalMessageService  # type: ignore
except Exception:
    GoalMessageService = None  # type: ignore


# ==================================================
# Utils
# ==================================================
def _ensure_bytes16(v) -> bytes:
    if isinstance(v, (bytes, bytearray)):
        b = bytes(v)
        if len(b) != 16:
            raise ValueError("user_id must be RAW(16)")
        return b
    if isinstance(v, str):
        s = v.strip()
        if s.lower().startswith("0x"):
            s = s[2:]
        b = bytes.fromhex(s)
        if len(b) != 16:
            raise ValueError("user_id hex must be 32 chars")
        return b
    raise TypeError("invalid user_id type")


def _to_date(v):
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    return None


# ✅ UI 값 -> 엔진/스키마 값 정규화
def _normalize_goal_type_for_engine(ui_goal_type: str) -> str:
    s = (ui_goal_type or "").strip()
    if s == "maintain":
        return "maintenance"
    if s == "weight_gain":
        return "muscle_gain"
    if s == "muscle_gain":
        return "muscle_gain"
    if s == "maintenance":
        return "maintenance"
    return "weight_loss"


def _normalize_goal_type_for_ui(db_goal_type: str) -> str:
    s = (db_goal_type or "").strip()
    if s in ("maintain", "maintenance"):
        return "maintain"
    if s in ("weight_gain", "muscle_gain"):
        return "weight_gain"
    return "weight_loss"


GOAL_TYPE_LABEL = {
    "weight_loss": "체중 감량",
    "maintain": "체중 유지",
    "weight_gain": "체중 증량",
}


# ==================================================
# Auth
# ==================================================
require_login()
user_id = st.session_state.get("user_id")
if not user_id:
    st.switch_page("pages/login.py")
    st.stop()

user_id_raw = _ensure_bytes16(user_id)

# ==================================================
# Load profile + goal
# ==================================================
with get_db_conn() as conn:
    cur = conn.cursor()

    cur.execute(
        """
        SELECT
            sex,
            birth_year,
            height_cm,
            weight_kg_baseline,
            activity_level
        FROM user_profile
        WHERE user_id = :u
        """,
        {"u": user_id_raw},
    )
    profile = cur.fetchone()

    cur.execute(
        """
        SELECT
            goal_type,
            start_date,
            target_date,
            target_weight_kg,
            kcal_target,
            macro_target
        FROM user_goal
        WHERE user_id = :u
          AND is_active = 'Y'
        """,
        {"u": user_id_raw},
    )
    goal_row = cur.fetchone()

app_container_start()
st.markdown("## 🎯 목표 수정")
spacer(8)

# ==================================================
# Guard
# ==================================================
if not profile or not goal_row:
    card_start("⚠️ 목표 수정 불가")
    muted_text("프로필 또는 목표 정보가 없습니다. 초기 설정을 먼저 완료해주세요.")
    card_end()
    spacer(16)
    app_container_end()
    bottom_nav(active="settings")
    st.stop()

sex, birth_year, height_cm, weight_kg, activity_level = profile
db_goal_type, start_dt, target_dt, target_weight, kcal_target_db, macro_target_db = goal_row

start_dt = _to_date(start_dt) or date.today()
target_dt = _to_date(target_dt) or (date.today().replace(year=date.today().year + 1))

# ==================================================
# Form
# ==================================================
card_start("🎯 목표 정보")

goal_type_ui_default = _normalize_goal_type_for_ui(db_goal_type)

goal_type_ui = st.selectbox(
    "목표 유형",
    options=["weight_loss", "maintain", "weight_gain"],
    index=["weight_loss", "maintain", "weight_gain"].index(goal_type_ui_default),
    format_func=lambda x: GOAL_TYPE_LABEL.get(x, x),
)

col1, col2 = st.columns(2)
with col1:
    start_dt_ui = st.date_input("시작일", value=start_dt)
with col2:
    target_dt_ui = st.date_input("목표일", value=target_dt)

target_weight_ui = st.number_input(
    "목표 체중 (kg)",
    value=float(target_weight or weight_kg),
    step=0.1,
    format="%.1f",
)

# --- 안내/경고 ---
engine_goal_type_preview = _normalize_goal_type_for_engine(goal_type_ui)
calc = GoalCalculatorService()
preview = calc.calculate(
    user_id=user_id_raw,
    sex=sex,
    birth_year=int(birth_year or date.today().year),
    height_cm=float(height_cm or 0),
    weight_kg=float(weight_kg or 0),
    activity_level=activity_level,
    goal_type=engine_goal_type_preview,
    start_date=start_dt_ui,
    target_weight=float(target_weight_ui),
    target_date=target_dt_ui,
)
preview_kcal = int(preview.get("kcal_target") or 0)

muted_text(
    f"하루 목표 열량과 매크로는 프로필·목표·기간 기준으로 자동 재계산됩니다.\n"
    f"➡️ 미리보기 목표 열량: **{preview_kcal:,} kcal**"
)

# GoalMessageService가 있으면 앵커 메시지까지(가능할 때만)
if GoalMessageService is not None:
    try:
        gms = GoalMessageService()  # type: ignore
        # GoalMessageService는 build_next_goal(goal=UserGoal...) 구조라
        # UI에서 여기까지 강결합하지 않고, Settings 쪽에서 사용하도록 두는 게 안전합니다.
    except Exception:
        pass

card_end()
spacer(12)

# ==================================================
# Save
# ==================================================
if primary_button("💾 저장"):
    engine_goal_type = _normalize_goal_type_for_engine(goal_type_ui)

    calculator = GoalCalculatorService()

    # ✅ 핵심 수정: user_id 반드시 전달
    result = calculator.calculate(
        user_id=user_id_raw,
        sex=sex,
        birth_year=int(birth_year or date.today().year),
        height_cm=float(height_cm or 0),
        weight_kg=float(weight_kg or 0),  # 현재(기준) 체중
        activity_level=activity_level,
        goal_type=engine_goal_type,
        start_date=start_dt_ui,
        target_weight=float(target_weight_ui),
        target_date=target_dt_ui,
    )

    kcal_target = int(result.get("kcal_target") or 0) or 2000
    macro_target = result.get("macro_target") or {}

    with get_db_conn() as conn:
        cur = conn.cursor()

        cur.execute(
            """
            UPDATE user_goal
            SET
                goal_type = :goal_type,
                start_date = :start_date,
                target_date = :target_date,
                target_weight_kg = :target_weight,
                kcal_target = :kcal_target,
                macro_target = :macro_target
            WHERE user_id = :u
              AND is_active = 'Y'
            """,
            {
                "goal_type": engine_goal_type,
                "start_date": start_dt_ui,
                "target_date": target_dt_ui,
                "target_weight": float(target_weight_ui),
                "kcal_target": kcal_target,
                "macro_target": json.dumps(macro_target, ensure_ascii=False),
                "u": user_id_raw,
            },
        )

        # engine_state 테이블이 있을 때만 무효화
        try:
            cur.execute(
                """
                UPDATE engine_state
                SET
                    last_execution = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE user_id = :u
                """,
                {"u": user_id_raw},
            )
        except Exception:
            pass

        conn.commit()

    # ✅ Streamlit 세션 캐시 무효화 (주간 식단표 즉시 반영)
    for k in [
        "weekly_meal_plan_cache",
        "last_recommendation",
        "event_dirty",
    ]:
        st.session_state.pop(k, None)

    st.success("목표가 저장되었고, 식단·루틴을 다시 계산합니다.")
    st.switch_page("pages/settings.py")

spacer(16)
app_container_end()
bottom_nav(active="settings")
