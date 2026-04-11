# ui_streamlit/pages/record_exercise.py
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, date
import json
import streamlit as st

from services.recommendation_history_loader import load_recommendation_state
from services.exercise_service import (
    ExerciseService,
    delete_exercise_record,
    update_exercise_record,
)

# ==================================================
# Path
# ==================================================
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ==================================================
# Imports
# ==================================================
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

# ==================================================
# 운동 선택 기준 (DB 저장용 표준값)
# ==================================================
EXERCISE_OPTIONS = {
    "유산소": {
        "걷기": "걷기",
        "조깅": "조깅",
        "러닝": "러닝",
        "러닝머신": "러닝머신",
        "자전거": "자전거",
    },
    "근력": {
        "홈트레이닝": "홈트레이닝",
        "헬스": "헬스",
        "크로스핏": "크로스핏",
    },
    "기타": {
        "잘모름 / 기타": "기타",
    },
}

DISPLAY_EXERCISE_LABEL = {
    "러닝": "러닝머신",
    "러닝머신": "러닝머신",
}

INTENSITY_LABEL = {
    "low": "🟢 낮음",
    "normal": "🟡 보통",
    "high": "🔴 높음",
}

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
        s = v.replace("0x", "").strip()
        b = bytes.fromhex(s)
        if len(b) != 16:
            raise ValueError("user_id hex must be 32 chars")
        return b
    raise TypeError("invalid user_id type")


def calc_hr_range(age: int, intensity: str) -> tuple[int, int]:
    max_hr = 220 - age
    if intensity == "low":
        return int(max_hr * 0.50), int(max_hr * 0.60)
    if intensity == "high":
        return int(max_hr * 0.75), int(max_hr * 0.85)
    return int(max_hr * 0.60), int(max_hr * 0.75)


def normalize_exercise_label(v) -> str:
    if not v:
        return ""

    if isinstance(v, (bytes, bytearray)):
        try:
            v = v.decode()
        except Exception:
            return ""

    if isinstance(v, list):
        return str(v[0]).strip() if v else ""

    s = str(v).strip()

    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list) and parsed:
                return str(parsed[0]).strip()
        except Exception:
            pass

    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]

    return s.strip()

# ==================================================
# Session Guard
# ==================================================
user_id = st.session_state.get("user_id")
if not user_id:
    st.error("로그인이 필요합니다.")
    st.stop()

user_id_raw = _ensure_bytes16(user_id)

# ==================================================
# Load user profile
# ==================================================
age: int | None = None
preferred_cardio: list[str] = []
preferred_strength: list[str] = []

with get_db_conn() as conn:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT birth_year, preferred_cardio, preferred_strength
        FROM user_profile
        WHERE user_id = :u
        """,
        {"u": user_id_raw},
    )
    prow = cur.fetchone()

if prow:
    birth_year, pref_cardio, pref_strength = prow
    if birth_year:
        age = date.today().year - int(birth_year)
    if pref_cardio:
        preferred_cardio = [normalize_exercise_label(x) for x in str(pref_cardio).split(",") if x.strip()]
    if pref_strength:
        preferred_strength = [normalize_exercise_label(x) for x in str(pref_strength).split(",") if x.strip()]

# ==================================================
# Header
# ==================================================
app_container_start()
st.markdown("### 🏃 운동 기록")

rec = load_recommendation_state(user_id=user_id_raw) or {}
badge = ((rec.get("ctx") or {}).get("recommendation") or {}).get("badge") or {}
if badge:
    st.info(badge.get("label", ""))

spacer(12)

# ==================================================
# Input
# ==================================================
card_start("운동 정보 입력")

category = st.selectbox("운동 분류", list(EXERCISE_OPTIONS.keys()))

base_options = [normalize_exercise_label(x) for x in EXERCISE_OPTIONS[category].keys()]
base_options = [x for x in base_options if x]
base_options = list(dict.fromkeys(base_options))

if category == "유산소" and preferred_cardio:
    pref = [x for x in preferred_cardio if x in base_options]
    options = pref + [x for x in base_options if x not in pref]
elif category == "근력" and preferred_strength:
    pref = [x for x in preferred_strength if x in base_options]
    options = pref + [x for x in base_options if x not in pref]
else:
    options = base_options

exercise_label = st.selectbox("운동 종류", options)
exercise_type = exercise_label

minutes = st.number_input("운동 시간 (분)", min_value=5, step=5, value=30)

intensity = st.selectbox(
    "운동 강도",
    ["low", "normal", "high"],
    index=1,
    format_func=lambda x: {
        "low": "낮음 (가볍게 숨이 참)",
        "normal": "보통 (대화가 약간 힘듦)",
        "high": "높음 (대화 어려움)",
    }[x],
)

if age:
    lo, hi = calc_hr_range(age, intensity)
    st.caption(f"💓 권장 심박수 범위: {lo} ~ {hi} bpm (만 {age}세 기준)")

card_end()
spacer(12)

# ==================================================
# Action
# ==================================================
if primary_button("운동 기록 저장"):
    ExerciseService().record_exercise(
        user_id=user_id_raw,
        exercise_type=exercise_type,
        minutes=int(minutes),
        intensity=intensity,
        performed_at=datetime.now(),
        source="manual",
    )
    st.cache_data.clear()  # ✅ 캐시가 있으면 같이 갱신
    st.success("✅ 운동 기록이 저장되었습니다.")
    st.rerun()

# ==================================================
# Today Records
# ==================================================
spacer(16)
card_start("📋 오늘의 운동 기록")

with get_db_conn() as conn:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT exercise_id, performed_at, exercise_type, minutes, intensity
        FROM (
            SELECT
                exercise_id,
                performed_at,
                exercise_type,
                minutes,
                intensity
            FROM exercise_record
            WHERE user_id = :u
              AND TRUNC(performed_at) = TRUNC(SYSDATE)
            ORDER BY performed_at DESC
        )
        WHERE ROWNUM <= 7
        """,
        {"u": user_id_raw},
    )
    rows = cur.fetchall()

if not rows:
    muted_text("아직 기록된 운동이 없습니다.")
else:
    for exercise_id, performed_at, ex_type, minutes, intensity in rows:
        key = exercise_id.hex()
        norm = normalize_exercise_label(ex_type)
        display = DISPLAY_EXERCISE_LABEL.get(norm, norm)
        intensity_label = INTENSITY_LABEL.get(intensity, intensity)

        with st.expander(
            f"{performed_at:%m/%d} · {display} · {int(minutes)}분 · {intensity_label}",
            expanded=False,
        ):
            new_minutes = st.number_input(
                "운동 시간(분)",
                min_value=5,
                step=5,
                value=int(minutes),
                key=f"m_{key}",
            )
            new_intensity = st.selectbox(
                "강도",
                ["low", "normal", "high"],
                index=["low", "normal", "high"].index(intensity),
                key=f"i_{key}",
            )

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("수정 저장", key=f"edit_{key}"):
                    update_exercise_record(
                        user_id=user_id_raw,
                        exercise_id=exercise_id,
                        minutes=int(new_minutes),
                        intensity=new_intensity,
                        performed_at=performed_at,
                    )
                    st.cache_data.clear()
                    st.success("수정되었습니다.")
                    st.rerun()

            with col_b:
                if st.button("삭제", key=f"del_{key}"):
                    delete_exercise_record(
                        user_id=user_id_raw,
                        exercise_id=exercise_id,
                        performed_at=performed_at,
                    )
                    st.cache_data.clear()
                    st.success("삭제되었습니다.")
                    st.rerun()

card_end()
app_container_end()

spacer(80)
bottom_nav(active="record")
