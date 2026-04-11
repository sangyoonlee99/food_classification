from __future__ import annotations

import sys
import json
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
from ui_streamlit.utils.auth import require_login
from infra.db_server import get_db_conn
from ui_streamlit.components import (
    app_container_start,
    app_container_end,
    card_start,
    card_end,
    section_header,
    spacer,
    primary_button,
    muted_text,
    bottom_nav,
)

# ==================================================
# Utils
# ==================================================
def _ensure_bytes16(v) -> bytes:
    if isinstance(v, (bytes, bytearray)):
        if len(v) != 16:
            raise ValueError("user_id must be RAW(16)")
        return bytes(v)
    if isinstance(v, str):
        b = bytes.fromhex(v.replace("0x", "").strip())
        if len(b) != 16:
            raise ValueError("user_id hex must be 32 chars")
        return b
    raise TypeError("invalid user_id type")

def _read_lob(v):
    if v is None:
        return None
    if hasattr(v, "read"):
        v = v.read()
    if isinstance(v, (bytes, bytearray)):
        try:
            v = v.decode("utf-8")
        except Exception:
            v = v.decode("utf-8", errors="ignore")
    return v

def _load_json(v, default):
    s = _read_lob(v)
    if not s:
        return default
    try:
        return json.loads(s)
    except Exception:
        return default

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
# Load profile
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
            activity_level,
            conditions,
            preferred_cardio,
            preferred_strength
        FROM user_profile
        WHERE user_id = :u
        """,
        {"u": user_id_raw},
    )
    row = cur.fetchone()

if not row:
    st.error("프로필 정보가 없습니다. 초기 설정이 필요합니다.")
    st.switch_page("pages/onboarding.py")
    st.stop()

sex, birth_year, height_cm, weight_kg, activity_level, conditions, preferred_cardio, preferred_strength = row


conditions = _load_json(conditions, [])
preferred_cardio = _load_json(preferred_cardio, [])
preferred_strength = _load_json(preferred_strength, [])

# ⛑ 방어 (무조건 list)
if not isinstance(conditions, list):
    conditions = []
if not isinstance(preferred_cardio, list):
    preferred_cardio = []
if not isinstance(preferred_strength, list):
    preferred_strength = []

# ==================================================
# UI
# ==================================================
app_container_start()
st.markdown("## 👤 프로필 수정")
st.caption("신체 정보와 생활 패턴을 수정할 수 있어요.")
spacer(12)

# ==================================================
# 기본 정보
# ==================================================
section_header("기본 정보")
card_start()

sex = st.radio(
    "성별",
    options=["male", "female"],
    index=0 if sex == "male" else 1,
    format_func=lambda x: "남성" if x == "male" else "여성",
)

birth_year = st.number_input(
    "출생년도",
    min_value=1900,
    max_value=date.today().year,
    value=int(birth_year),
    step=1,
)

height_cm = st.number_input(
    "키 (cm)",
    min_value=120.0,
    max_value=230.0,
    value=float(height_cm),
    step=0.1,
)

weight_kg = st.number_input(
    "시작 체중 (kg)",
    min_value=30.0,
    max_value=300.0,
    value=float(weight_kg),
    step=0.1,
)

card_end()
spacer(10)

# ==================================================
# 활동 수준
# ==================================================
section_header("활동 수준")
card_start()

activity_level = st.radio(
    "활동 수준",
    options=["low", "medium", "high"],
    index=["low", "medium", "high"].index(activity_level),
    format_func=lambda x: {
        "low": "낮음 (좌식 생활)",
        "medium": "보통 (일반 활동)",
        "high": "높음 (활동량 많음)",
    }[x],
)

card_end()
spacer(10)

# ==================================================
# 건강 유의사항
# ==================================================
section_header("건강 유의사항")
card_start()

CONDITIONS = ["해당없음", "당뇨", "고혈압"]
conditions = st.multiselect(
    "해당되는 항목을 선택하세요",
    options=CONDITIONS,
    default=conditions,
)

card_end()
spacer(10)

# ==================================================
# 선호 운동
# ==================================================
section_header("선호 운동")
card_start()

# === 옵션을 DB 저장 기준과 완전히 동일하게 ===
CARDIO_OPTIONS = ["걷기", "조깅", "러닝", "러닝머신", "자전거", "수영"]
STRENGTH_OPTIONS = ["홈트레이닝", "헬스", "크로스핏", "근력운동"]

# === 방어: DB 값이 옵션에 없으면 제거 ===
preferred_cardio = [x for x in preferred_cardio if x in CARDIO_OPTIONS]
preferred_strength = [x for x in preferred_strength if x in STRENGTH_OPTIONS]

preferred_cardio = st.multiselect(
    "선호 유산소 운동",
    options=CARDIO_OPTIONS,
    default=preferred_cardio,
)

preferred_strength = st.multiselect(
    "선호 근력 운동",
    options=STRENGTH_OPTIONS,
    default=preferred_strength,
)


card_end()
spacer(14)

# ==================================================
# Save
# ==================================================
if primary_button("저장하기"):
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE user_profile
            SET
                sex = :sex,
                birth_year = :birth,
                height_cm = :height,
                weight_kg_baseline = :weight,
                activity_level = :activity,
                conditions = :conditions,
                preferred_cardio = :preferred_cardio,
                preferred_strength = :preferred_strength
            WHERE user_id = :u
            """,
            {
                "sex": sex,
                "birth": int(birth_year),
                "height": float(height_cm),
                "weight": float(weight_kg),
                "activity": activity_level,
                "conditions": json.dumps(conditions, ensure_ascii=False),
                "preferred_cardio": json.dumps(preferred_cardio, ensure_ascii=False),
                "preferred_strength": json.dumps(preferred_strength, ensure_ascii=False),
                "u": user_id_raw,
            },
        )

        conn.commit()

    st.success("프로필이 저장되었습니다.")
    st.switch_page("pages/settings.py")

spacer(20)
app_container_end()
bottom_nav(active="settings")
