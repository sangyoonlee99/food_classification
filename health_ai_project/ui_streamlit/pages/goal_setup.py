# ui_streamlit/pages/goal_setup.py
from __future__ import annotations

import sys
from pathlib import Path
from datetime import date
import json
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui_streamlit.utils.auth import require_login
from ui_streamlit.components import (
    app_container_start,
    app_container_end,
    card_start,
    card_end,
    spacer,
)
from infra.db_server import get_db_conn
from infra.repositories.user_repository import UserRepository
from services.goal_calculator_service import GoalCalculatorService


# ==================================================
# Utils (RAW(16) 변환 – 다른 페이지와 동일)
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


# ==================================================
# Auth
# ==================================================
require_login()
user_id = st.session_state.get("user_id")
user_id_raw = _ensure_bytes16(user_id)

repo = UserRepository()

# ==================================================
# 프로필 확인
# ==================================================
profile = repo.get_user_profile(user_id=user_id_raw)

if not profile:
    app_container_start()
    st.warning("⚠️ 기본 정보가 없습니다. 먼저 온보딩을 완료해주세요.")
    st.page_link(
        "pages/onboarding.py",
        label="온보딩으로 이동",
        icon="➡️",
        use_container_width=True,
    )
    app_container_end()
    st.stop()

# 시작 체중 (baseline 기준)
weight_kg = (
    profile.get("weight_kg_baseline")
    or profile.get("weight_kg")
    or profile.get("current_weight")
    or 0
)

# ==================================================
# UI
# ==================================================
app_container_start()
st.markdown("### 🎯 목표 설정")
st.caption("목표만 선택하면 이후 추천의 기준이 됩니다.")
spacer(12)

card_start("목표 선택")

goal_type = st.selectbox(
    "목표 유형",
    ["weight_loss", "maintenance", "weight_gain"],
    format_func=lambda x: {
        "weight_loss": "체중 감량",
        "maintenance": "체중 유지",
        "weight_gain": "체중 증량",
    }[x],
)

# ===============================
# 🎯 목표 체중 기본값 자동 보정
# ===============================
default_target_weight = weight_kg
if goal_type == "weight_loss":
    default_target_weight = max(weight_kg - 5, 30)
elif goal_type == "weight_gain":
    default_target_weight = weight_kg + 5

use_target_weight = st.checkbox(
    "목표 체중 설정",
    value=(goal_type != "maintenance"),
)

target_weight = None
if use_target_weight:
    target_weight = st.number_input(
        "목표 체중 (kg)",
        min_value=30.0,
        max_value=200.0,
        value=float(default_target_weight),
    )

# ===============================
# ⏳ 목표 기간
# ===============================
target_date = None
if goal_type != "maintenance":
    target_date = st.date_input(
        "목표 기간 (종료일)",
        value=date.today().replace(month=date.today().month + 2)
        if date.today().month <= 10
        else date.today(),
    )

card_end()
spacer(12)

# ==================================================
# 🧠 AI 현실성 판단 (UI 전용)
# ==================================================
if target_weight and target_date and goal_type != "maintenance":
    days = (target_date - date.today()).days

    if days > 0:
        weeks = days / 7
        delta = abs(target_weight - weight_kg)
        weekly_change = delta / weeks

        if goal_type == "weight_loss":
            if weekly_change > 1.2:
                st.warning(
                    f"⚠️ **주당 {weekly_change:.1f}kg 감량 목표는 다소 빠른 편입니다.**\n\n"
                    "👉 일반적으로 **주당 0.5~1.0kg 감량**이 지속 가능해요."
                )
            else:
                st.info(
                    f"✅ **현실적인 감량 목표입니다.**\n\n"
                    f"👉 주당 약 {weekly_change:.1f}kg 감량 페이스예요."
                )

        elif goal_type == "weight_gain":
            if weekly_change > 0.7:
                st.warning(
                    f"⚠️ **주당 {weekly_change:.1f}kg 증량은 체지방 증가 위험이 있어요.**\n\n"
                    "👉 **주당 0.25~0.5kg 증량**을 권장합니다."
                )
            else:
                st.info(
                    f"✅ **현실적인 증량 목표입니다.**\n\n"
                    f"👉 주당 약 {weekly_change:.2f}kg 증량 페이스예요."
                )

spacer(16)

# ==================================================
# Save
# ==================================================
if st.button("이 설정으로 시작하기", use_container_width=True):
    svc = GoalCalculatorService()

    # 🔥 핵심 수정 포인트 (user_id, start_date 추가)
    result = svc.calculate(
        user_id=user_id_raw,
        start_date=date.today(),

        sex=profile["sex"],
        birth_year=profile["birth_year"],
        height_cm=profile["height_cm"],
        weight_kg=(
                profile.get("weight_kg_baseline")
                or profile.get("weight_kg")
                or profile.get("current_weight")
                or 0
        ),
        activity_level=profile["activity_level"],
        goal_type=goal_type,
        target_weight=target_weight,
        target_date=target_date,
    )

    with get_db_conn() as conn:
        cur = conn.cursor()

        cur.execute(
            """
            UPDATE user_goal
               SET is_active = 'N'
             WHERE user_id = :user_id
            """,
            {"user_id": user_id_raw},
        )

        cur.execute(
            """
            INSERT INTO user_goal (
                goal_id, user_id, goal_type,
                start_date, target_date, target_weight_kg,
                kcal_target, macro_target,
                is_active
            )
            VALUES (
                SYS_GUID(), :user_id, :goal_type,
                TRUNC(SYSDATE), :target_date, :target_weight,
                :kcal_target, :macro_target,
                'Y'
            )
            """,
            {
                "user_id": user_id_raw,
                "goal_type": goal_type,
                "target_date": target_date,
                "target_weight": target_weight,
                "kcal_target": result["kcal_target"],
                "macro_target": json.dumps(
                    result["macro_target"],
                    ensure_ascii=False,
                ),
            },
        )

        conn.commit()

    st.success("✅ 목표 및 하루 기준이 설정되었습니다.")
    st.switch_page("pages/home.py")
    st.stop()

app_container_end()
