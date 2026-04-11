from __future__ import annotations

import sys
from pathlib import Path
import json
from types import SimpleNamespace
import streamlit as st
from datetime import date, datetime

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
from infra.db_server import get_db_conn
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

from services.goal_message_service import GoalMessageService

# ==================================================
# Utils
# ==================================================
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


def _to_date(v):
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    return None


def _ko_sex(v):
    return {"male": "남성", "female": "여성"}.get((v or "").strip(), "-")


def _ko_activity(v):
    return {"low": "낮음", "medium": "보통", "high": "높음"}.get((v or "").strip(), "-")


def _goal_label(gtype: str) -> str:
    s = (gtype or "").strip()
    mapping = {
        "weight_loss": "체중 감량",
        "maintenance": "체중 유지",
        "maintain": "체중 유지",
        "muscle_gain": "근육 증가",
        "weight_gain": "체중 증량",
    }
    return mapping.get(s, s or "-")


def _safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
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
profile = None
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
            preferences
        FROM user_profile
        WHERE user_id = :u
        """,
        {"u": user_id_raw},
    )
    profile = cur.fetchone()

baseline_weight = None
if profile:
    baseline_weight = _safe_float(profile[3], None)

# ==================================================
# Load active goal
# ==================================================
goal = None
with get_db_conn() as conn:
    cur = conn.cursor()
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
    goal = cur.fetchone()

# ==================================================
# UI
# ==================================================
app_container_start()
st.markdown("## ⚙️ 설정")
st.caption("내 정보와 목표를 확인하고 관리할 수 있어요.")
spacer(12)

# ==================================================
# 👤 Profile
# ==================================================
card_start("👤 내 프로필")

if not profile:
    muted_text("프로필 정보가 없습니다.")
    if primary_button("프로필 설정"):
        st.switch_page("pages/onboarding.py")
else:
    sex, birth, height, weight, activity, cond, pref = profile

    st.markdown(f"- **성별**: {_ko_sex(sex)}")
    st.markdown(f"- **출생년도**: {birth or '-'}")
    st.markdown(f"- **키 / 시작 체중**: {height or '-'} cm / {weight or '-'} kg")
    st.markdown(f"- **활동 수준**: {_ko_activity(activity)}")

    cond_list = _load_json(cond, [])
    pref_list = _load_json(pref, [])

    st.markdown(f"- **건강 유의사항**: {', '.join(cond_list) if cond_list else '없음'}")
    st.markdown(f"- **선호 운동**: {', '.join(pref_list) if pref_list else '없음'}")

    spacer(6)
    if primary_button("✏️ 프로필 수정"):
        st.switch_page("pages/usr_profile.py")

card_end()
spacer(12)

# ==================================================
# 🎯 Goal
# ==================================================
card_start("🎯 목표")

if not goal:
    muted_text("설정된 목표가 없습니다.")
    if primary_button("목표 설정"):
        st.switch_page("pages/goal_setup.py")
else:
    g_type, start_dt, target_dt, target_w, kcal, macro_target = goal
    start_dt = _to_date(start_dt)
    target_dt = _to_date(target_dt)

    kcal_i = int(_safe_float(kcal, 0))
    target_w_f = _safe_float(target_w, None)

    st.markdown(f"- **목표 유형**: {_goal_label(g_type)}")
    st.markdown(f"- **하루 목표 열량**: **{kcal_i:,} kcal**")

    # ✅ 계산 근거(기간/목표체중 기반 문구 복구)
    if baseline_weight is not None and start_dt and target_dt and target_w_f is not None:
        days = (target_dt - start_dt).days
        if days > 0:
            st.caption(
                f"현재 체중 {baseline_weight:.1f}kg 기준 → 목표 체중 {target_w_f:.1f}kg, "
                f"{days}일 목표 설정을 반영한 값입니다."
            )

            # ✅ “변화 폭 크게 설정 시” 안내/경고 복구 (너무 빡센 속도)
            delta_kg = target_w_f - baseline_weight
            kg_per_week = abs(delta_kg) / (days / 7.0)

            # 보수적 기준(체중감량/증량 모두): 주 1.0kg 초과면 경고
            if kg_per_week >= 1.0:
                st.warning(
                    "⚠️ 목표 기간 대비 체중 변화 속도가 빠릅니다. "
                    "목표 기간을 늘리거나 중간 목표를 설정하는 것을 권장합니다."
                )

    # ✅ kcal 자체가 비현실적인 범위면 경고 (GoalMessageService 기준과 동일)
    if kcal_i and kcal_i < 1200:
        st.warning(
            "⚠️ 하루 목표 열량이 매우 낮습니다. "
            "지속 가능성을 위해 목표 기간을 늘리거나 중간 목표를 권장합니다."
        )
    if kcal_i and kcal_i > 3500:
        st.warning(
            "⚠️ 하루 목표 열량이 매우 높습니다. "
            "체지방 증가/소화 부담을 고려해 완만한 목표를 추천합니다."
        )

    # ✅ 앵커 메시지: GoalMessageService로 생성 (pydantic 리터럴/검증 때문에 안전 호출)
    try:
        msg_service = GoalMessageService()
        goal_obj = SimpleNamespace(
            goal_type=(g_type or "").strip(),
            kcal_target=kcal_i,
        )
        anchor = msg_service.build_next_goal(goal_obj, daily=None, weekly=None, event=None)
        muted_text(anchor)
    except Exception:
        # 절대 화면 깨지지 않게 최소 fallback
        if (g_type or "").strip() == "weight_loss":
            muted_text("체중 감량 목표를 유지합니다. 오늘 계획을 차분히 실천해 보세요.")
        elif (g_type or "").strip() in ("weight_gain", "muscle_gain"):
            muted_text("체중 증가 목표를 유지합니다. 오늘 계획을 차분히 실천해 보세요.")
        else:
            muted_text("현재 체중을 안정적으로 유지합니다. 오늘 계획을 차분히 실천해 보세요.")

    spacer(6)
    if primary_button("🎯 목표 수정"):
        st.switch_page("pages/goal_edit.py")

card_end()
spacer(12)

# ==================================================
# 🔐 Account
# ==================================================
card_start("🔐 계정")
if primary_button("로그아웃"):
    logout()
    st.stop()
card_end()

spacer(20)
app_container_end()
bottom_nav(active="settings")
