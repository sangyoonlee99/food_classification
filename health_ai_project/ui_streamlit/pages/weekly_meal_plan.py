# ui_streamlit/pages/weekly_meal_plan.py
from __future__ import annotations

import sys
from pathlib import Path
from datetime import date, timedelta, datetime
import json
import streamlit as st
import hashlib
from services.recommendation_layer import SIGNATURE_MESSAGES
from services.message_layer import build_replan_messages
from events.replan_orchestrator import ReplanOrchestrator
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
from routine.routine_engine import RoutineEngine

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

def _load_json(val, default=None):
    default = {} if default is None else default
    s = _read_lob(val)
    if not s:
        return default
    try:
        return json.loads(s)
    except Exception:
        return default

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

DOW_KR = ["월", "화", "수", "목", "금", "토", "일"]

MEAL_LABEL = {
    "breakfast": "🍳 아침",
    "lunch": "🍱 점심",
    "dinner": "🍽️ 저녁",
    "snack": "🍩 간식",
    "unknown": "🧩 기타",
}
MEAL_ORDER = ["breakfast", "lunch", "dinner", "snack"]

def _get_events(user_id_raw: bytes, d: date):
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT event_type, severity, note
            FROM event_log
            WHERE user_id = :u
              AND TRUNC(event_date) = TRUNC(:d)
            ORDER BY created_at
            """,
            {"u": user_id_raw, "d": d},
        )
        return cur.fetchall() or []


def _event_flags_from_rows(rows):
    flags = {
        "events": [],
        "social": False,
        "overtime": False,
        "travel": False,
        "meeting": False,
        "sleep_debt": False,
    }
    for et, sev, note in rows:
        flags["events"].append({"event_type": et, "severity": sev, "note": note})
        if et == "회식":
            flags["social"] = True
        if et == "야근":
            flags["overtime"] = True
        if et == "여행":
            flags["travel"] = True
        if et == "모임":
            flags["meeting"] = True
        if et == "수면부족":
            flags["sleep_debt"] = True
    return flags

def _get_actual_meals(user_id_raw: bytes, d: date):
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                eaten_at,
                NVL(meal_type, 'unknown') AS meal_type,
                food_name,
                NVL(amount_g, 0) AS amount_g,
                NVL(kcal, 0) AS kcal,
                NVL(carb_g, 0) AS carb_g,
                NVL(protein_g, 0) AS protein_g,
                NVL(fat_g, 0) AS fat_g
            FROM meal_record
            WHERE user_id = :u
              AND TRUNC(eaten_at) = TRUNC(:d)
            ORDER BY eaten_at
            """,
            {"u": user_id_raw, "d": d},
        )
        rows = cur.fetchall() or []

    by_type = {}
    totals = {"kcal": 0.0, "carb": 0.0, "protein": 0.0, "fat": 0.0}

    for eaten_at, meal_type, name, g, kcal, carb, protein, fat in rows:
        mt = (meal_type or "unknown").strip() or "unknown"
        by_type.setdefault(mt, []).append(
            {
                "time": eaten_at.strftime("%H:%M") if hasattr(eaten_at, "strftime") else "",
                "name": name or "",
                "g": float(g or 0),
                "kcal": float(kcal or 0),
                "carb": float(carb or 0),
                "protein": float(protein or 0),
                "fat": float(fat or 0),
            }
        )
        totals["kcal"] += float(kcal or 0)
        totals["carb"] += float(carb or 0)
        totals["protein"] += float(protein or 0)
        totals["fat"] += float(fat or 0)

    return by_type, totals

# --- 추천 식단: 그룹 정규화(탄/단/지/채소 고정 표기) ---
def _norm_group(x: str) -> str:
    s = (x or "").strip().lower()
    if any(k in s for k in ["carb", "탄수", "탄", "rice", "bread", "noodle", "곡", "면", "밥"]):
        return "carb"
    if any(k in s for k in ["protein", "단백", "단", "chicken", "egg", "fish", "meat", "두부"]):
        return "protein"
    if any(k in s for k in ["fat", "지방", "지", "oil", "nuts", "avocado", "견과", "올리브"]):
        return "fat"
    if any(k in s for k in ["vegetable", "veg", "채소", "야채", "샐러드", "broccoli", "spinach"]):
        return "vegetable"
    return "etc"

def _normalize_plan_items(items):
    """
    RoutineEngine/Planner에서 나오는 item dict를 탄/단/지/채 렌더링용 표준 형태로 정규화
    """
    out = []
    for item in items or []:
        if isinstance(item, dict):
            name = item.get("name") or item.get("food_name") or item.get("food") or ""
            grams = (
                item.get("grams")
                or item.get("portion_gram")
                or item.get("amount_g")
                or item.get("g")
                or 0
            )
            group = item.get("group") or item.get("category") or ""
        else:
            name = getattr(item, "name", None) or getattr(item, "food_name", None) or ""
            grams = getattr(item, "grams", None) or getattr(item, "portion_gram", None) or 0
            group = getattr(item, "group", None) or getattr(item, "category", "") or ""

        try:
            grams_i = int(float(grams or 0))
        except Exception:
            grams_i = 0

        out.append({"name": str(name), "grams": grams_i, "group": _norm_group(str(group))})
    return out

def _render_macro_blocks(items_norm):
    grouped = {"carb": [], "protein": [], "fat": [], "vegetable": [], "etc": []}
    for it in items_norm:
        grouped.setdefault(it["group"], []).append(it)

    blocks = [
        ("carb", "🍚 탄", "없음"),
        ("protein", "🥩 단", "없음"),
        ("fat", "🥑 지", "없음"),
        ("vegetable", "🥦 채", "없음"),
    ]

    cols = st.columns(4)
    for idx, (k, title, empty_txt) in enumerate(blocks):
        with cols[idx]:
            st.markdown(f"**{title}**")
            if not grouped.get(k):
                st.caption(empty_txt)
            else:
                items = grouped[k]
                show = items[:2]
                for it in show:
                    st.markdown(f"- {it['name']} {int(it['grams'])}g")
                if len(items) > 2:
                    with st.expander(f"+{len(items)-2} 더보기", expanded=False):
                        for it in items[2:]:
                            st.markdown(f"- {it['name']} {int(it['grams'])}g")

    if grouped.get("etc"):
        with st.expander("🧩 기타", expanded=False):
            for it in grouped["etc"]:
                st.markdown(f"- {it['name']} {int(it['grams'])}g")

def _get_meal_plan_for_day(engine: RoutineEngine, *, user_id_raw: bytes, d: date, profile: dict, goal: dict, event_flags: dict):
    """
    날짜별 meal_plan 생성(캐시 포함)
    - 날짜 + goal + signature 기반 캐시 (주간 7일 동일/꼬임 방지)
    """
    if "weekly_meal_plan_cache" not in st.session_state:
        st.session_state["weekly_meal_plan_cache"] = {}

    macro_key = json.dumps(goal.get("macro_target") or {}, sort_keys=True, ensure_ascii=False)
    event_key = json.dumps(event_flags or {}, sort_keys=True, ensure_ascii=False)

    base_key = (
        f"{user_id_raw.hex()}|{d.isoformat()}|"
        f"{goal.get('goal_type')}|{goal.get('kcal_target')}|"
        f"{macro_key}|{hashlib.md5(event_key.encode()).hexdigest()[:8]}"
    )

    try:
        actions = engine.build_actions(
            user_id=user_id_raw,
            target_date=d,
            event_flags=event_flags,
            state={},
        )
    except Exception:
        actions = {"meal": {}, "exercise": {}, "event_flags": event_flags, "recommendation_signature": "keep"}

    sig = str(actions.get("recommendation_signature") or "keep")
    cache_key = f"{base_key}|{sig}"

    if cache_key in st.session_state["weekly_meal_plan_cache"]:
        return st.session_state["weekly_meal_plan_cache"][cache_key]

    try:
        routine = engine.build_routine(
            user_id=user_id_raw,
            user_profile=profile,
            user_goal=goal,
            actions=actions,
            target_date=d,
        )
        meal_plan = (routine.get("meal_plan") or {}) if isinstance(routine, dict) else {}
    except Exception:
        meal_plan = {}

    st.session_state["weekly_meal_plan_cache"][cache_key] = meal_plan
    return meal_plan

def _kcal_badge(actual_k: float, target_k: float) -> str:
    diff = actual_k - target_k
    if actual_k <= 0:
        return "📝 기록 없음"
    if diff > 200:
        return f"🔶 {diff:+.0f} kcal"
    if diff < -200:
        return f"🔷 {diff:+.0f} kcal"
    return f"✅ {diff:+.0f} kcal"

def _render_event_summary(ev_rows):
    if not ev_rows:
        return "이벤트 없음"
    labels = []
    for et, sev, note in ev_rows:
        if sev:
            labels.append(f"{et}({sev})")
        else:
            labels.append(et)
    return " · ".join(labels)


def _render_event_comment(actions: dict):
    sig = (actions or {}).get("recommendation_signature") or "keep"
    msg = SIGNATURE_MESSAGES.get(sig, {}).get("text")
    if not msg:
        return None
    return msg
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
engine = RoutineEngine()

# ==================================================
# Profile / Goal
# ==================================================
profile = repo.get_user_profile(user_id=user_id_raw) or {}
profile = dict(profile)
profile["user_id"] = user_id_raw

with get_db_conn() as conn:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT kcal_target, macro_target, goal_type
        FROM user_goal
        WHERE user_id = :u AND is_active = 'Y'
        """,
        {"u": user_id_raw},
    )
    grow = cur.fetchone()

kcal_target = 2000
macro_target = {}
goal_type = "maintenance"
if grow:
    kcal_target = int(grow[0] or 2000)
    macro_target = _load_json(grow[1], {}) or {}
    goal_type = grow[2] or "maintenance"

goal = {
    "user_id": user_id_raw,
    "goal_type": goal_type,
    "kcal_target": kcal_target,
    "macro_target": macro_target,
}

# ==================================================
# Week range
# ==================================================
today = date.today()
week_start = today - timedelta(days=today.weekday())
week_days = [week_start + timedelta(days=i) for i in range(7)]
week_end = week_start + timedelta(days=6)

# ==================================================
# UI
# ==================================================
app_container_start()
st.markdown("## 📅 주간 식단표")
muted_text(f"{week_start} ~ {week_end}")
spacer(6)

tabs = st.tabs([f"{DOW_KR[i]}({d.month}/{d.day})" for i, d in enumerate(week_days)])

for i, d in enumerate(week_days):
    with tabs[i]:
        # 🔥 이벤트 변경 시 캐시 무효화
        if st.session_state.get("event_dirty"):
            st.session_state["weekly_meal_plan_cache"] = {}
            st.session_state["event_dirty"] = False

        # 🔥 DB → EventService 기준 단일 event_flags
        event_flags = None  # RoutineEngine이 직접 DB에서 로드
        ev_rows = _get_events(user_id_raw, d)

        meal_plan = _get_meal_plan_for_day(
            engine,
            user_id_raw=user_id_raw,
            d=d,
            profile=profile,
            goal=goal,
            event_flags=event_flags,
        )

        meals = (meal_plan.get("meals") or {}) if isinstance(meal_plan, dict) else {}

        # ✅ [수정 1] planner 기준 목표 kcal (없으면 DB fallback)
        planner_target_kcal = float(
            meal_plan.get("total_kcal")
            or meal_plan.get("target_kcal")
            or kcal_target
        )

        actuals_by_type, totals = _get_actual_meals(user_id_raw, d)
        actual_k = float(totals.get("kcal") or 0.0)

        # =========================
        # 상단 요약 (한 줄 헤더)
        # =========================
        col_left, col_right = st.columns([4, 1.5])

        with col_left:
            diff_kcal = actual_k - planner_target_kcal

            if diff_kcal > 0:
                status_label = "초과"
                status_color = "#e74c3c"  # 빨강
                status_icon = "🔥"
            elif diff_kcal < 0:
                status_label = "남음"
                status_color = "#2ecc71"  # 초록
                status_icon = "🟢"
            else:
                status_label = "달성"
                status_color = "#999999"  # 회색
                status_icon = "⚪"

            st.markdown(
                f"""
                <div style="
                    font-size:16px;
                    font-weight:600;
                    display:flex;
                    align-items:center;
                    gap:12px;
                    flex-wrap:wrap;
                ">
                    🎯 목표 {int(planner_target_kcal)} kcal
                    <span style="color:#bbb;">·</span>
                    🍴 섭취 {int(actual_k)} kcal
                    <span style="color:#bbb;">·</span>
                    <span style="color:{status_color};">
                        {status_icon} {status_label} {abs(int(diff_kcal))} kcal
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_right:
            event_text = _render_event_summary(ev_rows)
            if event_text and event_text != "이벤트 없음":
                st.markdown(
                    f"""
                    <div style="
                        font-size:14px;
                        font-weight:500;
                        color:#333;
                        text-align:right;
                        white-space:nowrap;
                    ">
                        📌 이벤트&nbsp; {event_text}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # 3️⃣ Replan → message_layer (이미 존재하는 로직)
            try:
                orchestrator = ReplanOrchestrator()
                replan_result = orchestrator.run(
                    user_id=user_id_raw,
                    target_date=d,
                    event_flags=event_flags,
                )
                ui_messages = build_replan_messages(replan_result)
            except Exception:
                ui_messages = {}

            # 4️⃣ horizon 카드만 추출
            cards = ui_messages.get("cards", []) if isinstance(ui_messages, dict) else []
            horizon_cards = [c for c in cards if c.get("type") == "horizon"]

            if horizon_cards:
                for c in horizon_cards:
                    st.markdown(
                        f"""
                        <div style="
                            margin-top:8px;
                            padding:10px 14px;
                            background:#eef3ff;
                            border-radius:12px;
                            font-size:13px;
                            color:#333;
                        ">
                            <div style="font-weight:600;">📅 이후 식단 가이드</div>
                            {"".join(
                            [f"<div style='margin-top:4px;'>• {b}</div>"
                             for b in c.get("bullets", [])]
                        )}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        if st.button(
                "🍽️ 오늘 식사 기록",
                key=f"btn_to_today_{d.isoformat()}",  # 🔥 날짜 기반 unique key
        ):
            st.switch_page("pages/record_meal.py")

        spacer(6)

        left, right = st.columns(2, gap="large")

        with left:
            card_start("📋 추천 식단(가이드)", right_text="탄·단·지·채소 기준")
            for mt in MEAL_ORDER:
                data = meals.get(mt, {}) or {}
                label = MEAL_LABEL.get(mt, mt)

                target_k = 0
                items = []

                if isinstance(data, dict):
                    target_k = int(data.get("target_kcal", 0) or 0)
                    items = data.get("items", []) or []
                elif isinstance(data, list):
                    items = data

                st.markdown(f"#### {label}" + (f" · {target_k} kcal" if target_k else ""))

                norm_items = _normalize_plan_items(items)
                if not norm_items:
                    st.caption("추천 항목 없음")
                else:
                    _render_macro_blocks(norm_items)

                spacer(6)

            card_end()

        with right:
            card_start("✅ 실제 섭취(기록)", right_text=f"총 {int(totals['kcal'])} kcal")

            st.markdown(
                f"**탄** {int(totals['carb'])}g · "
                f"**단** {int(totals['protein'])}g · "
                f"**지** {int(totals['fat'])}g"
            )

            has_any = any(len(v) > 0 for v in (actuals_by_type or {}).values())
            if not has_any:
                muted_text("아직 기록이 없습니다.")
            else:
                ordered_types = [t for t in MEAL_ORDER if t in actuals_by_type] + [
                    t for t in actuals_by_type.keys() if t not in MEAL_ORDER
                ]
                for mt in ordered_types:
                    rows = actuals_by_type.get(mt, [])
                    if not rows:
                        continue
                    st.markdown(f"#### {MEAL_LABEL.get(mt, '🧩 기타')}")
                    for r in rows[:6]:
                        st.markdown(
                            f"- **{r.get('name', '')}** {int(r.get('g', 0))}g · "
                            f"{int(r.get('kcal', 0))} kcal  "
                            f"(탄 {int(r.get('carb', 0))}g · 단 {int(r.get('protein', 0))}g · 지 {int(r.get('fat', 0))}g)"
                        )
                    if len(rows) > 6:
                        with st.expander(f"+{len(rows)-6}개 더보기", expanded=False):
                            for r in rows[6:]:
                                st.markdown(
                                    f"- **{r.get('name', '')}** {int(r.get('g', 0))}g · "
                                    f"{int(r.get('kcal', 0))} kcal  "
                                    f"(탄 {int(r.get('carb', 0))}g · 단 {int(r.get('protein', 0))}g · 지 {int(r.get('fat', 0))}g)"
                                )

            spacer(6)
            card_end()

        spacer(8)

app_container_end()
bottom_nav(active="weekly_meal_plan")
