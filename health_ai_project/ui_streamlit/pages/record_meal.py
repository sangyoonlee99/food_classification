# ui_streamlit/pages/record_meal.py
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, date
import json
import tempfile
from services.meal_service import MealService, get_default_serving_g
from services.meal_service import _lookup_excel  # 내부 엑셀 검색 함수
import streamlit as st
import hashlib
from services.recommendation_layer import SIGNATURE_MESSAGES
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui_streamlit.components import (
    app_container_start,
    app_container_end,
    card_start,
    card_end,
    spacer,
    primary_button,
    muted_text,    bottom_nav,
)


from infra.db_server import get_db_conn
from infra.repositories.user_repository import UserRepository
from routine.routine_engine import RoutineEngine

from services.food_ai_service import FoodAIService

ROUND_GRAM_UNIT = 5

MEAL_TIME_MAP = {
    "breakfast": "08:00",
    "lunch": "13:00",
    "dinner": "19:00",
    "snack": "15:00",
}
MEAL_LABEL = {
    "breakfast": "🍳 아침",
    "lunch": "🍱 점심",
    "dinner": "🍽️ 저녁",
    "snack": "🍩 간식",
    "unknown": "🧩 기타",
}
MEAL_ORDER = ["breakfast", "lunch", "dinner", "snack"]

# =========================
# AI UI policy (recommended)
# =========================
AI_TOP_SHOW = 2      # UI에서 기본 노출
AI_TOP_MAX = 6       # 분석은 최대 6개까지 받아두기
CONF_WARN = 0.50     # confidence 낮을 때 안내


# ==================================================
# Utils
# ==================================================
def _load_json(val):
    if val is None:
        return {}
    if hasattr(val, "read"):
        val = val.read()
    try:
        return json.loads(val) if val else {}
    except Exception:
        return {}

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

def _infer_meal_type(eaten_at: datetime | date | None) -> str:
    if eaten_at is None:
        return "dinner"
    if isinstance(eaten_at, date) and not isinstance(eaten_at, datetime):
        return "dinner"
    hh = int(getattr(eaten_at, "hour", 0))
    if 5 <= hh <= 10:
        return "breakfast"
    if 11 <= hh <= 15:
        return "lunch"
    if 16 <= hh <= 18:
        return "snack"
    return "dinner"

def _round_g(v: float | int) -> int:
    try:
        x = float(v or 0)
    except Exception:
        return 0
    unit = max(1, int(ROUND_GRAM_UNIT or 1))
    return int(round(x / unit) * unit)

def _get_events(user_id_raw: bytes, d: date):
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT event_type, severity, note
            FROM event_log
            WHERE user_id = :u
              AND event_date = :d
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

def _norm_group(x: str) -> str:
    s = (x or "").strip().lower()
    if any(k in s for k in ["carb", "탄수", "탄"]):
        return "carb"
    if any(k in s for k in ["protein", "단백", "단"]):
        return "protein"
    if any(k in s for k in ["fat", "지방", "지"]):
        return "fat"
    if any(k in s for k in ["vegetable", "veg", "veggie", "채소", "야채"]):
        return "vegetable"
    return "etc"

def _normalize_plan_items(items):
    out = []
    for item in items or []:
        if isinstance(item, dict):
            name = item.get("name") or item.get("food_name") or item.get("food") or ""
            grams = int(item.get("grams") or item.get("portion_gram") or item.get("amount_g") or 0)
            group = item.get("group") or item.get("category") or ""
        else:
            name = getattr(item, "name", None) or getattr(item, "food_name", None) or getattr(item, "food", "") or ""
            grams = int(getattr(item, "grams", None) or getattr(item, "portion_gram", None) or getattr(item, "amount_g", 0) or 0)
            group = getattr(item, "group", None) or getattr(item, "category", "") or ""
        out.append({"name": str(name), "grams": int(grams), "group": _norm_group(str(group))})
    return out

def _render_grouped_compact(items_norm):
    grouped = {"carb": [], "protein": [], "fat": [], "vegetable": [], "etc": []}
    for it in items_norm:
        grouped.setdefault(it["group"], []).append(it)

    cols = st.columns(4)
    blocks = [
        ("carb", "🍚 탄", cols[0]),
        ("protein", "🥩 단", cols[1]),
        ("fat", "🥑 지", cols[2]),
        ("vegetable", "🥦 채", cols[3]),
    ]

    for key, title, col in blocks:
        with col:
            st.markdown(f"**{title}**")
            if not grouped.get(key):
                st.caption("없음")
            else:
                for it in grouped[key]:
                    g = _round_g(it["grams"])
                    st.markdown(f"- {it['name']}  \n  <span style='color:#666'>{g}g</span>", unsafe_allow_html=True)

    if grouped.get("etc"):
        st.caption("기타")
        etc_names = [f"{x['name']}({_round_g(x['grams'])}g)" for x in grouped["etc"]]
        st.write(" · ".join(etc_names))

def _load_today_meals(user_id_raw: bytes, d: date):
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                meal_id,
                eaten_at,
                NVL(meal_type, 'unknown') AS meal_type,
                food_name,
                NVL(amount_g, 0) AS amount_g,
                NVL(kcal, 0) AS kcal,
                NVL(carb_g, 0) AS carb_g,
                NVL(protein_g, 0) AS protein_g,
                NVL(fat_g, 0) AS fat_g
            FROM meal_record
            WHERE user_id = :p_user_id
              AND TRUNC(eaten_at) = TRUNC(:p_d)
            ORDER BY eaten_at
            """,
            {"p_user_id": user_id_raw, "p_d": d},
        )
        return cur.fetchall() or []

def _build_reco_meal_plan(engine: RoutineEngine, *, user_id_raw: bytes, d: date, profile: dict, goal: dict, event_flags: dict):
    if "meal_plan_cache" not in st.session_state:
        st.session_state["meal_plan_cache"] = {}

    macro_key = json.dumps(goal.get("macro_target") or {}, sort_keys=True, ensure_ascii=False)
    event_key = json.dumps(event_flags or {}, sort_keys=True, ensure_ascii=False)

    base_key = (
        f"{user_id_raw.hex()}|{d.isoformat()}|"
        f"{goal.get('goal_type')}|{goal.get('kcal_target')}|"
        f"{macro_key}|{hashlib.md5(event_key.encode()).hexdigest()[:8]}"
    )

    actions_error = None
    try:
        actions = engine.build_actions(
            user_id=user_id_raw,
            target_date=d,
            event_flags=event_flags,
            state={},
        )
    except Exception as e:
        actions_error = e
        actions = {"meal": {}, "exercise": {}, "event_flags": event_flags, "recommendation_signature": "keep"}

    sig = str(actions.get("recommendation_signature") or "keep")
    cache_key = f"{base_key}|{sig}"

    if cache_key in st.session_state["meal_plan_cache"]:
        return st.session_state["meal_plan_cache"][cache_key], actions_error

    try:
        routine = engine.build_routine(
            user_id=user_id_raw,
            user_profile=profile,
            user_goal=goal,
            actions=actions,
            target_date=d,
        )
        meal_plan = (routine.get("meal_plan") or {}) if isinstance(routine, dict) else {}
    except Exception as e:
        meal_plan = {}
        if actions_error is None:
            actions_error = e

    st.session_state["meal_plan_cache"][cache_key] = meal_plan
    return {
        "meal_plan": meal_plan,
        "actions": actions,
    }, actions_error


def _normalize_meal_type(mt: str) -> str:
    m = (mt or "").strip().lower()
    if m in ["breakfast", "아침"]:
        return "breakfast"
    if m in ["lunch", "점심"]:
        return "lunch"
    if m in ["dinner", "저녁"]:
        return "dinner"
    if m in ["snack", "간식"]:
        return "snack"
    return "unknown"

def _meal_id_hex(meal_id_raw) -> str:
    try:
        return bytes(meal_id_raw).hex()
    except Exception:
        return str(meal_id_raw)

def _save_upload_to_temp(uploaded_file) -> Path:
    """
    Streamlit UploadedFile -> OS temp file
    (Windows/리눅스 모두 안전)
    """
    suffix = ""
    name = getattr(uploaded_file, "name", "") or ""
    if "." in name:
        suffix = "." + name.split(".")[-1].lower()

    fd, tmp = tempfile.mkstemp(prefix="food_", suffix=suffix)
    p = Path(tmp)
    try:
        with open(fd, "wb") as f:
            f.write(uploaded_file.read())
    except Exception:
        # fallback
        p.write_bytes(uploaded_file.getvalue())
    return p


def _filter_candidates_by_excel(candidates: list[dict]) -> list[dict]:
    """
    AI 후보 중 food_nutrition.xlsx 에 실제 존재하는 음식만 유지
    (DB ❌ / 엑셀 ✅)
    """
    if not candidates:
        return []

    valid = []
    for c in candidates:
        name = (c.get("food_name") or "").strip()
        if not name:
            continue

        row = _lookup_excel(name)
        if row:
            valid.append(c)

    return valid

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

# =========================
# FoodAI multi-select safe keys + reset (IMPORTANT)
# =========================
def _safe_key(name: str) -> str:
    h = hashlib.md5((name or "").encode("utf-8")).hexdigest()
    return h[:10]

def _request_food_ai_reset():
    # "다음 rerun에서" 초기화하도록 플래그만 세팅
    st.session_state["_food_ai_reset_pending"] = True

def _apply_food_ai_reset_if_pending():
    if not st.session_state.get("_food_ai_reset_pending"):
        return

    # 위젯 생성되기 "전"에 pop/del로 제거 (대입 금지!)
    for k in [
        "food_candidates",
        "food_ai_done",
        "food_ai_selected_name",
        "food_ai_choose_more",
        "food_ai_multi_selected",
        "food_ai_multi_grams",
        "food_ai_meal_type",
        "food_image_upload",
    ]:
        st.session_state.pop(k, None)

    # food_ai_g_* number_input 키들도 제거
    for k in list(st.session_state.keys()):
        if str(k).startswith("food_ai_g_"):
            st.session_state.pop(k, None)

    st.session_state["_food_ai_reset_pending"] = False


# ==================================================
# FoodAI cached loader (VERY IMPORTANT)
# ==================================================
@st.cache_resource(show_spinner=False)
def _get_food_ai() -> FoodAIService:
    return FoodAIService()


# ==================================================
# Auth
# ==================================================
user_id = st.session_state.get("user_id")
if not user_id:
    st.switch_page("pages/login.py")
    st.stop()

# ==================================================
# 🔥 record_meal은 '오늘 화면' → 항상 캐시 초기화
# ==================================================
if "meal_plan_cache" in st.session_state:
    st.session_state["meal_plan_cache"] = {}


user_id_raw = _ensure_bytes16(user_id)

today = date.today()
meal_svc = MealService()
repo = UserRepository()
engine = RoutineEngine()

# ==================================================
# Load Profile / Goal
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
        WHERE user_id = :p_user_id
          AND is_active = 'Y'
        """,
        {"p_user_id": user_id_raw},
    )
    row = cur.fetchone()

kcal_target = 2000
macro_target = {}
goal_type = "maintenance"
if row:
    kcal_target = int(row[0]) if row[0] is not None else 2000
    macro_target = _load_json(row[1]) or {}
    goal_type = row[2] or "maintenance"

# 🔒 fallback (기존 유저 보호)
if not macro_target:
    from services.user_goal_service import build_macro_target_from_kcal
    macro_target = build_macro_target_from_kcal(kcal_target)


goal = {
    "user_id": user_id_raw,
    "goal_type": goal_type,
    "kcal_target": kcal_target,
    "macro_target": macro_target,
}


# ==================================================
# Events -> flags
# ==================================================
ev_rows = _get_events(user_id_raw, today)
event_flags = _event_flags_from_rows(ev_rows)

# 🔥 이벤트 변경 시 오늘 식사 기록 캐시 무효화 (주간 식단표와 동일)
if st.session_state.get("event_dirty"):
    st.session_state["meal_plan_cache"] = {}
    st.session_state["event_dirty"] = False

# ==================================================
# Build Recommended Meal Plan (🔥 단 1회)
# ==================================================
reco_result, actions_error = _build_reco_meal_plan(
    engine,
    user_id_raw=user_id_raw,
    d=today,
    profile=profile,
    goal=goal,
    event_flags=event_flags,
)

meal_plan = reco_result.get("meal_plan", {})
actions = reco_result.get("actions", {})


meals = (meal_plan.get("meals") or {}) if isinstance(meal_plan, dict) else {}

# ✅ 오늘 목표 kcal = 엔진이 만든 meal_plan 기준
effective_kcal_target = int(
    meal_plan.get("total_kcal")   # ← 🔥 이걸 1순위로
    or meal_plan.get("target_kcal")
    or kcal_target
)




# ==================================================
# Aggregate Today (실제 섭취)
# ==================================================
with get_db_conn() as conn:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            NVL(SUM(kcal), 0),
            NVL(SUM(carb_g), 0),
            NVL(SUM(protein_g), 0),
            NVL(SUM(fat_g), 0)
        FROM meal_record
        WHERE user_id = :u
          AND TRUNC(eaten_at) = TRUNC(:d)
        """,
        {"u": user_id_raw, "d": today},
    )
    agg = cur.fetchone() or (0, 0, 0, 0)

eaten_kcal = int(float(agg[0] or 0))
remain_kcal = int(max(0, effective_kcal_target - eaten_kcal))



# ==================================================
# Load Today Records
# ==================================================
today_rows = _load_today_meals(user_id_raw, today)

by_type = {"breakfast": [], "lunch": [], "dinner": [], "snack": [], "unknown": []}
totals = {"kcal": 0.0, "carb": 0.0, "protein": 0.0, "fat": 0.0}

for meal_id, eaten_at, meal_type, name, g, kcal, carb, protein, fat in today_rows:
    mt = _normalize_meal_type(meal_type) or _infer_meal_type(eaten_at)
    if mt not in by_type:
        mt = "unknown"
    by_type[mt].append(
        (meal_id, name, float(g or 0), float(kcal or 0), float(carb or 0), float(protein or 0), float(fat or 0), eaten_at)
    )
    totals["kcal"] += float(kcal or 0)
    totals["carb"] += float(carb or 0)
    totals["protein"] += float(protein or 0)
    totals["fat"] += float(fat or 0)


# ==================================================
# UI
# ==================================================
app_container_start()
st.markdown("## 🍽️ 오늘 식사 기록")
muted_text("추천 식단(좌)과 실제 섭취(우)를 비교하며 기록하세요.")
spacer(10)

# =========================
# 상단 요약 (한 줄 헤더)
# =========================
col_left, col_right = st.columns([4, 1.5])


diff_kcal = eaten_kcal - effective_kcal_target

if diff_kcal > 0:
    status_label = "초과"
    status_color = "#e74c3c"   # 빨강
    status_icon = "🔥"
elif diff_kcal < 0:
    status_label = "남음"
    status_color = "#2ecc71"   # 초록
    status_icon = "🟢"
else:
    status_label = "달성"
    status_color = "#999999"   # 회색
    status_icon = "⚪"

with col_left:
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
            🎯 목표 {effective_kcal_target} kcal
            <span style="color:#bbb;">·</span>
            🍴 섭취 {eaten_kcal} kcal
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

spacer(6)

if st.button("📅 주간 식단표 열기", key="btn_open_weekly_plan"):
    st.switch_page("pages/weekly_meal_plan.py")

spacer(8)


# ==================================================
# MAIN LAYOUT (기존 유지)
# ==================================================
left, right = st.columns(2)

# -----------------------------
# LEFT: 추천 식단
# -----------------------------
with left:
    card_start("📋 추천 식단(가이드)", right_text="탄·단·지·채소 기준")
    st.markdown(f"**총 목표 열량**: `{int(kcal_target)} kcal`")

    if ev_rows:
        for et, sev, note in ev_rows:
            st.caption(f"📌 이벤트: {et} · {sev}" + (f" · {note}" if note else ""))
    else:
        st.caption("오늘 등록된 이벤트 없음")

    if actions_error is not None:
        with st.expander("⚠️ (디버그) 추천 생성 오류", expanded=False):
            st.exception(actions_error)

    spacer(6)

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

        st.markdown(f"### {label}" + (f" · `{target_k} kcal`" if target_k else ""))

        norm_items = _normalize_plan_items(items)
        if not norm_items:
            st.caption("추천 항목이 없습니다(템플릿/필터 조건 확인 필요).")
        else:
            _render_grouped_compact(norm_items)

        spacer(10)

    card_end()

# -----------------------------
# RIGHT: 실제 섭취(식사별)
# -----------------------------
with right:
    card_start("✅ 실제 섭취(기록)", right_text="식사별 확인 가능")
    st.markdown(
        f"- 총 `{int(totals['kcal'])} kcal`  |  "
        f"탄 `{int(totals['carb'])}g` · 단 `{int(totals['protein'])}g` · 지 `{int(totals['fat'])}g`"
    )

    any_row = any(len(v) > 0 for v in by_type.values())
    if not any_row:
        muted_text("아직 기록이 없습니다.")
    else:
        for mt in MEAL_ORDER:
            records = by_type.get(mt, [])
            if not records:
                continue

            st.markdown(f"### {MEAL_LABEL.get(mt)}")

            for meal_id, name, g, kcal, carb, protein, fat, eaten_at in records:
                st.markdown(
                    f"- **{name}** {int(g)}g · `{int(kcal)} kcal`\n"
                    f"  <span style='color:#666'>"
                    f"탄 {int(carb)}g · 단 {int(protein)}g · 지 {int(fat)}g"
                    f"</span>",
                    unsafe_allow_html=True
                )

    card_end()

spacer(14)

# ==================================================
# 📸 사진으로 자동 기록 (FINAL) - MULTI SELECT
# ==================================================
card_start("📸 사진으로 음식 기록", right_text="AI 분석 → 후보 선택 → 자동 기록")
st.caption("사진을 올리면 AI가 비슷한 음식을 추천합니다. 여러 개를 선택하고 g를 조절한 뒤 한 번에 기록하세요.")

# ✅ reset pending 적용 (위젯 생성 전에 반드시!)
_apply_food_ai_reset_if_pending()

# session init (대입 OK: 위젯 생성 전)
if "food_candidates" not in st.session_state:
    st.session_state["food_candidates"] = []
if "food_ai_done" not in st.session_state:
    st.session_state["food_ai_done"] = False
if "food_ai_selected_name" not in st.session_state:
    st.session_state["food_ai_selected_name"] = None
if "food_ai_choose_more" not in st.session_state:
    st.session_state["food_ai_choose_more"] = False
if "food_ai_multi_grams" not in st.session_state:
    st.session_state["food_ai_multi_grams"] = {}

img = st.file_uploader(
    "음식 사진 업로드",
    type=["jpg", "jpeg", "png"],
    key="food_image_upload",
)

btn_cols = st.columns([2, 2, 6])
with btn_cols[0]:
    analyze_clicked = st.button(
        "🔍 사진 분석",
        use_container_width=True,
        disabled=st.session_state.get("food_ai_done", False),
    )
with btn_cols[1]:
    reset_clicked = st.button("🧹 초기화", use_container_width=True)

# ✅ 초기화: 절대 session_state에 직접 대입하지 말고 "요청 → rerun"
if reset_clicked:
    _request_food_ai_reset()
    st.rerun()

# ✅ 분석 실행
if img and analyze_clicked:
    tmp_path = _save_upload_to_temp(img)

    with st.spinner("AI가 음식을 분석중입니다..."):
        try:
            food_ai = _get_food_ai()
            # 🔥 후보 더 보여주려면 top_k를 늘리세요 (원하면 8~12 추천)
            raw_candidates = food_ai.analyze(Path(tmp_path), top_k=AI_TOP_MAX)
        except Exception as e:
            st.error("사진 분석 중 오류가 발생했습니다.")
            st.exception(e)
            raw_candidates = []

    # 1️⃣ 표준화
    norm = []
    for c in raw_candidates or []:
        if isinstance(c, dict):
            name = (c.get("food_name") or "").strip()
            score = float(c.get("score", 0.0) or 0.0)
        elif isinstance(c, (list, tuple)) and len(c) >= 2:
            name = str(c[0]).strip()
            try:
                score = float(c[1])
            except Exception:
                score = 0.0
        else:
            continue
        if name:
            norm.append({"food_name": name, "score": score})

    # 2️⃣ 점수 정렬
    norm.sort(key=lambda x: x["score"], reverse=True)

    # 3️⃣ 엑셀 기준 필터링
    norm = _filter_candidates_by_excel(norm)

    st.session_state["food_candidates"] = norm
    st.session_state["food_ai_done"] = True
    st.session_state["food_ai_selected_name"] = norm[0]["food_name"] if norm else None

    st.rerun()

# =========================
# AI 후보 표시 (다중 선택)
# =========================
if st.session_state.get("food_ai_done"):
    cands = st.session_state.get("food_candidates") or []

    if not cands:
        st.warning("AI가 음식 후보를 찾지 못했습니다. 다른 사진을 시도해 주세요.")
    else:
        # 1️⃣ 식사 종류
        meal_type_ai = st.selectbox(
            "식사 종류",
            options=["breakfast", "lunch", "dinner", "snack"],
            index=["breakfast", "lunch", "dinner", "snack"].index(_infer_meal_type(datetime.now())),
            format_func=lambda x: MEAL_LABEL.get(x, x),
            key="food_ai_meal_type",
        )

        # 2️⃣ 후보 목록(표시 개수)
        show = cands[:AI_TOP_MAX]
        food_names = [c["food_name"] for c in show]
        # 점수 제거 위해
        #food_scores = {c["food_name"]: int(float(c.get("score", 0)) * 100) for c in show}

        # ✅ 기본 선택 자동으로 넣지 않음 (default=[])
        selected_foods = st.multiselect(
            "가장 가까운 음식들을 선택하세요 (복수 선택 가능)",
            options=food_names,
            default=[],
            format_func=lambda x: x,    # f"{x} ({food_scores.get(x, 0)}%)",  점수제거위해 주석처리
            key="food_ai_multi_selected",
        )

        # 3️⃣ 선택한 음식에 대해서만 섭취량 입력 노출
        if selected_foods:
            st.markdown("### 🍽️ 음식별 섭취량")

            for fname in selected_foods:
                sk = _safe_key(fname)
                g_key = f"food_ai_g_{sk}"

                if fname not in st.session_state["food_ai_multi_grams"]:
                    st.session_state["food_ai_multi_grams"][fname] = int(get_default_serving_g(fname) or 100)

                st.number_input(
                    f"{fname} 섭취량 (g)",
                    min_value=0,
                    step=10,
                    key=g_key,
                    value=int(st.session_state["food_ai_multi_grams"][fname]),
                )

                st.session_state["food_ai_multi_grams"][fname] = int(st.session_state.get(g_key, 0))

        # 4️⃣ 다중 기록
        if primary_button("✅ 선택한 음식 모두 기록", key="btn_ai_record_multi"):
            if not selected_foods:
                st.warning("기록할 음식을 하나 이상 선택하세요.")
            else:
                saved = 0
                now_dt = datetime.now()

                for fname in selected_foods:
                    grams = int(st.session_state["food_ai_multi_grams"].get(fname, 0) or 0)
                    if grams <= 0:
                        continue

                    meal_svc.record_meal(
                        user_id=user_id_raw,
                        food_name=fname,
                        amount_g=float(grams),
                        eaten_at=now_dt,
                        source=meal_type_ai,
                    )
                    saved += 1

                if saved > 0:
                    st.success(f"📸 선택한 음식 {saved}개 기록 완료")
                    # ✅ 여기서도 대입 초기화 금지 → reset 요청 후 rerun
                    _request_food_ai_reset()
                    st.rerun()
                else:
                    st.warning("섭취량이 0g인 음식만 선택되어 기록되지 않았습니다.")

card_end()
spacer(12)



# ==================================================
# ✍️ 식사별 입력 + 리스트 + 수정/삭제 (기존 유지)
# ==================================================
if "edit_meal_id_hex" not in st.session_state:
    st.session_state["edit_meal_id_hex"] = None

for mt in MEAL_ORDER:
    label = MEAL_LABEL[mt]
    card_start(f"✍️ {label}", right_text="입력 + 기록리스트")

    col_a, col_b = st.columns([7, 3])
    with col_a:
        food_name = st.text_input(
            f"{label} 음식명",
            placeholder="예: 닭가슴살",
            key=f"food_{mt}",
        )
    with col_b:
        grams = st.number_input(
            f"{label} 섭취량(g)",
            min_value=0,
            step=10,
            value=100,
            key=f"g_{mt}",
        )

    if primary_button("✅ 이 식사 기록", key=f"btn_add_{mt}"):
        if not food_name.strip():
            st.warning("음식명을 입력해주세요.")
        elif grams <= 0:
            st.warning("섭취량은 0보다 커야 합니다.")
        else:
            eaten_at = datetime.now()
            meal_svc.record_meal(
                user_id=user_id_raw,
                food_name=food_name.strip(),
                amount_g=float(grams),
                eaten_at=eaten_at,
                source=mt,
            )
            st.success(f"{label} 기록 완료")
            st.rerun()

    spacer(8)
    st.markdown("**기록된 음식**")

    records = by_type.get(mt, [])
    if not records:
        muted_text("아직 기록이 없습니다.")
        card_end()
        spacer(10)
        continue

    for meal_id_raw, name, g, kcal, carb, protein, fat, eaten_at in records:
        mid_hex = _meal_id_hex(meal_id_raw)
        row = st.columns([7, 2, 2])

        with row[0]:
            st.markdown(
                f"- **{name}** {int(g)}g | `{int(kcal)} kcal` "
                f"· 탄{int(carb)}g · 단{int(protein)}g · 지{int(fat)}g"
            )

        with row[1]:
            if st.button("✏️ 수정", key=f"edit_{mt}_{mid_hex}"):
                st.session_state["edit_meal_id_hex"] = mid_hex

        with row[2]:
            if st.button("🗑️ 삭제", key=f"del_{mt}_{mid_hex}"):
                meal_svc.delete_meal(user_id=user_id_raw, meal_id_raw=bytes(meal_id_raw))
                st.success("삭제 완료")
                st.rerun()

        if st.session_state.get("edit_meal_id_hex") == mid_hex:
            with st.expander("수정하기", expanded=True):
                new_mt = st.selectbox(
                    "식사 이동",
                    options=["breakfast", "lunch", "dinner", "snack"],
                    index=["breakfast", "lunch", "dinner", "snack"].index(mt) if mt in ["breakfast", "lunch", "dinner", "snack"] else 2,
                    format_func=lambda x: MEAL_LABEL.get(x, x),
                    key=f"edit_mt_{mid_hex}",
                )
                new_name = st.text_input("음식명", value=str(name), key=f"edit_name_{mid_hex}")
                new_g = st.number_input("섭취량(g)", min_value=0, step=10, value=int(g), key=f"edit_g_{mid_hex}")

                cols2 = st.columns(2)
                with cols2[0]:
                    if primary_button("💾 저장", key=f"save_{mid_hex}"):
                        meal_svc.update_meal(
                            user_id=user_id_raw,
                            meal_id_raw=bytes(meal_id_raw),
                            new_meal_type=new_mt,
                            new_food_name=new_name.strip(),
                            new_amount_g=float(new_g),
                            eaten_at=None,
                        )
                        st.session_state["edit_meal_id_hex"] = None
                        st.success("수정 완료")
                        st.rerun()
                with cols2[1]:
                    if st.button("닫기", key=f"close_{mid_hex}"):
                        st.session_state["edit_meal_id_hex"] = None
                        st.rerun()

    card_end()
    spacer(12)

app_container_end()
bottom_nav(active="record_meal")
