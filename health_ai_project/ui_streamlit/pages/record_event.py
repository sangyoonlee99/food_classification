# ui_streamlit/pages/record_event.py
from __future__ import annotations

import sys
import os
import json
from pathlib import Path
from datetime import date
import streamlit as st

# ==================================================
# Auth
# ==================================================
from ui_streamlit.utils.auth import require_login
require_login()

# ==================================================
# Path
# ==================================================
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ==================================================
# Imports
# ==================================================
from infra.db_server import get_db_conn
from ui_streamlit.components import (
    app_container_start,
    app_container_end,
    card_start,
    card_end,
    section_header,
    muted_text,
    spacer,
    bottom_nav,
)

# ==================================================
# Session Defaults
# ==================================================
ss = st.session_state
ss.setdefault("event_date", date.today())
ss.setdefault("event_type", None)
ss.setdefault("event_severity", "보통")
ss.setdefault("event_note", "")


def _ensure_bytes16(v) -> bytes:
    if isinstance(v, (bytes, bytearray)):
        b = bytes(v)
        if len(b) != 16:
            raise ValueError("user_id must be RAW(16)")
        return b
    if isinstance(v, str):
        b = bytes.fromhex(v.replace("0x", "").strip())
        if len(b) != 16:
            raise ValueError("user_id hex must be 32 chars")
        return b
    raise TypeError("invalid user_id type")


def _fetch_events_for_date(*, user_id_raw: bytes, d: date):
    # created_at 없을 수도 있으니 2단계 fallback
    try:
        with get_db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT event_id, event_type, severity, note
                FROM event_log
                WHERE user_id = :u
                  AND event_date = :d
                ORDER BY created_at DESC
                """,
                {"u": user_id_raw, "d": d},
            )
            return cur.fetchall() or []
    except Exception:
        with get_db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT event_id, event_type, severity, note
                FROM event_log
                WHERE user_id = :u
                  AND event_date = :d
                """,
                {"u": user_id_raw, "d": d},
            )
            return cur.fetchall() or []


def _lob_to_str(x):
    if x is None:
        return None
    if hasattr(x, "read"):
        try:
            return x.read()
        except Exception:
            return str(x)
    return x


# ==================================================
# Auth user
# ==================================================
user_id = ss.get("user_id")
if not user_id:
    st.switch_page("pages/login.py")
    st.stop()

user_id_raw = _ensure_bytes16(user_id)

# ==================================================
# UI
# ==================================================
app_container_start()
st.markdown("### 🧩 이벤트 기록")
spacer(12)

# ==================================================
# 1) 날짜
# ==================================================
section_header("날짜")
card_start()
ss.event_date = st.date_input(
    "이벤트 날짜",
    value=ss.event_date,
    label_visibility="collapsed",
)
card_end()
spacer(10)

# ==================================================
# 2) 이벤트 유형
# ==================================================
section_header("이벤트 유형")
EVENT_TYPES = [
    ("🍻 회식", "회식"),
    ("💼 야근", "야근"),
    ("✈️ 여행", "여행"),
    ("🎉 모임", "모임"),
    ("😴 수면부족", "수면부족"),
]

card_start()
cols = st.columns(len(EVENT_TYPES))
for idx, (icon_label, value) in enumerate(EVENT_TYPES):
    with cols[idx]:
        if st.button(icon_label, key=f"evt_type_{idx}", use_container_width=True):
            ss.event_type = value
card_end()

if ss.event_type:
    muted_text(f"선택됨: {ss.event_type}")
else:
    muted_text("이벤트 유형을 선택해주세요.")
spacer(10)

# ==================================================
# 3) 영향도
# ==================================================
section_header("영향도")
card_start()
ss.event_severity = st.radio(
    "영향도",
    options=["낮음", "보통", "높음"],
    horizontal=True,
    index=["낮음", "보통", "높음"].index(ss.event_severity),
    label_visibility="collapsed",
)
card_end()
spacer(10)

# ==================================================
# 4) 메모
# ==================================================
section_header("메모 (선택)")
card_start()
ss.event_note = st.text_area(
    "메모",
    value=ss.event_note,
    placeholder="예: 회식으로 늦게 귀가",
    label_visibility="collapsed",
)
card_end()
spacer(14)

# ==================================================
# 5) 저장
# ==================================================
if st.button("저장하기", use_container_width=False):
    if not ss.event_type:
        st.warning("이벤트 유형을 선택해주세요.")
        st.stop()

    try:
        raw_flags = {
            "event_type": ss.event_type,
            "severity": ss.event_severity,
            "has_note": bool(ss.event_note),
        }

        with get_db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO event_log (
                    event_id,
                    user_id,
                    event_date,
                    event_type,
                    severity,
                    note,
                    raw_flags
                )
                VALUES (
                    :event_id,
                    :user_id,
                    :event_date,
                    :event_type,
                    :severity,
                    :note,
                    :raw_flags
                )
                """,
                {
                    "event_id": os.urandom(16),
                    "user_id": user_id_raw,
                    "event_date": ss.event_date,
                    "event_type": ss.event_type,
                    "severity": ss.event_severity,
                    "note": (ss.event_note or "").strip(),
                    "raw_flags": json.dumps(raw_flags, ensure_ascii=False),
                },
            )
            conn.commit()

        # 🔥 재계산 트리거
        ss.event_dirty = True
        ss.recommendation_result = None
        ss.last_recommendation_date = None

        st.success("이벤트가 저장되었습니다.")

        # ✅ 페이지 유지 + 입력값 일부 초기화
        ss.event_type = None
        ss.event_note = ""

        st.rerun()

    except Exception as e:
        st.error(f"저장 실패: {e}")

spacer(12)

# ==================================================
# 6) (선택날짜) 이벤트 목록 + 삭제
# ==================================================
section_header("기록된 이벤트")
rows = _fetch_events_for_date(user_id_raw=user_id_raw, d=ss.event_date)

card_start()
if not rows:
    muted_text("해당 날짜에 기록된 이벤트가 없습니다.")
else:
    for i, (event_id, event_type, severity, note) in enumerate(rows):
        note_s = _lob_to_str(note)
        left, right = st.columns([8, 2])
        with left:
            st.markdown(f"- **{event_type}** · 영향도 `{severity}`")
            if note_s:
                muted_text(str(note_s))
        with right:
            if st.button("삭제", key=f"del_evt_{i}", use_container_width=True):
                try:
                    with get_db_conn() as conn:
                        cur = conn.cursor()
                        cur.execute(
                            """
                            DELETE FROM event_log
                            WHERE event_id = :eid
                              AND user_id = :u
                            """,
                            {"eid": event_id, "u": user_id_raw},
                        )
                        conn.commit()

                    ss.event_dirty = True
                    ss.recommendation_result = None
                    ss.last_recommendation_date = None

                    st.success("삭제되었습니다.")
                    st.rerun()
                except Exception as e:
                    st.error(f"삭제 실패: {e}")
card_end()

spacer(16)
app_container_end()

# ==================================================
# Bottom Nav (루틴에서 들어왔으면 루틴 유지)
# ==================================================
active_tab = "routine" if ss.get("nav_from") == "routine" else "record"
bottom_nav(active=active_tab)
