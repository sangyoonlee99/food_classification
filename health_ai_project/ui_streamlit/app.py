# ui_streamlit/app.py
from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st

# ==================================================
# Project Root Path
# ==================================================
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ==================================================
# Streamlit Config
# ==================================================
st.set_page_config(
    page_title="Health AI",
    layout="centered",
)

# ==================================================
# 🚪 Entry Router (최초 1회만)
# ==================================================
if "app_routed" not in st.session_state:
    st.session_state["app_routed"] = True

    if st.session_state.get("is_logged_in") and st.session_state.get("user_id"):
        st.switch_page("pages/home.py")
    else:
        st.switch_page("pages/login.py")
