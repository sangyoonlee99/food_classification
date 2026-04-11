# ui_streamlit/theme.py
import streamlit as st

def apply_theme():
    st.markdown("""
    <style>
    /* 전체 배경 */
    html, body, [class*="css"]  {
        background-color: #f4f4f4;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto;
    }

    /* 중앙 모바일 프레임 */
    .app-container {
        max-width: 420px;
        margin: 0 auto;
        padding: 12px;
    }

    /* 카드 */
    .card {
        background: #ffffff;
        border-radius: 20px;
        padding: 16px;
        margin-bottom: 16px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }

    .card-title {
        font-weight: 700;
        font-size: 16px;
        margin-bottom: 12px;
    }

    /* 큰 버튼 */
    .primary-btn button {
        width: 100%;
        height: 52px;
        border-radius: 26px;
        font-size: 16px;
        font-weight: 600;
    }

    /* 하단 여백 */
    .bottom-space {
        height: 80px;
    }
    </style>
    """, unsafe_allow_html=True)
