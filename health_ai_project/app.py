# app.py
import streamlit as st

st.set_page_config(page_title="Health AI", layout="centered")

# =================================================
# MOCK DATA (STEP 6-6에서는 구조만 고정)
# =================================================
daily_output = {
    "goal": "체중 감량 목표를 유지합니다. 회식 이후 회복에 집중하는 하루입니다.",
    "action": "수분 섭취를 충분히 하고 가벼운 활동 위주로 하루를 마무리하세요.",
    "next_meal": "다음 끼니에서는 단백질을 먼저 보충하세요.",
}

weekly_output = {
    "score": 55,
    "grade": "주의",
    "feedback": [
        "단백질 섭취가 부족합니다. 생선, 계란, 두부를 추가해 보세요.",
        "하루 식사량이 충분한지 점검해 보세요.",
        "좋은 지방 섭취도 필요합니다.",
    ],
    "nutrient_boost": [
        "단백질 섭취가 부족합니다. 닭가슴살 100g을 추가해 보세요."
    ],
}

# =================================================
# UI START
# =================================================
st.title("🍽️ 오늘의 건강 가이드")

# -------------------------------
# DAILY CARDS
# -------------------------------
st.subheader("📌 오늘의 요약")

st.markdown("### 🎯 오늘의 목표")
st.info(daily_output["goal"])

st.markdown("### 🔑 오늘 핵심 행동")
st.success(daily_output["action"])

st.markdown("### 🍴 다음 끼니 추천")
st.warning(daily_output["next_meal"])

st.divider()

# -------------------------------
# WEEKLY CARDS
# -------------------------------
st.subheader("📊 이번 주 요약")

col1, col2 = st.columns(2)
with col1:
    st.metric("주간 점수", weekly_output["score"])
with col2:
    st.metric("주간 등급", weekly_output["grade"])

st.markdown("### 📝 주간 피드백")
for msg in weekly_output["feedback"]:
    st.write(f"- {msg}")

st.markdown("### 💡 부족 영양소 보완")
for msg in weekly_output["nutrient_boost"]:
    st.write(f"- {msg}")
