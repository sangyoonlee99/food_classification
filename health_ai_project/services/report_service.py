# services/report_service.py
from __future__ import annotations

from typing import Dict, List

from services.record_service import get_meal_logs
from services.event_service import load_events_for_period, has_protective_event


# =====================================================
# STEP 1-1: Mock 주간 로그 데이터 (fallback)
# =====================================================
def _load_weekly_meal_log(user_id: int) -> List[Dict]:
    """
    mock: 최근 7일 식단 로그
    (record 데이터가 없을 때만 사용)
    """
    return [
        {"carb": 280, "protein": 90, "fat": 60, "kcal": 2100},
        {"carb": 250, "protein": 85, "fat": 55, "kcal": 1950},
        {"carb": 300, "protein": 80, "fat": 70, "kcal": 2200},
        {"carb": 260, "protein": 75, "fat": 60, "kcal": 2000},
        {"carb": 240, "protein": 70, "fat": 58, "kcal": 1850},
        {"carb": 290, "protein": 88, "fat": 65, "kcal": 2150},
        {"carb": 270, "protein": 82, "fat": 62, "kcal": 2050},
    ]


# =====================================================
# STEP 1-2: 영양소 비율 계산
# =====================================================
def _calculate_macro_ratio(meal_logs: List[Dict]) -> Dict[str, float]:
    total_carb = sum(d["carb"] for d in meal_logs)
    total_protein = sum(d["protein"] for d in meal_logs)
    total_fat = sum(d["fat"] for d in meal_logs)

    total = total_carb + total_protein + total_fat
    if total == 0:
        return {"carb": 0.0, "protein": 0.0, "fat": 0.0}

    return {
        "carb": round(total_carb / total, 2),
        "protein": round(total_protein / total, 2),
        "fat": round(total_fat / total, 2),
    }


# =====================================================
# STEP 2: 규칙 기반 주간 점수 계산
# =====================================================
def _calculate_weekly_score(
    meal_logs: List[Dict],
    macro_ratio: Dict[str, float],
) -> int:
    """
    총점 100점
    - 열량 40
    - 단백질 30
    - 매크로 균형 20
    - 일관성 10
    """

    score = 0

    # 1️⃣ 평균 칼로리 (40)
    avg_kcal = sum(d["kcal"] for d in meal_logs) / len(meal_logs)

    if 1800 <= avg_kcal <= 2300:
        score += 40
    elif 1600 <= avg_kcal < 1800 or 2300 < avg_kcal <= 2500:
        score += 25
    else:
        score += 10

    # 2️⃣ 단백질 비율 (30)
    protein_ratio = macro_ratio["protein"]

    if 0.25 <= protein_ratio <= 0.35:
        score += 30
    elif 0.20 <= protein_ratio < 0.25:
        score += 20
    elif protein_ratio >= 0.35:
        score += 25
    else:
        score += 10

    # 3️⃣ 매크로 균형 (20)
    carb_ratio = macro_ratio["carb"]
    fat_ratio = macro_ratio["fat"]

    if carb_ratio <= 0.65 and fat_ratio <= 0.30:
        score += 20
    elif carb_ratio <= 0.70:
        score += 12
    else:
        score += 5

    # 4️⃣ 식사 일관성 (10)
    kcal_values = [d["kcal"] for d in meal_logs]
    variation = max(kcal_values) - min(kcal_values)

    if variation <= 300:
        score += 10
    elif variation <= 500:
        score += 6
    else:
        score += 3

    return min(score, 100)


# =====================================================
# STEP 4-2: 이벤트 기반 점수 보호
# =====================================================
def _apply_event_adjustment(
    *,
    user_id: int,
    raw_score: int,
) -> Dict[str, str | int | None]:
    """
    이벤트가 있을 경우
    - 점수 하한 보호
    - 상태(state) 반환
    - 설명 문구(event_note) 반환
    """

    events = load_events_for_period(user_id)

    state = "normal"
    event_note = None
    score = raw_score

    # 보호 이벤트 (회식 / 여행 / 휴식일)
    if has_protective_event(events):
        score = max(score, 60)
        state = "protected"
        event_note = "이번 주에는 일정상 관리가 어려운 이벤트가 있었어요."

    # 질병 이벤트는 회복 우선
    if any(e["type"] == "illness" for e in events):
        score = max(score, 65)
        state = "recovery"
        event_note = "컨디션 회복이 가장 중요했던 한 주였어요."

    return {
        "score": score,
        "state": state,
        "event_note": event_note,
    }


# =====================================================
# STEP 1-4: 요약 문구 생성
# =====================================================
def _build_weekly_summary(
    score: int,
    macro_ratio: Dict[str, float],
) -> List[str]:
    summary: List[str] = []

    if macro_ratio["protein"] < 0.25:
        summary.append("이번 주는 단백질 섭취가 부족한 편이었어요.")

    if score < 60:
        summary.append("전반적인 식사 균형을 다시 점검해보는 것이 좋아 보여요.")
    else:
        summary.append("큰 무리 없이 비교적 안정적인 한 주였어요.")

    return summary


# =====================================================
# ✅ PUBLIC API (STEP 4-2 최종)
# =====================================================
def build_weekly_report(user_id: int) -> Dict:
    """
    주간 리포트 집계 결과 반환
    (recommendation_layer 이전 단계)
    """

    meal_logs = get_meal_logs(user_id)

    # 기록 없을 경우 fallback
    if not meal_logs:
        meal_logs = _load_weekly_meal_log(user_id)

    macro_ratio = _calculate_macro_ratio(meal_logs)

    raw_score = _calculate_weekly_score(meal_logs, macro_ratio)

    adjusted = _apply_event_adjustment(
        user_id=user_id,
        raw_score=raw_score,
    )

    final_score = adjusted["score"]
    state = adjusted["state"]
    event_note = adjusted["event_note"]

    summary = _build_weekly_summary(final_score, macro_ratio)

    return {
        "weekly": {
            "score": final_score,
            "macro_ratio": macro_ratio,
            "summary": summary,
            "state": state,          # STEP 4-2
            "event_note": event_note # STEP 4-2
        }
    }
