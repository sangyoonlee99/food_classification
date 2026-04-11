from __future__ import annotations
from typing import Any, Dict, List


def build_replan_messages(
    replan_result: Dict[str, Any],
    state: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    - 입력: replan_result(오케스트레이터 실행 결과), state(선택)
    - 출력: UI에서 그대로 렌더링 가능한 cards
    """
    state = state or {}
    cards: List[Dict[str, Any]] = []

    # ---- safest extraction ----
    meal = (replan_result.get("actions") or {}).get("meal") or {}
    ex = (replan_result.get("actions") or {}).get("exercise") or {}
    horizon = replan_result.get("horizon") or {}
    signature = replan_result.get("recommendation_signature", "keep")

    # 1️⃣ summary
    summary_lines: List[str] = []
    if signature == "keep":
        summary_lines.append("오늘은 큰 변경 없이 유지로 진행합니다.")
    else:
        summary_lines.append("오늘은 상태에 맞춰 일부 조정을 반영합니다.")

    if meal.get("change_menu"):
        summary_lines.append("식단은 메뉴 변경이 필요합니다.")
    elif meal.get("adjust_grams"):
        summary_lines.append("식단은 g 단위 조정을 권장합니다.")

    if ex.get("change_routine"):
        summary_lines.append("운동은 루틴 변경이 필요합니다.")
    elif ex.get("adjust_minutes"):
        summary_lines.append("운동은 시간(분) 조정을 권장합니다.")

    cards.append(
        {
            "type": "summary",
            "title": "오늘의 요약",
            "severity": "info",
            "bullets": summary_lines,
        }
    )

    # 2️⃣ meal
    if meal.get("change_menu") or meal.get("adjust_grams"):
        bullets = []
        if meal.get("change_menu"):
            bullets.append("메뉴를 변경해 목표에 맞춥니다.")
        if meal.get("adjust_grams"):
            bullets.append("섭취량을 g 단위로 조절합니다.")

        cards.append(
            {
                "type": "meal",
                "title": "식단 조정",
                "severity": "warning" if meal.get("change_menu") else "info",
                "bullets": bullets,
            }
        )

    # 3️⃣ exercise
    if ex.get("change_routine") or ex.get("adjust_minutes"):
        bullets = []
        if ex.get("change_routine"):
            bullets.append("운동 루틴을 변경합니다.")
        if ex.get("adjust_minutes"):
            bullets.append("운동 시간을 조절합니다.")

        cards.append(
            {
                "type": "exercise",
                "title": "운동 조정",
                "severity": "warning" if ex.get("change_routine") else "info",
                "bullets": bullets,
            }
        )

    # 4️⃣ horizon
    next_day = (horizon or {}).get("next_day")
    if next_day:
        hz = []
        if "kcal_delta" in next_day:
            hz.append(f"내일 목표 열량: {next_day['kcal_delta']} kcal")
        if "carb_ratio" in next_day:
            hz.append(f"탄수 비율: {next_day['carb_ratio']}")

        cards.append(
            {
                "type": "horizon",
                "title": "내일 예고",
                "severity": "info",
                "bullets": hz,
            }
        )

    return {
        "signature": signature,
        "cards": cards,
        "raw": replan_result,
    }
