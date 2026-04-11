# services/recommendation_parser.py
from __future__ import annotations
from typing import List, Dict, Any


def parse_signature(signature: str) -> List[Dict[str, Any]]:
    """
    recommendation_layer → execution_policy → executor 공통 파서

    signature 예:
      - "keep"
      - "diet:adjust"
      - "diet:menu"
      - "exercise:adjust"
      - "exercise:routine"
      - "diet:adjust|exercise:adjust"  (확장 대비)
    """

    if not signature or signature == "keep":
        return []

    actions: List[Dict[str, Any]] = []

    parts = signature.split("|")

    for part in parts:
        if ":" not in part:
            continue

        domain, action = part.split(":", 1)

        # --------------------
        # DIET
        # --------------------
        if domain == "diet":
            if action == "adjust":
                actions.append({
                    "type": "diet",
                    "mode": "micro",          # kcal 미세조정
                })
            elif action == "menu":
                actions.append({
                    "type": "diet",
                    "mode": "macro",          # 식단 구조 변경
                })

        # --------------------
        # EXERCISE
        # --------------------
        elif domain == "exercise":
            if action == "adjust":
                actions.append({
                    "type": "exercise",
                    "mode": "micro",          # 시간/강도 조정
                })
            elif action == "routine":
                actions.append({
                    "type": "exercise",
                    "mode": "macro",          # 루틴 변경
                })

    return actions
