# decision/ui_policy.py

"""
UI Policy (Decision Layer)

역할:
- AI 결과(top1 similarity)를 보고
  ▶ 자동 확정
  ▶ 사용자 선택
  ▶ 인식 실패
중 하나를 결정한다

⚠️ 모델 연산 없음
⚠️ core와 완전히 분리
"""

# =========================
# Threshold Config
# =========================
AUTO_ACCEPT_THRES = 0.70    # 이 이상이면 자동 확정
USER_SELECT_THRES = 0.45    # 이 이상이면 후보 선택 UI
# 그 이하는 REJECT


# =========================
# Decision Function
# =========================
def decide_ui_mode(similarity: float) -> str:
    """
    Args:
        similarity (float): top1 cosine similarity

    Returns:
        str:
          - "AUTO_ACCEPT"
          - "USER_SELECT"
          - "REJECT"
    """
    if similarity >= AUTO_ACCEPT_THRES:
        return "AUTO_ACCEPT"
    elif similarity >= USER_SELECT_THRES:
        return "USER_SELECT"
    else:
        return "REJECT"


# =========================
# Helper (optional)
# =========================
def attach_ui_mode(results: list) -> list:
    """
    core.inference 결과에 ui_mode를 붙여준다

    Args:
        results: [
          {
            "bbox": [...],
            "top1": {"class": str, "similarity": float},
            "candidates": [...]
          }
        ]

    Returns:
        same list, but with "ui_mode" added
    """
    for r in results:
        sim = r["top1"]["similarity"]
        r["ui_mode"] = decide_ui_mode(sim)

    return results
