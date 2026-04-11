# service/selection_logger.py
import json
from pathlib import Path
from datetime import datetime

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_PATH = LOG_DIR / "selection_log.jsonl"


def log_user_selection(
    image_id: str,
    bbox: list,
    ai_top1: dict,
    candidates: list,
    user_selected: str
):
    """
    사용자 선택 로그 저장 (1 detection = 1 log)

    Args:
        image_id: 이미지 ID
        bbox: [x1, y1, x2, y2]
        ai_top1: {"class": str, "similarity": float}
        candidates: 후보 리스트
        user_selected: 사용자가 선택한 클래스
    """
    record = {
        "image_id": image_id,
        "bbox": bbox,
        "ai_top1": ai_top1,
        "candidates": candidates,
        "user_selected": user_selected,
        "timestamp": datetime.now().isoformat()
    }

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
