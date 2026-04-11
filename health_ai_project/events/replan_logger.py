from __future__ import annotations

print("🔥🔥🔥 events/replan_logger.py LOADED 🔥🔥🔥")

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import json
import uuid


class ReplanLogger:
    """
    Step 8-A-9-2 (FINAL)
    - circular reference 원천 차단
    - 파일 로그는 '실행 추적용' 최소 정보만 저장
    """

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path("data") / "replan_logs"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # 저장
    # -----------------------------
    def save(
        self,
        *,
        user_id: str,
        replan_result: Dict[str, Any],
    ) -> str:
        """
        결과 저장 후 replan_id 반환
        """

        replan_id = str(uuid.uuid4())
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")

        # ✅ [핵심 수정 포인트]
        uid = user_id.hex() if isinstance(user_id, (bytes, bytearray)) else str(user_id)

        user_dir = self.base_dir / f"user_{uid}"
        user_dir.mkdir(parents=True, exist_ok=True)
        file_path = user_dir / f"{ts}_{replan_id}.json"

        # ✅ payload도 동일 uid 사용
        payload = {
            "replan_id": replan_id,
            "saved_at": datetime.utcnow().isoformat(),
            "user_id": uid,
            "meta": replan_result.get("meta"),
            "status": (
                replan_result
                .get("messages", {})
                .get("status")
            ),
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        return replan_id

    # -----------------------------
    # 조회
    # -----------------------------
    def load(self, *, user_id: str, replan_id: str) -> Dict[str, Any]:
        uid = user_id.hex() if isinstance(user_id, (bytes, bytearray)) else str(user_id)
        user_dir = self.base_dir / f"user_{uid}"

        for p in user_dir.glob(f"*_{replan_id}.json"):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)

        raise FileNotFoundError(f"replan_id not found: {replan_id}")

    def list(
        self,
        *,
        user_id: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        uid = user_id.hex() if isinstance(user_id, (bytes, bytearray)) else str(user_id)
        user_dir = self.base_dir / f"user_{uid}"

        if not user_dir.exists():
            return []

        files = sorted(user_dir.glob("*.json"), reverse=True)[:limit]
        results: List[Dict[str, Any]] = []

        for p in files:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    results.append({
                        "replan_id": data.get("replan_id"),
                        "saved_at": data.get("saved_at"),
                        "meta": data.get("meta", {}),
                    })
            except Exception:
                continue

        return results
