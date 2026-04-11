from __future__ import annotations

from datetime import date
from typing import Dict

from infra.db_server import get_db_conn


class GoalCalculatorService:
    """
    🎯 목표 기반 하루 목표 kcal / macro 계산

    핵심 보장:
    - 목표 체중 변경 시 kcal 반드시 변함
    - 목표 기간 변경 시 kcal 반드시 변함
    - 이전 값 캐시/고정 없음
    """

    # 안전 하한 / 상한
    MIN_KCAL = 1000
    MAX_KCAL = 4500

    # kg 당 kcal 환산 (보수적)
    KCAL_PER_KG = 7700

    # ==================================================
    # 현재 체중 조회 (Oracle 안전)
    # ==================================================
    def _get_current_weight(self, user_id: bytes, fallback: float) -> float:
        """
        body_log 최신값 우선, 없으면 fallback 사용
        """
        with get_db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT weight_kg
                FROM (
                    SELECT weight_kg
                    FROM body_log
                    WHERE user_id = :u
                    ORDER BY measured_at DESC
                )
                WHERE ROWNUM = 1
                """,
                {"u": user_id},
            )
            row = cur.fetchone()

        if row and row[0]:
            return float(row[0])

        return float(fallback)

    # ==================================================
    # 메인 계산
    # ==================================================
    def calculate(
        self,
        *,
        user_id: bytes,
        sex: str,
        birth_year: int,
        height_cm: float,
        weight_kg: float,
        activity_level: str,
        goal_type: str,
        start_date: date,
        target_weight: float,
        target_date: date,
    ) -> Dict:
        """
        반환:
        {
            kcal_target: int,
            macro_target: dict,
            debug: dict
        }
        """

        # 1️⃣ 현재 체중 확정
        current_weight = self._get_current_weight(user_id, weight_kg)

        # 2️⃣ 목표 기간 (최소 1일)
        # 목표 체중 유지인 경우
        if goal_type in ("maintain", "maintenance") or target_date is None:
            days = 1
        else:
            days = max((target_date - start_date).days, 1)

        # 3️⃣ BMR (Mifflin-St Jeor)
        age = date.today().year - birth_year
        if sex == "male":
            bmr = 10 * current_weight + 6.25 * height_cm - 5 * age + 5
        else:
            bmr = 10 * current_weight + 6.25 * height_cm - 5 * age - 161

        # 4️⃣ 활동 계수
        activity_factor = {
            "low": 1.2,
            "medium": 1.4,
            "high": 1.6,
        }.get(activity_level, 1.3)

        tdee = bmr * activity_factor

        # 5️⃣ 목표 체중 변화 → kcal/day
        delta_kg = target_weight - current_weight
        delta_kcal_total = delta_kg * self.KCAL_PER_KG
        delta_kcal_per_day = delta_kcal_total / days

        if goal_type == "weight_loss":
            kcal_target = tdee + delta_kcal_per_day
        elif goal_type in ("weight_gain", "muscle_gain"):
            kcal_target = tdee + delta_kcal_per_day
        else:  # maintenance
            kcal_target = tdee

        # 6️⃣ 안전 범위 보정
        kcal_target = int(
            max(self.MIN_KCAL, min(self.MAX_KCAL, kcal_target))
        )

        # 7️⃣ 매크로 비율
        if goal_type == "weight_loss":
            macro = {"carb": 0.40, "protein": 0.35, "fat": 0.25}
        elif goal_type in ("weight_gain", "muscle_gain"):
            macro = {"carb": 0.45, "protein": 0.30, "fat": 0.25}
        else:
            macro = {"carb": 0.45, "protein": 0.30, "fat": 0.25}

        return {
            "kcal_target": int(kcal_target),
            "macro_target": dict(macro),  # 항상 새 객체
            "debug": {
                "current_weight": current_weight,
                "target_weight": target_weight,
                "days": days,
                "tdee": int(tdee),
            },
        }
