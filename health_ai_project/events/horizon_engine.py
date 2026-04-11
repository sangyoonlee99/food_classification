# events/horizon_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from datetime import date, timedelta

DEFAULT_HORIZON_DAYS = 3
NEXT_DAY_WEIGHT = [0.55, 0.30, 0.15]
WEEKLY_MIX_RATIO = 0.25

MAX_DAILY_DEFICIT_KCAL = 450
MAX_DAILY_SURPLUS_KCAL = 300

MAX_EXTRA_CARDIO_MIN = 35
MAX_EXTRA_STRENGTH_MIN = 25


@dataclass
class HorizonDecision:
    horizon_days: int
    show_user_view: bool
    strategy: str
    diet_day_adjustments: Dict[date, Dict[str, Any]]
    exercise_day_adjustments: Dict[date, Dict[str, Any]]
    notes: List[str]


class HorizonEngine:
    def __init__(self):
        pass

    def plan(
        self,
        *,
        event_date: date,
        base_week_dates: List[date],
        event_type: str,
        intensity: Optional[str],
        actual_delta_kcal: Optional[float] = None,
        estimated_delta_kcal: float = 400.0,
        user_show_details: bool = False,
        user_exercise_available: bool = True,
        user_exercise_level: str = "mid",
        preferred_exercises: Optional[List[str]] = None,
    ) -> HorizonDecision:

        notes: List[str] = []
        preferred_exercises = preferred_exercises or []

        delta = actual_delta_kcal if actual_delta_kcal is not None else estimated_delta_kcal
        if actual_delta_kcal is not None:
            notes.append(f"실제 기록 기반 Δkcal={delta:+.0f}")
        else:
            notes.append(f"기록이 없어 추정 Δkcal={delta:+.0f} 사용")

        horizon_days = self._choose_horizon_days(event_type=event_type, intensity=intensity, delta_kcal=delta)
        notes.append(f"Horizon={horizon_days}일(다음날 비중 최대)")

        week_dates = [d for d in base_week_dates if d >= event_date] or [event_date]

        future_days = [event_date + timedelta(days=i) for i in range(1, horizon_days + 1)]
        future_days = [d for d in future_days if d in week_dates]
        remainder_days = [d for d in week_dates if d not in ([event_date] + future_days)]

        diet_share, exercise_share = self._split_diet_exercise(
            delta_kcal=delta,
            user_exercise_available=user_exercise_available,
            user_exercise_level=user_exercise_level,
            preferred_exercises=preferred_exercises,
        )
        notes.append(f"분배: 식단 {diet_share:.0%} / 운동 {exercise_share:.0%}")

        horizon_kcal = delta * (1.0 - WEEKLY_MIX_RATIO)
        weekly_kcal = delta * WEEKLY_MIX_RATIO

        horizon_alloc = self._allocate_by_weights(horizon_kcal, future_days)
        weekly_alloc = self._allocate_even(weekly_kcal, remainder_days)

        alloc_map: Dict[date, float] = {}
        for d, v in horizon_alloc.items():
            alloc_map[d] = alloc_map.get(d, 0.0) + v
        for d, v in weekly_alloc.items():
            alloc_map[d] = alloc_map.get(d, 0.0) + v

        diet_day_adjustments: Dict[date, Dict[str, Any]] = {}
        exercise_day_adjustments: Dict[date, Dict[str, Any]] = {}

        for d, day_delta in alloc_map.items():
            diet_kcal = day_delta * diet_share
            ex_kcal = day_delta * exercise_share

            diet_policy = self._diet_policy_from_kcal(diet_kcal)
            ex_policy = self._exercise_policy_from_kcal(ex_kcal, user_exercise_available, user_exercise_level, preferred_exercises)

            diet_day_adjustments[d] = diet_policy
            exercise_day_adjustments[d] = ex_policy

        notes.append("상세 표시" if user_show_details else "상세 숨김(요약만)")

        return HorizonDecision(
            horizon_days=horizon_days,
            show_user_view=user_show_details,
            strategy="B+partialC",
            diet_day_adjustments=diet_day_adjustments,
            exercise_day_adjustments=exercise_day_adjustments,
            notes=notes,
        )

    # ✅ 서비스 호환용 래퍼: replan_horizon_service가 calculate(user_settings=...)로 불러도 OK
    def calculate(
        self,
        *,
        event_date: date,
        base_week_dates: List[date],
        event_type: str,
        intensity: Optional[str] = None,
        user_settings: Optional[Dict[str, Any]] = None,
        actual_delta_kcal: Optional[float] = None,
        estimated_delta_kcal: float = 400.0,
    ) -> HorizonDecision:

        user_settings = user_settings or {}

        return self.plan(
            event_date=event_date,
            base_week_dates=base_week_dates,
            event_type=event_type,
            intensity=intensity,
            actual_delta_kcal=actual_delta_kcal,
            estimated_delta_kcal=estimated_delta_kcal,
            user_show_details=bool(user_settings.get("show_details", False)),
            user_exercise_available=bool(user_settings.get("exercise_available", True)),
            user_exercise_level=str(user_settings.get("exercise_level", "mid")),
            preferred_exercises=list(user_settings.get("preferred_exercises", [])) or None,
        )

    # -----------------------------
    # 내부 로직(기존 그대로)
    # -----------------------------
    def _choose_horizon_days(self, *, event_type: str, intensity: Optional[str], delta_kcal: float) -> int:
        et = (event_type or "").lower()
        big = abs(delta_kcal) >= 600

        if "travel" in et:
            return 3
        if "overeating" in et or "binge" in et:
            return 3
        if "dinner" in et or "restaurant" in et or "dinner_out" in et:
            return 2 if not big else 3

        if intensity in ("high", "strong", "heavy"):
            return 3
        return 2

    def _split_diet_exercise(
        self,
        *,
        delta_kcal: float,
        user_exercise_available: bool,
        user_exercise_level: str,
        preferred_exercises: List[str],
    ) -> Tuple[float, float]:
        if not user_exercise_available:
            return 0.90, 0.10

        lvl = (user_exercise_level or "mid").lower()
        if lvl == "high":
            return 0.55, 0.45
        if lvl == "low":
            return 0.75, 0.25

        pref = " ".join(preferred_exercises).lower()
        if any(k in pref for k in ["gym", "weights", "strength", "crossfit"]):
            return 0.60, 0.40

        return 0.65, 0.35

    def _allocate_by_weights(self, total_kcal: float, days: List[date]) -> Dict[date, float]:
        if not days:
            return {}
        w = NEXT_DAY_WEIGHT[: len(days)]
        s = sum(w)
        w = [x / s for x in w]
        return {d: total_kcal * w[i] for i, d in enumerate(days)}

    def _allocate_even(self, total_kcal: float, days: List[date]) -> Dict[date, float]:
        if not days:
            return {}
        each = total_kcal / len(days)
        return {d: each for d in days}

    def _diet_policy_from_kcal(self, day_delta_kcal: float) -> Dict[str, Any]:
        if day_delta_kcal > 0:
            day_delta_kcal = min(day_delta_kcal, MAX_DAILY_DEFICIT_KCAL)
        else:
            day_delta_kcal = max(day_delta_kcal, -MAX_DAILY_SURPLUS_KCAL)

        if day_delta_kcal > 0:
            return {
                "same_day": {},
                "next_day": {
                    "carb_ratio": -0.12,
                    "protein_ratio": +0.06,
                    "kcal_delta": -round(day_delta_kcal, 1),
                },
            }
        elif day_delta_kcal < 0:
            return {
                "same_day": {},
                "next_day": {
                    "carb_ratio": +0.08,
                    "fat_ratio": +0.05,
                    "kcal_delta": -round(day_delta_kcal, 1),
                },
            }
        return {"same_day": {}, "next_day": {}}

    def _exercise_policy_from_kcal(
        self,
        day_delta_kcal: float,
        user_exercise_available: bool,
        user_exercise_level: str,
        preferred_exercises: List[str],
    ) -> Dict[str, Any]:
        if not user_exercise_available:
            return {"same_day": {"skip": True}, "next_day": {"rest": True}}

        cardio_min = 0
        strength_min = 0

        if day_delta_kcal > 0:
            pref = " ".join(preferred_exercises).lower()
            lvl = (user_exercise_level or "mid").lower()

            if any(k in pref for k in ["gym", "weights", "strength", "crossfit"]):
                strength_min = int(min(MAX_EXTRA_STRENGTH_MIN, day_delta_kcal / 12))
                cardio_min = int(min(MAX_EXTRA_CARDIO_MIN, day_delta_kcal / 10))
            else:
                cardio_min = int(min(MAX_EXTRA_CARDIO_MIN, day_delta_kcal / 8))
                strength_min = int(min(MAX_EXTRA_STRENGTH_MIN, day_delta_kcal / 18))

            if lvl == "low":
                cardio_min = min(cardio_min, 20)
                strength_min = min(strength_min, 15)

            return {
                "same_day": {},
                "next_day": {
                    "cardio_min": cardio_min,
                    "strength_min": strength_min,
                    "intensity": "up" if (lvl in ("mid", "high")) else "keep",
                },
            }

        return {"same_day": {}, "next_day": {"rest": False, "intensity": "keep"}}
