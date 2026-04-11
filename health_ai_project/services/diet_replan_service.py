# services/diet_replan_service.py
# STEP 8-A-2: 이벤트 정책 기반 식단 재계획 서비스 (최종 수정본)
# - 이벤트 당일 kcal 증가 ❌
# - 당일: 유지 또는 감소 / 다음날·주간: 회수
# - 기존 API / 정책 / 메모 로직 100% 유지

from __future__ import annotations

from typing import Any, Dict, Optional, List

from common.schemas import Event, DailyDietPlan, WeeklyDietPlan, UserGoal


class DietReplanService:
    """
    이벤트 발생 시 기존 식단 계획을 '정책(policy)'에 따라 수정한다.
    """

    # ----------------------------
    # ✅ 신형 API (EventService / ReplanOrchestrator용)
    # ----------------------------
    def replan(
        self,
        base_plan: Dict[str, Any],
        event_type: str,
        intensity: Optional[str] = None,
        policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        policy = policy or {}
        notes: List[str] = []

        # 0) 안전 하한
        safety = policy.get("safety", {})
        min_calorie = float(safety.get("min_calorie", 1200))

        # 1) 총 칼로리 계산
        total_cal = float(base_plan.get("total_calorie", 0.0))
        if total_cal <= 0:
            total_cal = self._recalc_total(base_plan)

        # -------------------------------------------------
        # 2) same_day 정책 (당일: 유지 또는 감소 ONLY)
        # -------------------------------------------------
        same_day = policy.get("same_day", {})
        if same_day:
            base_plan, same_notes = self._apply_same_day_policy(
                base_plan,
                same_day,
                prevent_increase=True,   # 🔥 핵심
            )
            notes.extend(same_notes)

        total_cal = self._recalc_total(base_plan)

        # -------------------------------------------------
        # 3) next_day / horizon / weekly_budget
        #    → 오늘 플랜 직접 변경 ❌
        #    → 후속 가이드로 notes에만 남김
        # -------------------------------------------------
        next_day = policy.get("next_day", {})
        horizon_days = int(policy.get("horizon_days", 1))
        weekly_budget = policy.get("weekly_budget", {})

        if next_day:
            notes.append(f"[next_day] 회수 정책 적용 예정: {next_day}")

        if horizon_days and horizon_days > 1:
            notes.append(f"[horizon] 이벤트 영향 기간: {horizon_days}일")

        if weekly_budget.get("enabled"):
            notes.append(f"[weekly_budget] 활성화: {weekly_budget}")

        # -------------------------------------------------
        # 4) 안전 하한 강제
        # -------------------------------------------------
        total_cal = self._recalc_total(base_plan)
        if 0 < total_cal < min_calorie:
            ratio = min_calorie / total_cal
            base_plan = self._scale_all_meals(base_plan, ratio)
            notes.append(
                f"[safety] total_cal {total_cal:.0f} < {min_calorie:.0f} → 하한 보정"
            )
            total_cal = self._recalc_total(base_plan)

        base_plan["total_calorie"] = round(total_cal, 2)

        return {
            "plan": base_plan,
            "notes": notes,
            "applied": policy,
        }

    # ----------------------------
    # ✅ 구형 API (하위호환 유지)
    # ----------------------------
    def replan_daily(
        self,
        event: Event,
        daily_plan: DailyDietPlan,
        goal: UserGoal,
    ) -> DailyDietPlan:
        """
        ⚠️ 구형 로직 수정:
        - 이벤트 당일 kcal 증가 ❌
        - 항상 유지 또는 감소
        """
        total_cal = float(daily_plan.total_calorie or 0.0)
        if total_cal <= 0:
            total_cal = sum(
                i.calorie for i in (
                    daily_plan.breakfast
                    + daily_plan.lunch
                    + daily_plan.dinner
                    + daily_plan.snacks
                )
            )

        # 🔥 기존 delta → 감소 전용으로 변환
        delta = abs(self._event_delta(event))
        adjusted_cal = max(total_cal - delta, 1200)

        ratio = adjusted_cal / total_cal if total_cal else 1.0

        def _scale(meals):
            for m in meals:
                m.calorie *= ratio
                m.carb *= ratio
                m.protein *= ratio
                m.fat *= ratio
            return meals

        daily_plan.breakfast = _scale(daily_plan.breakfast)
        daily_plan.lunch = _scale(daily_plan.lunch)
        daily_plan.dinner = _scale(daily_plan.dinner)
        daily_plan.snacks = _scale(daily_plan.snacks)

        daily_plan.total_calorie = round(adjusted_cal, 1)
        return daily_plan

    def replan_weekly(
        self,
        event: Event,
        weekly_plan: WeeklyDietPlan,
        goal: UserGoal,
    ) -> WeeklyDietPlan:
        for day_plan in weekly_plan.days:
            if day_plan.date >= event.datetime_start.date():
                self.replan_daily(event, day_plan, goal)
        return weekly_plan

    # ----------------------------
    # 내부 유틸
    # ----------------------------
    def _event_delta(self, event: Event) -> float:
        """
        ❗ 의미 변경:
        - 반환값은 '감산 기준치'
        """
        et = event.event_type
        if et in ("dinner_drinking", "회식"):
            return 400
        if et in ("restaurant_meal",):
            return 300
        if et in ("travel", "여행"):
            return 200
        if et in ("sleep_debt", "수면부족"):
            return 0
        return 0

    def _apply_same_day_policy(
        self,
        plan: Dict[str, Any],
        same_day_policy: Dict[str, Any],
        prevent_increase: bool = True,
    ):
        """
        prevent_increase=True:
        - calorie / macro 어떤 경우에도 증가 불가
        """
        notes: List[str] = []
        meals = plan.get("meals", {}) or {}

        if not isinstance(meals, dict):
            return plan, ["[same_day] meals 구조 오류 → 스킵"]

        # 1) mode 처리
        for meal_key, rule in same_day_policy.items():
            if meal_key not in meals or not isinstance(rule, dict):
                continue

            mode = rule.get("mode")
            if mode == "skip":
                meals[meal_key] = []
                notes.append(f"[same_day] {meal_key}: skip")
            elif mode == "free":
                notes.append(f"[same_day] {meal_key}: free(유지)")

        # 2) macro ratio (증가 방지)
        for meal_key, rule in same_day_policy.items():
            if meal_key not in meals or not isinstance(rule, dict):
                continue

            items = meals.get(meal_key) or []
            if not items:
                continue

            for macro, key in (
                ("carb_ratio", "carb"),
                ("protein_ratio", "protein"),
                ("fat_ratio", "fat"),
            ):
                if macro in rule:
                    raw = float(rule[macro])
                    factor = 1.0 + raw
                    if prevent_increase:
                        factor = min(1.0, factor)

                    factor = max(0.0, factor)

                    for it in items:
                        if isinstance(it, dict) and key in it:
                            it[key] = float(it[key]) * factor

                    notes.append(
                        f"[same_day] {meal_key}: {macro}={raw} → factor={factor:.2f}"
                    )

        plan["meals"] = meals
        plan["total_calorie"] = round(self._recalc_total(plan), 2)
        return plan, notes

    def _scale_all_meals(self, plan: Dict[str, Any], ratio: float) -> Dict[str, Any]:
        meals = plan.get("meals", {}) or {}
        for items in meals.values():
            if not isinstance(items, list):
                continue
            for it in items:
                if not isinstance(it, dict):
                    continue
                for k in ("calorie", "carb", "protein", "fat", "portion_gram"):
                    if k in it and it[k] is not None:
                        it[k] = float(it[k]) * ratio

        plan["meals"] = meals
        plan["total_calorie"] = round(self._recalc_total(plan), 2)
        return plan

    def _recalc_total(self, plan: Dict[str, Any]) -> float:
        meals = plan.get("meals", {}) or {}
        total = 0.0
        for items in meals.values():
            if not isinstance(items, list):
                continue
            for it in items:
                if isinstance(it, dict) and "calorie" in it and it["calorie"] is not None:
                    total += float(it["calorie"])
        return total


# ✅ 하위호환 alias
ReplanService = DietReplanService
