from __future__ import annotations

print("🔥🔥🔥 routine/routine_engine.py LOADED 🔥🔥🔥")

from typing import Dict, Any, List
from datetime import date, timedelta
import json
import hashlib

from routine.routine_template import build_exercise_template
from routine.routine_policy import apply_meal_policy, apply_exercise_policy
from services.exercise_service import decide_exercise_actions

from events.event_service import EventService
from diet.daily_diet_planner import generate_daily_diet
from common.schemas import UserProfile, UserGoal


# ==================================================
# Utils
# ==================================================
def _stable_hash(obj: Any) -> str:
    try:
        s = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        s = str(obj)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]


def _mealplan_to_base_diet(meal_plan: Dict[str, Any]) -> Dict[str, Any]:
    meals = meal_plan.get("meals") or {}
    base_meals = {k: list(v.get("items", [])) for k, v in meals.items()}
    total = sum(float(i.get("calorie", 0) or 0) for v in base_meals.values() for i in v)

    return {
        "meals": base_meals,
        "total_calorie": round(total, 1),
    }


def _base_diet_to_mealplan(
    base_diet: Dict[str, Any],
    new_target_kcal: float,
) -> Dict[str, Any]:
    meals = base_diet.get("meals") or {}

    def kcal(items):
        return round(sum(float(i.get("calorie", 0) or 0) for i in items), 1)

    return {
        "target_kcal": float(new_target_kcal),
        "total_kcal": float(base_diet.get("total_calorie", 0)),
        "meals": {
            k: {
                "target_kcal": kcal(v),
                "items": v,
            }
            for k, v in meals.items()
        },
    }


# ==================================================
# Engine
# ==================================================
class RoutineEngine:
    def build_actions(
        self,
        *,
        user_id: bytes,
        target_date: date,
        event_flags: Dict[str, Any] | None = None,
        state: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        state = state or {}

        if event_flags is None:
            event_flags = EventService().build_event_flags(
                user_id=user_id,
                target_date=target_date,
            )

        exercise_action = decide_exercise_actions(
            event_flags=event_flags,
            state=state,
        )

        sig = f"{user_id.hex()}:{target_date}:{_stable_hash(event_flags)}"

        return {
            "meal": {"event_flags": event_flags},
            "exercise": exercise_action,
            "event_flags": event_flags,
            "recommendation_signature": sig,
        }

    # ==================================================
    # 🔥 B안 핵심
    # ==================================================
    def build_routine(
        self,
        *,
        user_id: bytes,
        user_profile: Dict[str, Any],
        user_goal: Dict[str, Any],
        actions: Dict[str, Any],
        target_date: date,
        state: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        state = state or {}

        profile = UserProfile.model_validate(user_profile)
        goal = UserGoal.model_validate(user_goal)

        # ------------------------------
        # 1️⃣ 기본 식단 생성
        # ------------------------------
        plan = generate_daily_diet(
            profile=profile,
            goal=goal,
            date_value=target_date,
        )

        meal_plan = {
            "target_kcal": plan.target_kcal,
            "total_kcal": plan.total_kcal,
            "meals": {
                k: {
                    "target_kcal": sum(i.calorie for i in v),
                    "items": [i.model_dump() for i in v],
                }
                for k, v in (plan.meals or {}).items()
            },
        }

        es = EventService()

        # ==================================================
        # 2️⃣ D0 (당일) 이벤트 → 저녁 제거
        # ==================================================
        today_events = es.load_events_for_date(
            user_id=user_id,
            target_date=target_date,
        )

        base_diet = _mealplan_to_base_diet(meal_plan)
        new_target_kcal = plan.target_kcal

        for ev in today_events:
            if ev.get("event_type_std") == "DINNER_OUT":
                # 🔥 당일 저녁 비우기
                base_diet["meals"]["dinner"] = []
                base_diet["total_calorie"] = max(
                    base_diet["total_calorie"] - 300,
                    1200,
                )

        # ==================================================
        # 3️⃣ D-1 (전날) 이벤트 → 다음날 kcal 회수
        # ==================================================
        prev_date = target_date - timedelta(days=1)
        prev_events = es.load_events_for_date(
            user_id=user_id,
            target_date=prev_date,
        )

        for ev in prev_events:
            if ev.get("event_type_std") == "DINNER_OUT":
                # 🔥 다음날 kcal 회수 (B안)
                new_target_kcal = max(new_target_kcal - 200, 1200)

                # 아침/점심에서 우선 회수
                for mt in ("breakfast", "lunch"):
                    items = base_diet["meals"].get(mt) or []
                    if not items:
                        continue
                    cut = min(
                        100,
                        sum(float(i.get("calorie", 0)) for i in items),
                    )
                    base_diet["total_calorie"] -= cut
                    break
        # ==================================================
        # 3️⃣-2 A안: 전날 실제 섭취 초과 → 다음날 kcal 보정 (상한 300)
        # ==================================================
        from infra.db_server import get_db_conn

        prev_date = target_date - timedelta(days=1)

        with get_db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT NVL(SUM(kcal), 0)
                FROM meal_record
                WHERE user_id = :u
                  AND TRUNC(eaten_at) = TRUNC(:d)
                """,
                {"u": user_id, "d": prev_date},
            )
            eaten_prev = float(cur.fetchone()[0] or 0)

        # 🔴 기준은 plan.target_kcal (전날 목표)
        prev_target_kcal = float(user_goal.get("target_kcal") or plan.target_kcal or 0)
        excess_kcal = eaten_prev - prev_target_kcal

        if excess_kcal > 0:
            recover_kcal = min(excess_kcal, 300)

            new_target_kcal = max(
                new_target_kcal - recover_kcal,
                1200,
            )

            # 🔥 UI 반영 핵심
            base_diet["total_calorie"] = max(
                base_diet["total_calorie"] - recover_kcal,
                1200,
            )

        # ------------------------------
        # 4️⃣ meal_plan 재구성
        # ------------------------------
        meal_plan = _base_diet_to_mealplan(
            base_diet=base_diet,
            new_target_kcal=new_target_kcal,
        )

        # 🔥 B안: horizon kcal delta 적용 (다음날/주간)
        horizon_delta = (
            actions.get("meal", {})
            .get("event_flags", {})
            .get("horizon_kcal_delta", {})
            .get(target_date.isoformat())
        )
        if horizon_delta:
            new_target = max(
                float(meal_plan.get("target_kcal", 0)) + float(horizon_delta),
                1200,
            )

            meal_plan["target_kcal"] = new_target

            # 🔥 핵심: 식사별 kcal 비율 재분배
            meals = meal_plan.get("meals", {})
            total_before = sum(
                float(v.get("target_kcal", 0)) for v in meals.values()
            ) or 1.0

            ratio = new_target / total_before

            for m in meals.values():
                m["target_kcal"] = round(float(m.get("target_kcal", 0)) * ratio)

        # ------------------------------
        # 5️⃣ 운동 / 정책 적용
        # ------------------------------
        exercise_plan = build_exercise_template(
            user_profile=user_profile,
            user_goal=user_goal,
            state=state,
            exercise_context={},
        )

        meal_plan = apply_meal_policy(
            meal_plan=meal_plan,
            meal_actions=actions.get("meal") or {},
            user_profile=user_profile,
        )

        exercise_plan = apply_exercise_policy(
            exercise_plan=exercise_plan,
            exercise_actions=actions.get("exercise") or {},
            user_profile=user_profile,
        )

        return {
            "meal_plan": meal_plan,
            "exercise_plan": exercise_plan,
        }
