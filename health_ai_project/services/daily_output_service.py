# services/daily_output_service.py
# STEP 6-1: 일일 출력 문장 3줄 고정 서비스 (FINAL + UI Facade)

from __future__ import annotations

from typing import Dict, Any, List
from datetime import date
import json

from infra.db_server import get_db_conn

from common.schemas import (
    DailyDietPlan,
    DailyExercisePlan,
    UserGoal,
    Event,
)
from services.daily_goal_service import DailyGoalService
from services.next_meal_service import NextMealService


class DailyOutputService:
    """
    하루에 사용자에게 노출할 메시지를 3줄로 고정
    - 목표
    - 핵심 행동
    - 다음 끼니

    ✅ 엔진/도메인용: build(goal, diet, exercise, event)
    ✅ UI(Home)용: build_ui(user_id, today)  -> summary_lines/status 반환
    """

    def __init__(self):
        self.goal_svc = DailyGoalService()
        self.next_meal_svc = NextMealService()

    # ==================================================
    # ✅ 기존 엔진/도메인 계약 (절대 깨지면 안됨)
    # ==================================================
    def build(
        self,
        goal: UserGoal,
        diet: DailyDietPlan,
        exercise: DailyExercisePlan,
        event: Event | None = None,
    ) -> Dict[str, str]:

        goal_sentence = self.goal_svc.build_daily_goal(
            goal=goal,
            diet=diet,
            exercise=exercise,
            event=event,
        )

        action_sentence = self._build_core_action(
            diet=diet,
            exercise=exercise,
            event=event,
        )

        next_meal = self.next_meal_svc.recommend(
            diet_plan=diet,
            goal=goal,
        )
        next_meal_sentence = next_meal["headline"]

        return {
            "goal": goal_sentence,
            "action": action_sentence,
            "next_meal": next_meal_sentence,
        }

    # ==================================================
    # ✅ UI(Home) 전용 진입 (DB 기반)
    # - Home에서 요구하는 형태로 반환:
    #   { "summary_lines": [...], "status": "neutral|partial|applied", "raw": {...} }
    # ==================================================
    def build_ui(self, *, user_id: bytes, today: date) -> Dict[str, Any]:
        """
        Home 화면용 3줄 요약 생성.
        - DB에 데이터 거의 없어도 안전하게 동작
        - 기존 build() 계약과 분리 (테스트용 아님, UI Facade)
        """

        goal_row = self._load_active_goal_row(user_id=user_id)
        if not goal_row:
            return {
                "summary_lines": [],
                "status": "neutral",
                "raw": {"reason": "no_active_goal"},
            }

        goal_type, kcal_target, macro_target = goal_row
        macro = {}
        if macro_target:
            try:
                macro = json.loads(macro_target)
            except Exception:
                macro = {}

        # 오늘 요약(식사/운동) 없을 수 있으니 안전하게
        meal_sum = self._load_daily_meal_summary(user_id=user_id, day=today)
        ex_sum = self._load_daily_exercise_summary(user_id=user_id, day=today)

        # 1) 목표 한 줄
        goal_label = {
            "weight_loss": "체중 감량",
            "maintenance": "체중 유지",
            "weight_gain": "체중 증량",
        }.get(goal_type, goal_type)

        goal_line = (
            f"🎯 오늘 목표: {goal_label} · {kcal_target or '-'} kcal"
        )

        # 2) 핵심 행동 한 줄 (기록 기반 간단 로직)
        action_line = self._build_core_action_ui(
            goal_type=goal_type,
            meal_sum=meal_sum,
            ex_sum=ex_sum,
        )

        # 3) 다음 끼니 한 줄 (기록/목표 기반 간단 추천)
        next_meal_line = self._build_next_meal_ui(
            goal_type=goal_type,
            macro=macro,
        )

        status = self._decide_status_ui(meal_sum=meal_sum, ex_sum=ex_sum)

        return {
            "summary_lines": [goal_line, action_line, next_meal_line],
            "status": status,
            "raw": {
                "goal_type": goal_type,
                "kcal_target": kcal_target,
                "macro": macro,
                "meal_sum": meal_sum,
                "ex_sum": ex_sum,
            },
        }

    # -------------------------------------------------
    # 기존 core action (도메인 객체 기반)
    # -------------------------------------------------
    def _build_core_action(
        self,
        diet: DailyDietPlan,
        exercise: DailyExercisePlan,
        event: Event | None,
    ) -> str:

        if event:
            if event.event_type == "dinner_drinking":
                return "수분 섭취를 충분히 하고 가벼운 활동 위주로 하루를 마무리하세요."
            if event.event_type == "overeating":
                return "오늘은 유산소 활동을 조금 늘려 균형을 맞춰보세요."

        if exercise.total_minutes < 20:
            return "오늘은 가벼운 걷기라도 20분 이상 해보세요."

        return "오늘 계획된 루틴을 그대로 유지해 주세요."

    # -------------------------------------------------
    # UI용 core action (DB summary 기반)
    # -------------------------------------------------
    def _build_core_action_ui(
        self,
        *,
        goal_type: str,
        meal_sum: Dict[str, Any],
        ex_sum: Dict[str, Any],
    ) -> str:
        total_kcal = meal_sum.get("total_kcal")
        cardio_min = ex_sum.get("cardio_minutes") or 0
        total_min = ex_sum.get("total_minutes") or 0

        # 기록 자체가 없을 때
        if total_kcal is None and total_min == 0:
            return "👉 오늘은 먼저 식단/운동 기록을 남겨주세요. 기록이 있어야 추천이 시작돼요."

        # 감량 목표면: 과식/운동부족에 민감
        if goal_type == "weight_loss":
            if total_kcal is not None and total_kcal >= 2200:
                return "오늘은 섭취량이 높았어요. 저녁은 가볍게 + 유산소 20분을 추천해요."
            if total_min < 20:
                return "오늘 활동량이 적어요. 가벼운 걷기 20분부터 시작해보세요."
            return "좋아요. 오늘 루틴을 유지하면서 저녁은 단백질 위주로 가주세요."

        # 유지/증량은 완화
        if total_min < 15:
            return "오늘은 가벼운 활동이라도 15분만 추가해보세요."
        return "오늘 계획을 안정적으로 유지해 주세요."

    def _build_next_meal_ui(self, *, goal_type: str, macro: Dict[str, Any]) -> str:
        # 간단 문구 (DB 거의 비어도 동작)
        if goal_type == "weight_loss":
            return "🍱 다음 끼니: 단백질+채소 중심, 탄수는 소량으로 조절해요."
        if goal_type == "weight_gain":
            return "🍱 다음 끼니: 탄수+단백질을 함께(밥/고구마 + 닭가슴살/계란) 챙겨요."
        return "🍱 다음 끼니: 균형 잡힌 한 끼(탄/단/지 고르게)로 가주세요."

    def _decide_status_ui(self, *, meal_sum: Dict[str, Any], ex_sum: Dict[str, Any]) -> str:
        # 아주 단순한 상태 분류 (UI 배지용)
        total_kcal = meal_sum.get("total_kcal")
        total_min = ex_sum.get("total_minutes") or 0

        if total_kcal is None and total_min == 0:
            return "neutral"

        if total_kcal is not None and total_kcal >= 2200:
            return "partial"

        if total_min >= 30:
            return "applied"

        return "partial"

    # -------------------------------------------------
    # DB Loaders
    # -------------------------------------------------
    def _load_active_goal_row(self, *, user_id: bytes):
        with get_db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT goal_type, kcal_target, macro_target
                FROM user_goal
                WHERE user_id = :uid
                  AND is_active = 'Y'
                """,
                {"uid": user_id},
            )
            return cur.fetchone()

    def _load_daily_meal_summary(self, *, user_id: bytes, day: date) -> Dict[str, Any]:
        with get_db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT total_kcal, carb_g, protein_g, fat_g, daily_score, daily_grade
                FROM daily_meal_summary
                WHERE user_id = :uid
                  AND summary_date = :d
                """,
                {"uid": user_id, "d": day},
            )
            row = cur.fetchone()

        if not row:
            return {"total_kcal": None}

        return {
            "total_kcal": row[0],
            "carb_g": row[1],
            "protein_g": row[2],
            "fat_g": row[3],
            "daily_score": row[4],
            "daily_grade": row[5],
        }

    def _load_daily_exercise_summary(self, *, user_id: bytes, day: date) -> Dict[str, Any]:
        with get_db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT total_minutes, total_met, cardio_minutes, strength_minutes, intensity_level
                FROM daily_exercise_summary
                WHERE user_id = :uid
                  AND summary_date = :d
                """,
                {"uid": user_id, "d": day},
            )
            row = cur.fetchone()

        if not row:
            return {"total_minutes": 0, "cardio_minutes": 0}

        return {
            "total_minutes": row[0] or 0,
            "total_met": row[1],
            "cardio_minutes": row[2] or 0,
            "strength_minutes": row[3] or 0,
            "intensity_level": row[4],
        }
