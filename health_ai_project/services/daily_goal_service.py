# services/daily_goal_service.py
# STEP 6-1: 하루 목표 1줄 생성 서비스 (Anchor Sentence)

from common.schemas import (
    DailyDietPlan,
    DailyExercisePlan,
    UserGoal,
    Event,
)

from infra.db_server import get_db_conn


class DailyGoalService:
    """
    하루의 모든 추천/가이드의 기준이 되는
    '하루 목표 1줄(anchor sentence)' 생성
    """

    # ==================================================
    # 🔹 UI / Loader 진입용 (현재 단계용)
    # ==================================================
    def build_today(self, *, user_id: bytes) -> str | None:
        """
        DB 기준으로 오늘 목표 1줄 반환
        (diet / exercise / event 미연결 단계)
        """
        with get_db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT goal_type
                FROM user_goal
                WHERE user_id = :user_id
                  AND is_active = 'Y'
                """,
                {"user_id": user_id},
            )
            row = cur.fetchone()

        if not row:
            return None

        goal_type = row[0]

        if goal_type == "weight_loss":
            return "🎯 오늘은 체중 감량을 목표로 관리합니다."
        elif goal_type in ("muscle_gain", "weight_gain"):
            return "🎯 오늘은 체중 증량을 목표로 관리합니다."
        else:
            return "🎯 오늘은 현재 컨디션 유지를 목표로 합니다."

    # ==================================================
    # 🔹 Domain 정식 로직 (STEP 6-1 핵심)
    # ==================================================
    def build_daily_goal(
        self,
        goal: UserGoal,
        diet: DailyDietPlan,
        exercise: DailyExercisePlan,
        event: Event | None = None,
    ) -> str:

        # 1️⃣ 목표 문장
        if goal.goal_type == "weight_loss":
            base = "체중 감량 목표를 유지합니다."
        elif goal.goal_type == "muscle_gain":
            base = "근육 증가를 위한 하루입니다."
        else:
            base = "균형 잡힌 컨디션 유지를 목표로 합니다."

        # 2️⃣ 이벤트 보정
        if event:
            if event.event_type == "dinner_drinking":
                return f"{base} 회식 이후 회복에 집중하는 하루입니다."
            if event.event_type == "overeating":
                return f"{base} 과식 이후 조절에 집중하는 하루입니다."
            if event.event_type == "travel":
                return f"{base} 여행 중 유지 중심으로 관리합니다."

        # 3️⃣ 섭취/소모 기반 보정
        if diet.total_calorie < 1400:
            return f"{base} 에너지 보충을 우선합니다."

        if exercise.total_calorie_burn > 400:
            return f"{base} 활동량이 높은 날입니다."

        # 4️⃣ 기본 반환
        return base
