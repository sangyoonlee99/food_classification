# services/event_adjustment_service.py
# STEP 5-4: 이벤트 기반 식단·운동 통합 조정 서비스 (최종)

from common.schemas import (
    Event,
    DailyDietPlan,
    DailyExercisePlan,
    UserGoal,
)
from services.diet_replan_service import ReplanService
from services.exercise_replan_service import ExerciseReplanService
from services.feedback_service import FeedbackService
from services.daily_goal_service import DailyGoalService


class EventAdjustmentService:
    def __init__(self):
        self.diet_replan = ReplanService()
        self.exercise_replan = ExerciseReplanService()
        self.feedback = FeedbackService()
        self.daily_goal = DailyGoalService()

    def adjust_daily(
        self,
        event: Event,
        diet_plan: DailyDietPlan,
        exercise_plan: DailyExercisePlan,
        goal: UserGoal,
    ) -> dict:

        # 1️⃣ 식단 재계획
        new_diet = self.diet_replan.replan_daily(
            event=event,
            daily_plan=diet_plan,
            goal=goal,
        )

        diet_changed = new_diet.total_calorie != diet_plan.total_calorie

        # 2️⃣ 운동 재계획
        replan_result = self.exercise_replan.replan(
            base_plan=exercise_plan.exercises,
            event_type=event.event_type,
            intensity=event.intensity,
            policy=exercise_plan.policy if hasattr(exercise_plan, "policy") else None,
            apply_scope="next_day",
        )

        new_exercises = replan_result.get("exercises", [])

        # 🔑 총 운동 시간 재계산
        total_minutes = sum(e.minutes for e in new_exercises)

        # 🔑 칼로리는 여기서는 비교용으로만 (보수적으로)
        new_total_burn = exercise_plan.total_calorie_burn
        exercise_changed = total_minutes != exercise_plan.total_minutes

        from common.schemas import DailyExercisePlan

        new_exercise = DailyExercisePlan(
            user_id=exercise_plan.user_id,
            date=exercise_plan.date,
            exercises=new_exercises,
            total_minutes=total_minutes,
            total_calorie_burn=new_total_burn,
            cardio_minutes=sum(e.minutes for e in new_exercises if e.category == "cardio"),
            strength_minutes=sum(e.minutes for e in new_exercises if e.category == "strength"),
        )

        # 3️⃣ 이벤트 피드백
        feedback = self.feedback.event_feedback(
            event_type=event.event_type,
            diet_changed=diet_changed,
            exercise_changed=exercise_changed,
        )

        # 4️⃣ ✅ 오늘의 목표 1줄 생성 (핵심 고정 문장)
        daily_goal = None

        return {
            "diet": new_diet,
            "exercise": new_exercise,
            "feedback": feedback,
            "daily_goal": daily_goal,
        }

