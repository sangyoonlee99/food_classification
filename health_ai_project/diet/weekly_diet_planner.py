# diet/weekly_diet_planner.py
from datetime import timedelta, date
from typing import Dict, List

from common.schemas import WeeklyDietPlan, UserProfile, UserGoal
from diet.daily_diet_planner import generate_daily_diet


MAX_DAILY_RETRY = 3


def generate_weekly_diet(
    profile: UserProfile,
    goal: UserGoal,
    week_start: date,
    elderly_mode: bool = False,
) -> WeeklyDietPlan:
    weekly_days: List = []
    weekly_food_usage: Dict[str, int] = {}

    for i in range(7):
        current_date = week_start + timedelta(days=i)
        daily_plan = None

        for attempt in range(MAX_DAILY_RETRY):
            try:
                daily_plan = generate_daily_diet(
                    profile=profile,
                    goal=goal,
                    date_value=current_date,
                    elderly_mode=elderly_mode,
                    used_foods=weekly_food_usage,
                )
                break
            except ValueError as e:
                if attempt == MAX_DAILY_RETRY - 1:
                    raise RuntimeError(
                        f"Failed to generate daily diet for {current_date}: {e}"
                    )

        weekly_days.append(daily_plan)

    return WeeklyDietPlan(
        user_id=profile.user_id,
        week_start=week_start,
        week_end=week_start + timedelta(days=6),
        days=weekly_days,
    )
