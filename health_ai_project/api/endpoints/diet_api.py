from fastapi import APIRouter
from services.diet_service import DietService
from common.schemas import UserProfile, UserGoal

router = APIRouter(prefix="/diet", tags=["Diet"])

diet_service = DietService()


@router.post("/daily")
def generate_daily_diet_api(
    profile: UserProfile,
    goal: UserGoal,
    elderly_mode: bool = False,
):
    """
    실제 서비스용 하루 식단 생성 API
    """
    plan = diet_service.generate_today_diet(
        profile=profile,
        goal=goal,
        elderly_mode=elderly_mode,
    )

    return {
        "status": "success",
        "diet_plan": plan,
    }
