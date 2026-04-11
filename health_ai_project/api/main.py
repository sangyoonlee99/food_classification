# api/main.py
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import date, timedelta

from events.replan_orchestrator import ReplanOrchestrator
from api.ui_card import router as ui_card_router

# -------------------------------------------------
# App
# -------------------------------------------------
app = FastAPI(
    title="Health AI Replan API",
    version="0.1.0",
)

# ✅ 여기서 router 등록
app.include_router(ui_card_router)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = ReplanOrchestrator()
# -------------------------------------------------
# Schemas
# -------------------------------------------------
class ReplanRequest(BaseModel):
    user_id: str
    event_date: date
    event_type: str
    intensity: Optional[str] = None

    # 프론트에서 안 보내면 자동 생성
    base_week_dates: Optional[List[date]] = None

    user_settings: Dict[str, Any]
    event_flags: Dict[str, Any]
    state: Dict[str, Any]

    actual_delta_kcal: Optional[float] = None
    estimated_delta_kcal: float = 400.0


class ReplanResponse(BaseModel):
    meta: Dict[str, Any]
    horizon: Dict[str, Any]
    actions: Dict[str, Any]
    next_day_plan: Optional[Dict[str, Any]] = None
    replan_id: str


# -------------------------------------------------
# Health Check
# -------------------------------------------------
@app.get("/health")
def health():
    """
    서버 상태 확인용
    """
    return {"status": "ok"}


# -------------------------------------------------
# Replan API
# -------------------------------------------------
@app.post("/replan", response_model=ReplanResponse)
async def replan(req: ReplanRequest):
    """
    이벤트 → Horizon → Actions → (선택) Next Day Plan
    단일 진입점
    """

    # 1️⃣ base_week_dates 자동 생성 (UX 개선)
    if not req.base_week_dates:
        req.base_week_dates = [
            req.event_date + timedelta(days=i)
            for i in range(7)
        ]

    # 2️⃣ Orchestrator 호출
    result = orchestrator.build(
        event_date=req.event_date,
        event_type=req.event_type,
        intensity=req.intensity,
        base_week_dates=req.base_week_dates,
        user_settings={**req.user_settings, "user_id": req.user_id},
        event_flags=req.event_flags,
        state=req.state,
        actual_delta_kcal=req.actual_delta_kcal,
        estimated_delta_kcal=req.estimated_delta_kcal,
    )

    return result
