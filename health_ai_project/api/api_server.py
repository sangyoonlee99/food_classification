# api/api_server.py
from fastapi import FastAPI
from api.endpoints.meal_api import router as meal_router
from services.diet_service import DietService

print("🔥🔥🔥 api_server.py LOADED 🔥🔥🔥")

app = FastAPI(title="Health AI Meal Analyzer")

# ✅ Router 등록 (이게 핵심)
app.include_router(meal_router)

diet_service = DietService()

@app.get("/health")
def health():
    return {"status": "ok"}
