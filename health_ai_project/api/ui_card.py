from fastapi import APIRouter, Depends, HTTPException
from infra.db_server import get_db_conn
from services.ui_card_service import get_today_ui_card
import traceback

router = APIRouter()

@router.get("/ui/cards/today")
def ui_card_today(user_id: str, conn = Depends(get_db_conn)):
    try:
        card = get_today_ui_card(
            conn=conn,
            user_id=bytes.fromhex(user_id)
        )
        return {"exists": card is not None, "card": card}

    except Exception as e:
        print("🔥 UI CARD ERROR 🔥")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

