# ui_streamlit/adapters/service_api.py
from __future__ import annotations

from typing import Any, Dict, List
from datetime import date, timedelta
import streamlit as st

from infra.db_server import get_db_conn

# (옵션) RAW(16) 바인딩 안전용
try:
    import oracledb  # type: ignore
except Exception:  # pragma: no cover
    oracledb = None


# ==================================================
# Utils
# ==================================================
def _week_range(d: date):
    start = d - timedelta(days=d.weekday())
    end = start + timedelta(days=7)
    return start, end


def _uid():
    return st.session_state.get("user_id")


def _bind_uid(user_id: bytes):
    if oracledb is not None:
        return oracledb.Binary(user_id)
    return user_id


# ==================================================
# DB Aggregates (기존 ReportService 용)
# ==================================================
def _daily_meal_macro(conn, user_id: bytes, target_dt: date):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            NVL(SUM(carb_g), 0),
            NVL(SUM(protein_g), 0),
            NVL(SUM(fat_g), 0)
        FROM meal_record
        WHERE user_id = :p_uid
          AND TRUNC(eaten_at) = :p_dt
        """,
        {
            "p_uid": _bind_uid(user_id),
            "p_dt": target_dt,
        },
    )
    carb, protein, fat = cur.fetchone()
    return {
        "carb": float(carb or 0),
        "protein": float(protein or 0),
        "fat": float(fat or 0),
    }


def _weekly_exercise(conn, user_id: bytes, start: date, end: date):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT exercise_type, SUM(minutes)
        FROM exercise_record
        WHERE user_id = :p_uid
          AND performed_at >= :p_start
          AND performed_at <  :p_end
        GROUP BY exercise_type
        """,
        {
            "p_uid": _bind_uid(user_id),
            "p_start": start,
            "p_end": end,
        },
    )

    total = cardio = strength = 0
    for ex_type, minutes in cur.fetchall():
        m = int(minutes or 0)
        total += m
        if ex_type in ("걷기", "조깅", "러닝", "자전거"):
            cardio += m
        else:
            strength += m

    return {
        "total_minutes": total,
        "cardio_minutes": cardio,
        "strength_minutes": strength,
    }


# ==================================================
# Score Rules (기존 ReportService 용)
# ==================================================
def _calc_daily_score(macro: Dict[str, float]) -> int:
    score = 0

    if macro["protein"] >= 20:
        score += 40
    elif macro["protein"] >= 10:
        score += 25

    if macro["carb"] <= 80:
        score += 30
    elif macro["carb"] <= 120:
        score += 15

    if macro["fat"] <= 30:
        score += 30
    elif macro["fat"] <= 50:
        score += 15

    return min(score, 100)


def _daily_score_comment(macro: Dict[str, float], score: int) -> str:
    msgs = []

    if macro["protein"] >= 20:
        msgs.append("단백질 섭취가 매우 좋아요")
    elif macro["protein"] >= 10:
        msgs.append("단백질 섭취가 적절해요")
    else:
        msgs.append("단백질 섭취가 부족해요")

    if macro["carb"] <= 80:
        msgs.append("탄수화물 조절이 잘 되었어요")
    elif macro["carb"] <= 120:
        msgs.append("탄수화물 섭취가 보통 수준이에요")
    else:
        msgs.append("탄수화물 섭취가 조금 많아요")

    if macro["fat"] <= 30:
        msgs.append("지방 섭취가 안정적이에요")
    else:
        msgs.append("지방 섭취가 다소 높아요")

    if score >= 90:
        return " · ".join(msgs) + " 👍"
    elif score >= 70:
        return " · ".join(msgs)
    else:
        return "기본 기록은 되었어요. 조금만 더 조절해볼까요?"


def _calc_weekly_score(exercise: Dict[str, int]) -> int:
    score = 0

    total = exercise.get("total_minutes", 0)
    cardio = exercise.get("cardio_minutes", 0)
    strength = exercise.get("strength_minutes", 0)

    if total >= 150:
        score += 50
    elif total >= 90:
        score += 35
    elif total >= 60:
        score += 20

    if cardio >= 60:
        score += 25
    if strength >= 60:
        score += 25

    return min(score, 100)


# ==================================================
# UI Services (기존)
# ==================================================
class ReportService:
    def get_daily_report(self, target_dt: date) -> Dict[str, Any]:
        user_id = _uid()
        if not user_id:
            return {}

        with get_db_conn() as conn:
            macro = _daily_meal_macro(conn, user_id, target_dt)

        score = _calc_daily_score(macro)
        comment = _daily_score_comment(macro, score)

        return {
            "score": score,
            "comment": comment,
            "macro": macro,
            "meals": [],
        }


def get_weekly_report(user_id: int) -> Dict[str, Any]:
    user_id = _uid()
    if not user_id:
        return {"weekly": {}}

    today = date.today()
    start, end = _week_range(today)

    with get_db_conn() as conn:
        exercise = _weekly_exercise(conn, user_id, start, end)

    score = _calc_weekly_score(exercise)

    return {
        "weekly": {
            "score": score,
            "exercise": exercise,
            "meals": {"carb": 0.0, "protein": 0.0, "fat": 0.0},
        }
    }


# ==================================================
# ✅ NEW: Routine / Event (routine.py에서 필요)
# ==================================================
class RoutineService:
    """
    routine 화면에서 바로 쓰는 최소 계약:
    {
      "meals": { "아침": [{"food_name":..., "grams":...}, ...], ... },
      "exercise": [{"name":..., "minutes":..., "intensity":...}, ...],
    }
    """
    def get_daily_routine(self, *, date: date) -> Dict[str, Any]:
        user_id = _uid()
        if not user_id:
            return {}

        meals: Dict[str, List[Dict[str, Any]]] = {}
        exercise: List[Dict[str, Any]] = []

        with get_db_conn() as conn:
            # meal_record: eaten_at, meal_type, food_name, amount_g
            cur = conn.cursor()
            cur.execute(
                """
                SELECT
                    NVL(meal_type, '식사') AS meal_type,
                    NVL(food_name, '')    AS food_name,
                    NVL(amount_g, 0)      AS amount_g
                FROM meal_record
                WHERE user_id = :p_uid
                  AND TRUNC(eaten_at) = :p_dt
                ORDER BY eaten_at ASC
                """,
                {"p_uid": _bind_uid(user_id), "p_dt": date},
            )
            for meal_type, food_name, amount_g in cur.fetchall():
                meals.setdefault(str(meal_type), []).append(
                    {"food_name": str(food_name), "grams": float(amount_g or 0)}
                )

            # exercise_record: performed_at, exercise_type, minutes, intensity
            cur = conn.cursor()
            cur.execute(
                """
                SELECT
                    NVL(exercise_type, '') AS exercise_type,
                    NVL(minutes, 0)        AS minutes,
                    NVL(intensity, '중간') AS intensity
                FROM exercise_record
                WHERE user_id = :p_uid
                  AND TRUNC(performed_at) = :p_dt
                ORDER BY performed_at ASC
                """,
                {"p_uid": _bind_uid(user_id), "p_dt": date},
            )
            for ex_type, minutes, intensity in cur.fetchall():
                exercise.append(
                    {
                        "name": str(ex_type),
                        "minutes": int(minutes or 0),
                        "intensity": str(intensity),
                    }
                )

        return {"meals": meals, "exercise": exercise}


class EventService:
    def get_events(self, *, date: date) -> List[Dict[str, Any]]:
        user_id = _uid()
        if not user_id:
            return []

        with get_db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT event_type, severity, note
                FROM event_log
                WHERE user_id = :p_uid
                  AND event_date = :p_dt
                ORDER BY created_at DESC
                """,
                {"p_uid": _bind_uid(user_id), "p_dt": date},
            )
            rows = cur.fetchall()

        out: List[Dict[str, Any]] = []
        for event_type, severity, note in rows:
            out.append(
                {
                    "event_type": str(event_type or ""),
                    "severity": str(severity or "normal"),
                    "note": str(note or ""),
                }
            )
        return out


# ==================================================
# Service Instances (UI에서 import 하는 것들)
# ==================================================
report_service = ReportService()
routine_service = RoutineService()
event_service = EventService()
