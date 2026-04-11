from __future__ import annotations

print("🔥🔥🔥 services/exercise_service.py LOADED 🔥🔥🔥")

from typing import Dict, Any
from datetime import date, datetime
import uuid

from infra.db_server import get_db_conn

# =====================================================
# Constants
# =====================================================
POLICY_MICRO = "micro"
POLICY_MESO = "meso"
POLICY_MACRO = "macro"

CARDIO_TYPES = {"걷기", "조깅", "러닝", "자전거"}

# =====================================================
# Event Weight Table
# =====================================================
EVENT_EXERCISE_WEIGHTS = {
    "overtime": {
        "intensity_cap": "keep",
        "max_cardio": 10,
    },
    "travel": {
        "allow_macro": True,
        "suppress_micro": True,
    },
    "sick": {
        "disable_exercise": True,
    },
}

# =====================================================
# Policy Decision
# =====================================================
def decide_exercise_policy(
    event_flags: Dict[str, Any],
    state: Dict[str, Any],
) -> Dict[str, Any]:

    decision = {
        "level": None,
        "micro": {},
        "meso": {},
        "macro": {},
    }

    plateau_days = state.get("plateau_days", 0)
    event_type = event_flags.get("event_type")
    weight = EVENT_EXERCISE_WEIGHTS.get(event_type)

    if weight and weight.get("disable_exercise"):
        return decision

    # ---------------- MICRO ----------------
    micro: Dict[str, Any] = {}

    if plateau_days >= 7 and event_type is None:
        micro.setdefault("adjust_minutes", {})
        micro["adjust_minutes"]["cardio"] = 10
        micro["intensity"] = "keep"

    if event_flags.get("activity_drop"):
        micro.setdefault("adjust_minutes", {})
        micro["adjust_minutes"]["cardio"] = micro["adjust_minutes"].get("cardio", 0) + 10

    if event_flags.get("is_kcal_over"):
        micro.setdefault("adjust_minutes", {})
        micro["adjust_minutes"]["cardio"] = micro["adjust_minutes"].get("cardio", 0) + 10

    if plateau_days >= 7:
        micro["intensity"] = "up"

    if weight:
        if "max_cardio" in weight and "adjust_minutes" in micro:
            micro["adjust_minutes"]["cardio"] = min(
                micro["adjust_minutes"].get("cardio", 0),
                weight["max_cardio"],
            )
        if weight.get("intensity_cap") == "keep":
            micro["intensity"] = "keep"
        if weight.get("suppress_micro"):
            micro = {}

    if micro:
        decision["micro"] = micro
        decision["level"] = POLICY_MICRO

    # ---------------- MESO ----------------
    if plateau_days >= 10:
        decision["meso"] = {
            "weekly_freq_delta": +1,
            "volume_rebalance": True,
        }
        decision["level"] = POLICY_MESO

    # ---------------- MACRO ----------------
    allow_macro = not (weight and weight.get("allow_macro") is False)

    if plateau_days >= 14 and allow_macro:
        decision["macro"] = {"change_routine": True}
        decision["level"] = POLICY_MACRO

    return decision


def to_exercise_action(decision: Dict[str, Any]) -> Dict[str, Any]:
    action = {
        "adjust_minutes": {},
        "change_routine": False,
        "intensity": "keep",
    }

    micro = decision.get("micro", {})
    if micro:
        action["adjust_minutes"] = micro.get("adjust_minutes", {})
        action["intensity"] = micro.get("intensity", "keep")

    if decision.get("macro", {}).get("change_routine"):
        action["change_routine"] = True

    return action


def decide_exercise_actions(
    event_flags: Dict[str, Any],
    state: Dict[str, Any],
) -> Dict[str, Any]:
    return to_exercise_action(decide_exercise_policy(event_flags, state))


# =====================================================
# Exercise Service (UI Entry Point)
# =====================================================
class ExerciseService:
    def record_exercise(
        self,
        *,
        user_id: bytes,
        exercise_type: str,
        minutes: int,
        intensity: str,
        performed_at: datetime,
        source: str = "manual",
    ) -> Dict[str, Any]:

        met = self._estimate_met(exercise_type)

        insert_exercise_record(
            user_id=user_id,
            exercise_type=exercise_type,
            minutes=minutes,
            intensity=intensity,
            met=met,
            source=source,
            performed_at=performed_at,
        )

        return aggregate_daily_exercise(
            user_id=user_id,
            target_date=performed_at.date(),
        )

    def _estimate_met(self, exercise_type: str) -> float:
        return {
            "걷기": 3.3,
            "조깅": 7.0,
            "러닝": 9.8,
            "자전거": 6.8,
        }.get(exercise_type, 4.0)


# =====================================================
# DB Insert
# =====================================================
def insert_exercise_record(
    *,
    user_id: bytes,
    exercise_type: str,
    minutes: int,
    intensity: str,
    met: float,
    source: str,
    performed_at: datetime,
):
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO exercise_record (
                exercise_id,
                user_id,
                performed_at,
                exercise_type,
                minutes,
                intensity,
                source,
                met
            )
            VALUES (
                :exercise_id,
                :user_id,
                :performed_at,
                :exercise_type,
                :minutes,
                :intensity,
                :source,
                :met
            )
            """,
            {
                "exercise_id": uuid.uuid4().bytes,
                "user_id": user_id,
                "performed_at": performed_at,
                "exercise_type": exercise_type,
                "minutes": minutes,
                "intensity": intensity,
                "source": source,
                "met": met,
            },
        )
        conn.commit()


# =====================================================
# Daily Aggregate (E2E Core)
# =====================================================
def aggregate_daily_exercise(*, user_id: bytes, target_date: date) -> Dict[str, Any] | None:
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT minutes, met, exercise_type
            FROM exercise_record
            WHERE user_id = :user_id
              AND TRUNC(performed_at) = :d
            """,
            {"user_id": user_id, "d": target_date},
        )
        rows = cur.fetchall()

        # ✅ 운동 기록이 하나도 없으면 → 집계 테이블 삭제 (핵심)
        if not rows:
            cur.execute(
                """
                DELETE FROM daily_exercise_summary
                WHERE user_id = :u
                  AND summary_date = :d
                """,
                {"u": user_id, "d": target_date},
            )
            conn.commit()
            return None

        total_minutes = 0
        total_met_minutes = 0.0
        cardio = 0
        strength = 0

        for minutes, met, ex_type in rows:
            m = int(minutes or 0)
            total_minutes += m
            total_met_minutes += float(met or 0.0) * m

            if ex_type in CARDIO_TYPES:
                cardio += m
            else:
                strength += m

        if total_minutes >= 60:
            intensity_level = "high"
        elif total_minutes >= 30:
            intensity_level = "normal"
        else:
            intensity_level = "low"

        cur.execute(
            """
            MERGE INTO daily_exercise_summary d
            USING dual
            ON (
                d.user_id = :user_id
                AND d.summary_date = :summary_date
            )
            WHEN MATCHED THEN UPDATE SET
                total_minutes     = :tm,
                total_met         = :tmet,
                cardio_minutes    = :cardio,
                strength_minutes  = :strength,
                intensity_level   = :intensity
            WHEN NOT MATCHED THEN INSERT
                (
                    user_id,
                    summary_date,
                    total_minutes,
                    total_met,
                    cardio_minutes,
                    strength_minutes,
                    intensity_level
                )
            VALUES
                (
                    :user_id,
                    :summary_date,
                    :tm,
                    :tmet,
                    :cardio,
                    :strength,
                    :intensity
                )
            """,
            {
                "user_id": user_id,
                "summary_date": target_date,
                "tm": total_minutes,
                "tmet": round(total_met_minutes, 2),
                "cardio": cardio,
                "strength": strength,
                "intensity": intensity_level,
            },
        )
        conn.commit()

        return {
            "total_minutes": total_minutes,
            "total_met": round(total_met_minutes, 2),
            "cardio_minutes": cardio,
            "strength_minutes": strength,
            "intensity_level": intensity_level,
        }


# =====================================================
# Public Query Helper (A단계 핵심)
# =====================================================
def get_today_exercise_summary(*, user_id: bytes) -> Dict[str, Any] | None:
    today = date.today()

    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                total_minutes,
                cardio_minutes,
                strength_minutes,
                intensity_level
            FROM daily_exercise_summary
            WHERE user_id = :u
              AND summary_date = :d
            """,
            {"u": user_id, "d": today},
        )
        row = cur.fetchone()

    if not row:
        return None

    total, cardio, strength, intensity = row
    return {
        "total_minutes": int(total or 0),
        "cardio_minutes": int(cardio or 0),
        "strength_minutes": int(strength or 0),
        "intensity_level": (intensity or "low"),
    }

# =====================================================
# Delete Exercise Record (B단계 핵심)
# =====================================================
def delete_exercise_record(
    *,
    user_id: bytes,
    exercise_id: bytes,
    performed_at: datetime,
) -> None:
    target_date = performed_at.date()

    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            DELETE FROM exercise_record
            WHERE exercise_id = :p_exercise_id
              AND user_id = :p_user_id
            """,
            {
                "p_exercise_id": exercise_id,  # RAW(16) ← bytes
                "p_user_id": user_id,          # RAW ← bytes
            },
        )
        conn.commit()

    aggregate_daily_exercise(
        user_id=user_id,
        target_date=target_date,
    )


# =====================================================
# Update Exercise Record (C-1)
# =====================================================
def update_exercise_record(
    *,
    user_id: bytes,
    exercise_id: bytes,
    minutes: int,
    intensity: str,
    performed_at: datetime,
) -> None:
    target_date = performed_at.date()

    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE exercise_record
            SET
                minutes   = :p_minutes,
                intensity = :p_intensity
            WHERE exercise_id = :p_exercise_id
              AND user_id     = :p_user_id
            """,
            {
                "p_minutes": minutes,
                "p_intensity": intensity,
                "p_exercise_id": exercise_id,  # RAW(16)
                "p_user_id": user_id,          # RAW
            },
        )
        conn.commit()

    # 🔁 재집계
    aggregate_daily_exercise(
        user_id=user_id,
        target_date=target_date,
    )

# =====================================================
# Weekly Exercise Report (C-2)
# =====================================================
from datetime import timedelta

def get_weekly_exercise_report(
    *,
    user_id: bytes,
    end_date: date | None = None,
) -> dict:
    if end_date is None:
        end_date = date.today()

    start_date = end_date - timedelta(days=6)

    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                summary_date,
                total_minutes,
                cardio_minutes,
                strength_minutes,
                intensity_level
            FROM daily_exercise_summary
            WHERE user_id = :u
              AND summary_date BETWEEN :s AND :e
            ORDER BY summary_date
            """,
            {"u": user_id, "s": start_date, "e": end_date},
        )
        rows = cur.fetchall()

    total = cardio = strength = 0
    intensity_count = {"low": 0, "normal": 0, "high": 0}

    for _, tm, c, s, intensity in rows:
        total += int(tm or 0)
        cardio += int(c or 0)
        strength += int(s or 0)
        if intensity:
            intensity_count[intensity] = intensity_count.get(intensity, 0) + 1

    return {
        "range": f"{start_date.strftime('%m/%d')} ~ {end_date.strftime('%m/%d')}",
        "total_minutes": total,
        "cardio_minutes": cardio,
        "strength_minutes": strength,
        "intensity_days": intensity_count,
        "days": len(rows),
    }
