# services/meal_service.py
from __future__ import annotations

from datetime import datetime, date
from pathlib import Path
import uuid
import json
from typing import Optional, Dict, Any
from functools import lru_cache

import pandas as pd  # ✅ 추가

from config import BASE_DIR
from infra.db_server import get_db_conn

from common.schemas import (
    MealRecord,
    MealContext,
    MealEvaluation,
    DailyMealSummary,
    NutritionSummary,
)

# =========================================================
# Excel Nutrition DB
# =========================================================
NUTRITION_XLSX_PATH = BASE_DIR / "data" / "food_nutrition.xlsx"  # ✅ 필요시 경로만 맞추세요


def _norm_food_name(s: str) -> str:
    return (s or "").strip().replace("\u3000", " ").replace("\xa0", " ")


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    # 느슨 매칭
    for col in cols:
        low = str(col).strip().lower()
        for c in candidates:
            if str(c).strip().lower() in low:
                return col
    return None


@lru_cache(maxsize=1)
def _load_nutrition_df() -> pd.DataFrame:
    if not NUTRITION_XLSX_PATH.exists():
        raise FileNotFoundError(f"food_nutrition.xlsx not found: {NUTRITION_XLSX_PATH}")

    df = pd.read_excel(NUTRITION_XLSX_PATH)
    df = df.copy()

    # 컬럼명 표준화(엑셀 실제 컬럼 기준)
    col_name = _pick_col(df, ["음식명", "food_name", "name"])
    col_w = _pick_col(df, ["중량(g)", "중량", "1회제공량", "serving_g", "portion_g", "weight_g"])
    col_k = _pick_col(df, ["에너지(kcal)", "에너지", "kcal", "칼로리"])
    col_c = _pick_col(df, ["탄수화물(g)", "탄수화물", "carb", "carbs"])
    col_p = _pick_col(df, ["단백질(g)", "단백질", "protein"])
    col_f = _pick_col(df, ["지방(g)", "지방", "fat"])

    needed = {"name": col_name, "w": col_w, "k": col_k, "c": col_c, "p": col_p, "f": col_f}
    if not all(needed.values()):
        raise ValueError(f"nutrition xlsx columns not found. detected={needed}")

    df = df.rename(
        columns={
            col_name: "_food_name",
            col_w: "_serving_g",
            col_k: "_kcal",
            col_c: "_carb_g",
            col_p: "_protein_g",
            col_f: "_fat_g",
        }
    )

    df["_food_name"] = df["_food_name"].astype(str).map(_norm_food_name)
    # 숫자형 정리
    for c in ["_serving_g", "_kcal", "_carb_g", "_protein_g", "_fat_g"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # key lookup 용
    df["_key"] = df["_food_name"].str.replace(" ", "", regex=False).str.lower()

    return df


def _lookup_excel(food_name: str) -> dict | None:
    name = _norm_food_name(food_name)
    if not name:
        return None

    df = _load_nutrition_df()
    key = name.replace(" ", "").lower()

    hit = df[df["_key"] == key]
    if hit.empty:
        return None

    r = hit.iloc[0]
    return {
        "food_name": str(r["_food_name"]),
        "serving_g": float(r["_serving_g"] or 0.0),
        "kcal": float(r["_kcal"] or 0.0),
        "carbs_g": float(r["_carb_g"] or 0.0),
        "protein_g": float(r["_protein_g"] or 0.0),
        "fat_g": float(r["_fat_g"] or 0.0),
    }


def get_default_serving_g(food_name: str) -> int | None:
    row = _lookup_excel(food_name)
    if not row:
        return None
    g = int(round(float(row["serving_g"] or 0)))
    return g if g > 0 else None


def _calc_nutrition(food_name: str, amount_g: float) -> Dict[str, float]:
    """
    ✅ 최종: 엑셀(1회 제공량) 기반으로 계산
    - amount_g = 사용자가 입력한 g
    - 엑셀 row: serving_g 기준 kcal/c/p/f → 1g당 환산 → amount_g로 스케일링
    """
    g = float(amount_g or 0.0)
    if g <= 0:
        return {"kcal": 0.0, "carbs_g": 0.0, "protein_g": 0.0, "fat_g": 0.0}

    row = _lookup_excel(food_name)
    if not row:
        # 엑셀에서 못 찾으면(임시 fallback)
        return {
            "kcal": round(g * 2.0, 1),
            "carbs_g": round(g * 0.25, 1),
            "protein_g": round(g * 0.15, 1),
            "fat_g": round(g * 0.06, 1),
        }

    serving_g = float(row["serving_g"] or 0.0)
    if serving_g <= 0:
        # serving_g가 없으면 fallback
        return {
            "kcal": round(g * 2.0, 1),
            "carbs_g": round(g * 0.25, 1),
            "protein_g": round(g * 0.15, 1),
            "fat_g": round(g * 0.06, 1),
        }

    # 1g당 환산
    kcal_pg = float(row["kcal"]) / serving_g
    carb_pg = float(row["carbs_g"]) / serving_g
    prot_pg = float(row["protein_g"]) / serving_g
    fat_pg = float(row["fat_g"]) / serving_g

    return {
        "kcal": round(kcal_pg * g, 1),
        "carbs_g": round(carb_pg * g, 1),
        "protein_g": round(prot_pg * g, 1),
        "fat_g": round(fat_pg * g, 1),
    }


# =========================================================
# RAW(16) helpers
# =========================================================
def _ensure_bytes16(user_id: bytes) -> bytes:
    if not isinstance(user_id, (bytes, bytearray)):
        raise TypeError("user_id must be bytes(16)")
    b = bytes(user_id)
    if len(b) != 16:
        raise ValueError(f"user_id must be 16 bytes (RAW16). got len={len(b)}")
    return b


def _ensure_raw16(v) -> bytes:
    if isinstance(v, uuid.UUID):
        return v.bytes
    if isinstance(v, (bytes, bytearray)):
        b = bytes(v)
        if len(b) == 16:
            return b
    if isinstance(v, str):
        s = v.strip()
        if s.lower().startswith("0x"):
            s = s[2:]
        if len(s) == 32:
            try:
                b = bytes.fromhex(s)
                if len(b) == 16:
                    return b
            except Exception:
                pass
    return uuid.uuid4().bytes


# =========================================================
# json helpers
# =========================================================
def _json_default(o):
    if isinstance(o, (bytes, bytearray)):
        return bytes(o).hex()
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    return str(o)


def _coerce_bytes(v):
    if isinstance(v, (bytes, bytearray)):
        return _ensure_bytes16(bytes(v))
    if isinstance(v, str):
        return _ensure_bytes16(bytes.fromhex(v))
    raise TypeError(f"cannot coerce to bytes: {type(v)}")


def _coerce_date(v) -> date:
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    if isinstance(v, str):
        return date.fromisoformat(v[:10])
    raise TypeError(f"cannot coerce to date: {type(v)}")


def _coerce_datetime(v) -> datetime:
    if isinstance(v, datetime):
        return v
    if isinstance(v, str):
        return datetime.fromisoformat(v)
    raise TypeError(f"cannot coerce to datetime: {type(v)}")


def _load_daily_json(path: Path) -> DailyMealSummary:
    raw = json.loads(path.read_text("utf-8"))
    raw["user_id"] = _coerce_bytes(raw["user_id"])
    raw["date"] = _coerce_date(raw["date"])

    meals = raw.get("meals", [])
    for m in meals:
        m["user_id"] = _coerce_bytes(m["user_id"])
        m["created_at"] = _coerce_datetime(m["created_at"])
    return DailyMealSummary(**raw)


def _infer_meal_type_from_time(dt: datetime) -> str:
    hh = int(getattr(dt, "hour", 0))
    if 5 <= hh <= 10:
        return "breakfast"
    if 11 <= hh <= 15:
        return "lunch"
    if 16 <= hh <= 18:
        return "snack"
    return "dinner"


# =========================================================
# Service
# =========================================================
class MealService:
    def record_meal(
        self,
        *,
        user_id: bytes,
        food_name: str,
        amount_g: float,
        eaten_at: datetime,
        source: str | None = None,
    ) -> None:
        user_id = _ensure_bytes16(user_id)

        nutrition = _calc_nutrition(food_name, amount_g)  # ✅ 변경

        meal = MealRecord(
            meal_id=uuid.uuid4().hex.upper(),
            user_id=user_id,
            image=None,
            created_at=eaten_at,
            primary_food={"food_name": food_name, "amount_g": float(amount_g)},
            foods=[],
            nutrition_summary=NutritionSummary(total=nutrition, items_count=1),
            meal_evaluation=MealEvaluation(meal_score=0, grade="미평가", flags={}, advice=[]),
            context=MealContext(meal_type="manual", source=source),
        )

        MealLogger().save_meal(meal)

    def update_meal(
        self,
        *,
        user_id: bytes,
        meal_id_raw: bytes,
        new_meal_type: str,
        new_food_name: str,
        new_amount_g: float,
        eaten_at: Optional[datetime] = None,
    ) -> None:
        user_id = _ensure_bytes16(user_id)
        MealLogger().update_meal_record(
            user_id=user_id,
            meal_id_raw=_ensure_raw16(meal_id_raw),
            new_meal_type=new_meal_type,
            new_food_name=new_food_name,
            new_amount_g=float(new_amount_g),
            eaten_at=eaten_at,
        )

    def delete_meal(self, *, user_id: bytes, meal_id_raw: bytes) -> None:
        user_id = _ensure_bytes16(user_id)
        MealLogger().delete_meal_record(user_id=user_id, meal_id_raw=_ensure_raw16(meal_id_raw))

    def decide_actions(self, *, user_id: bytes, date: date):
        return self.build_meal_actions(user_id=user_id, date=date)

    def build_meal_actions(self, *, user_id: bytes, date: date):
        user_id = _ensure_bytes16(user_id)

        with get_db_conn() as conn:
            cur = conn.cursor()

            cur.execute(
                """
                SELECT NVL(SUM(kcal), 0)
                FROM meal_record
                WHERE user_id = :user_id
                  AND TRUNC(eaten_at) = :d
                """,
                {"user_id": user_id, "d": date},
            )
            total_kcal = float(cur.fetchone()[0] or 0)

            cur.execute(
                """
                SELECT kcal_target
                FROM user_goal
                WHERE user_id = :user_id
                  AND is_active = 'Y'
                """,
                {"user_id": user_id},
            )
            row = cur.fetchone()
            target_kcal = float(row[0]) if row and row[0] is not None else None

        if target_kcal:
            if total_kcal > target_kcal * 1.15:
                return {"adjust_grams": False, "change_menu": True}
            if total_kcal > target_kcal * 1.05:
                return {"adjust_grams": True, "change_menu": False}

        return {"adjust_grams": False, "change_menu": False}


# =========================================================
# Logger
# =========================================================
class MealLogger:
    def __init__(self):
        self.base_dir = BASE_DIR / "data" / "meal_logs"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_meal(self, meal: MealRecord):
        user_id: bytes = _ensure_bytes16(meal.user_id)
        meal_date = meal.created_at.date()

        amount_g = 0.0
        try:
            amount_g = float((meal.primary_food or {}).get("amount_g", 0) or 0)
        except Exception:
            amount_g = 0.0

        meal_id_raw = _ensure_raw16(getattr(meal, "meal_id", None))

        meal_type = None
        try:
            meal_type = getattr(getattr(meal, "context", None), "source", None)
        except Exception:
            meal_type = None
        meal_type = (meal_type or "").strip() or _infer_meal_type_from_time(meal.created_at)

        with get_db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO meal_record (
                    meal_id,
                    user_id,
                    eaten_at,
                    meal_type,
                    food_name,
                    amount_g,
                    kcal,
                    carb_g,
                    protein_g,
                    fat_g
                )
                VALUES (
                    :meal_id,
                    :user_id,
                    :eaten_at,
                    :meal_type,
                    :food_name,
                    :amount_g,
                    :kcal,
                    :carb,
                    :protein,
                    :fat
                )
                """,
                {
                    "meal_id": meal_id_raw,
                    "user_id": user_id,
                    "eaten_at": meal.created_at,
                    "meal_type": meal_type,
                    "food_name": (meal.primary_food or {}).get("food_name", ""),
                    "amount_g": amount_g,
                    "kcal": float(meal.nutrition_summary.total.get("kcal", 0)),
                    "carb": float(meal.nutrition_summary.total.get("carbs_g", 0)),
                    "protein": float(meal.nutrition_summary.total.get("protein_g", 0)),
                    "fat": float(meal.nutrition_summary.total.get("fat_g", 0)),
                },
            )
            conn.commit()

        # 이하(요약/업서트)는 기존 코드 그대로 두셔도 됩니다.
        user_hex = user_id.hex()
        user_dir = self.base_dir / f"user_{user_hex}"
        user_dir.mkdir(parents=True, exist_ok=True)
        path = user_dir / f"{meal_date}.json"

        if path.exists():
            try:
                daily = _load_daily_json(path)
            except Exception:
                daily = self._new_daily(user_id, meal_date)
        else:
            daily = self._new_daily(user_id, meal_date)

        daily.meals.append(meal)
        self._recalculate_daily(daily)

        json_str = json.dumps(daily.model_dump(), ensure_ascii=False, indent=2, default=_json_default)
        path.write_text(json_str, encoding="utf-8")

        self._upsert_daily_meal_summary(daily)

    def update_meal_record(
        self,
        *,
        user_id: bytes,
        meal_id_raw: bytes,
        new_meal_type: str,
        new_food_name: str,
        new_amount_g: float,
        eaten_at: Optional[datetime] = None,
    ) -> None:
        user_id = _ensure_bytes16(user_id)
        meal_id_raw = _ensure_raw16(meal_id_raw)

        nut = _calc_nutrition(new_food_name, new_amount_g)  # ✅ 변경

        with get_db_conn() as conn:
            cur = conn.cursor()
            if eaten_at is None:
                cur.execute(
                    """
                    UPDATE meal_record
                    SET
                        meal_type = :meal_type,
                        food_name = :food_name,
                        amount_g  = :amount_g,
                        kcal      = :kcal,
                        carb_g    = :carb_g,
                        protein_g = :protein_g,
                        fat_g     = :fat_g
                    WHERE user_id = :user_id
                      AND meal_id = :meal_id
                    """,
                    {
                        "meal_type": (new_meal_type or "").strip(),
                        "food_name": (new_food_name or "").strip(),
                        "amount_g": float(new_amount_g),
                        "kcal": float(nut["kcal"]),
                        "carb_g": float(nut["carbs_g"]),
                        "protein_g": float(nut["protein_g"]),
                        "fat_g": float(nut["fat_g"]),
                        "user_id": user_id,
                        "meal_id": meal_id_raw,
                    },
                )
            else:
                cur.execute(
                    """
                    UPDATE meal_record
                    SET
                        eaten_at  = :eaten_at,
                        meal_type = :meal_type,
                        food_name = :food_name,
                        amount_g  = :amount_g,
                        kcal      = :kcal,
                        carb_g    = :carb_g,
                        protein_g = :protein_g,
                        fat_g     = :fat_g
                    WHERE user_id = :user_id
                      AND meal_id = :meal_id
                    """,
                    {
                        "eaten_at": eaten_at,
                        "meal_type": (new_meal_type or "").strip(),
                        "food_name": (new_food_name or "").strip(),
                        "amount_g": float(new_amount_g),
                        "kcal": float(nut["kcal"]),
                        "carb_g": float(nut["carbs_g"]),
                        "protein_g": float(nut["protein_g"]),
                        "fat_g": float(nut["fat_g"]),
                        "user_id": user_id,
                        "meal_id": meal_id_raw,
                    },
                )
            conn.commit()

            cur.execute(
                """
                SELECT TRUNC(eaten_at)
                FROM meal_record
                WHERE user_id = :user_id
                  AND meal_id = :meal_id
                """,
                {"user_id": user_id, "meal_id": meal_id_raw},
            )
            row = cur.fetchone()
            d = row[0] if row else date.today()

        self._rebuild_daily_from_db(user_id, d)

    def delete_meal_record(self, *, user_id: bytes, meal_id_raw: bytes) -> None:
        user_id = _ensure_bytes16(user_id)
        meal_id_raw = _ensure_raw16(meal_id_raw)

        with get_db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT TRUNC(eaten_at)
                FROM meal_record
                WHERE user_id = :user_id
                  AND meal_id = :meal_id
                """,
                {"user_id": user_id, "meal_id": meal_id_raw},
            )
            row = cur.fetchone()
            d = row[0] if row else date.today()

            cur.execute(
                """
                DELETE FROM meal_record
                WHERE user_id = :user_id
                  AND meal_id = :meal_id
                """,
                {"user_id": user_id, "meal_id": meal_id_raw},
            )
            conn.commit()

        self._rebuild_daily_from_db(user_id, d)

    # 아래는 기존 그대로
    def _new_daily(self, user_id: bytes, d: date) -> DailyMealSummary:
        return DailyMealSummary(
            user_id=_ensure_bytes16(user_id),
            date=d,
            meals=[],
            daily_nutrition=NutritionSummary(
                total={"kcal": 0.0, "carbs_g": 0.0, "protein_g": 0.0, "fat_g": 0.0},
                items_count=0,
            ),
            daily_score=0,
            daily_grade="미평가",
        )

    def _recalculate_daily(self, daily: DailyMealSummary):
        total = {"kcal": 0.0, "carbs_g": 0.0, "protein_g": 0.0, "fat_g": 0.0}
        for meal in daily.meals:
            for k in total:
                total[k] += float(meal.nutrition_summary.total.get(k, 0))
        daily.daily_nutrition.total = {k: round(v, 1) for k, v in total.items()}

        kcal = float(daily.daily_nutrition.total.get("kcal", 0))
        if kcal < 1600:
            daily.daily_score, daily.daily_grade = 90, "우수"
        elif kcal < 2000:
            daily.daily_score, daily.daily_grade = 75, "보통"
        else:
            daily.daily_score, daily.daily_grade = 55, "주의"

    def _rebuild_daily_from_db(self, user_id: bytes, d: date) -> None:
        user_id = _ensure_bytes16(user_id)

        with get_db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT
                    NVL(SUM(kcal), 0) AS total_kcal,
                    NVL(SUM(carb_g), 0) AS carb_g,
                    NVL(SUM(protein_g), 0) AS protein_g,
                    NVL(SUM(fat_g), 0) AS fat_g
                FROM meal_record
                WHERE user_id = :u
                  AND TRUNC(eaten_at) = TRUNC(:d)
                """,
                {"u": user_id, "d": d},
            )
            agg = cur.fetchone() or (0, 0, 0, 0)

        daily = self._new_daily(user_id, d)
        daily.daily_nutrition.total = {
            "kcal": float(agg[0] or 0),
            "carbs_g": float(agg[1] or 0),
            "protein_g": float(agg[2] or 0),
            "fat_g": float(agg[3] or 0),
        }

        kcal = float(daily.daily_nutrition.total.get("kcal", 0))
        if kcal < 1600:
            daily.daily_score, daily.daily_grade = 90, "우수"
        elif kcal < 2000:
            daily.daily_score, daily.daily_grade = 75, "보통"
        else:
            daily.daily_score, daily.daily_grade = 55, "주의"

        self._upsert_daily_meal_summary(daily)


    def _upsert_daily_meal_summary(self, daily: DailyMealSummary):
        user_id = _ensure_bytes16(daily.user_id)

        with get_db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                MERGE INTO daily_meal_summary s
                USING (
                    SELECT
                        :user_id AS user_id,
                        :summary_date AS summary_date,
                        :kcal AS total_kcal,
                        :carb AS carb_g,
                        :protein AS protein_g,
                        :fat AS fat_g,
                        :score AS daily_score,
                        :grade AS daily_grade
                    FROM dual
                ) src
                ON (
                    s.user_id = src.user_id
                    AND s.summary_date = src.summary_date
                )
                WHEN MATCHED THEN UPDATE SET
                    s.total_kcal = src.total_kcal,
                    s.carb_g = src.carb_g,
                    s.protein_g = src.protein_g,
                    s.fat_g = src.fat_g,
                    s.daily_score = src.daily_score,
                    s.daily_grade = src.daily_grade
                WHEN NOT MATCHED THEN INSERT (
                    user_id, summary_date, total_kcal,
                    carb_g, protein_g, fat_g,
                    daily_score, daily_grade
                )
                VALUES (
                    src.user_id, src.summary_date, src.total_kcal,
                    src.carb_g, src.protein_g, src.fat_g,
                    src.daily_score, src.daily_grade
                )
                """,
                {
                    "user_id": user_id,
                    "summary_date": daily.date,
                    "kcal": float(daily.daily_nutrition.total.get("kcal", 0)),
                    "carb": float(daily.daily_nutrition.total.get("carbs_g", 0)),
                    "protein": float(daily.daily_nutrition.total.get("protein_g", 0)),
                    "fat": float(daily.daily_nutrition.total.get("fat_g", 0)),
                    "score": int(daily.daily_score),
                    "grade": str(daily.daily_grade),
                },
            )
            conn.commit()
