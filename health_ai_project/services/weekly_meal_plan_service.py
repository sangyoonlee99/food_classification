from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from infra.db_server import get_db_conn


def _lob_to_str(val: Any) -> Optional[str]:
    if val is None:
        return None
    try:
        if hasattr(val, "read"):
            return val.read()
    except Exception:
        pass
    if isinstance(val, (bytes, bytearray)):
        try:
            return val.decode("utf-8")
        except Exception:
            return str(val)
    return str(val)


def _safe_json_loads(val: Any, default):
    s = _lob_to_str(val)
    if not s:
        return default
    try:
        return json.loads(s)
    except Exception:
        return default


def _find_food_nutrition_path() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / "data" / "food_nutrition.xlsx",
        here.parents[2] / "data" / "food_nutrition.xlsx",
        here.parents[1] / "food_nutrition.xlsx",
        here.parents[2] / "food_nutrition.xlsx",
    ]
    for p in candidates:
        if p.exists():
            return p
    p = Path("food_nutrition.xlsx")
    if p.exists():
        return p
    raise FileNotFoundError("food_nutrition.xlsx 를 찾지 못했습니다. data/food_nutrition.xlsx 위치를 확인하세요.")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {}

    def pick(*names):
        for n in names:
            if n in df.columns:
                return n
        return None

    c_class = pick("class_id", "CLASS_ID", "클래스ID", "클래스_id", "id")
    c_name = pick("음식명", "식품명", "food_name", "FOOD_NAME", "name", "NAME")
    c_g = pick("중량(g)", "중량", "grams", "GRAMS", "g", "weight_g", "amount_g")
    c_k = pick("에너지(kcal)", "에너지", "kcal", "KCAL", "energy_kcal")
    c_c = pick("탄수화물(g)", "탄수화물", "carb_g", "CARB_G")
    c_p = pick("단백질(g)", "단백질", "protein_g", "PROTEIN_G")
    c_f = pick("지방(g)", "지방", "fat_g", "FAT_G")
    c_na = pick("나트륨(mg)", "나트륨", "sodium_mg", "SODIUM_MG")
    c_big = pick("대분류", "big_category", "BIG_CATEGORY")
    c_mid = pick("중분류", "mid_category", "MID_CATEGORY")
    c_rec = pick("권장도", "recommend", "RECOMMEND")

    if c_class:
        colmap[c_class] = "class_id"
    if c_name:
        colmap[c_name] = "food_name"
    if c_g:
        colmap[c_g] = "grams_base"
    if c_k:
        colmap[c_k] = "kcal"
    if c_c:
        colmap[c_c] = "carb_g"
    if c_p:
        colmap[c_p] = "protein_g"
    if c_f:
        colmap[c_f] = "fat_g"
    if c_na:
        colmap[c_na] = "sodium_mg"
    if c_big:
        colmap[c_big] = "big_cat"
    if c_mid:
        colmap[c_mid] = "mid_cat"
    if c_rec:
        colmap[c_rec] = "recommend"

    df = df.rename(columns=colmap)

    for need, default in [
        ("class_id", None),
        ("food_name", None),
        ("grams_base", 100.0),
        ("kcal", 0.0),
        ("carb_g", 0.0),
        ("protein_g", 0.0),
        ("fat_g", 0.0),
        ("sodium_mg", 0.0),
        ("big_cat", ""),
        ("mid_cat", ""),
        ("recommend", ""),
    ]:
        if need not in df.columns:
            df[need] = default

    for c in ["class_id", "grams_base", "kcal", "carb_g", "protein_g", "fat_g", "sodium_mg"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["food_name"] = df["food_name"].astype(str)
    df["big_cat"] = df["big_cat"].fillna("").astype(str)
    df["mid_cat"] = df["mid_cat"].fillna("").astype(str)
    df["recommend"] = df["recommend"].fillna("").astype(str)

    return df


@dataclass
class MacroTargets:
    carb_ratio: float = 0.45
    protein_ratio: float = 0.30
    fat_ratio: float = 0.25


class WeeklyMealPlanService:
    """
    - class_id >= 800 구간을 "식단표(추천)" 후보로 사용
    - 탄/단/지/야채 4파트 기본 포함
    - ✅ g 폭주 방지(강화): 각 카테고리별 clamp + 스케일 cap
    """

    def __init__(self, nutrition_path: Optional[Path] = None):
        self.nutrition_path = Path(nutrition_path) if nutrition_path else _find_food_nutrition_path()

    @lru_cache(maxsize=1)
    def _load_food_df(self) -> pd.DataFrame:
        df = pd.read_excel(self.nutrition_path, sheet_name=0)
        df = _normalize_columns(df)

        df = df[df["class_id"].fillna(-1) >= 800].copy()

        if (df["recommend"].str.len() > 0).any():
            df = df[(df["recommend"].str.contains("권장")) | (df["recommend"].str.len() == 0)].copy()

        df = df[df["kcal"].fillna(0) > 0].copy()
        df = df[df["grams_base"].fillna(0) > 0].copy()

        return df

    def _load_user_context(self, user_id: str) -> Tuple[int, MacroTargets, List[str]]:
        daily_kcal = 0
        macro = MacroTargets()
        conditions: List[str] = []

        with get_db_conn() as conn:
            cur = conn.cursor()

            cur.execute(
                """
                SELECT kcal_target, macro_target
                FROM user_goal
                WHERE user_id = :u AND is_active = 'Y'
                """,
                {"u": user_id},
            )
            row = cur.fetchone()
            if row:
                kcal_target, macro_json = row
                if kcal_target:
                    daily_kcal = int(kcal_target)

                m = _safe_json_loads(macro_json, {})
                carb = m.get("carb") or m.get("carb_ratio") or m.get("carbohydrate")
                pro = m.get("protein") or m.get("protein_ratio")
                fat = m.get("fat") or m.get("fat_ratio")
                try:
                    if carb is not None:
                        macro.carb_ratio = float(carb) / (100.0 if float(carb) > 1 else 1.0)
                    if pro is not None:
                        macro.protein_ratio = float(pro) / (100.0 if float(pro) > 1 else 1.0)
                    if fat is not None:
                        macro.fat_ratio = float(fat) / (100.0 if float(fat) > 1 else 1.0)
                except Exception:
                    pass

                s = macro.carb_ratio + macro.protein_ratio + macro.fat_ratio
                if s > 0:
                    macro.carb_ratio /= s
                    macro.protein_ratio /= s
                    macro.fat_ratio /= s

            cur.execute(
                """
                SELECT conditions
                FROM user_profile
                WHERE user_id = :u
                """,
                {"u": user_id},
            )
            prow = cur.fetchone()
            if prow:
                conditions = _safe_json_loads(prow[0], [])
                if not isinstance(conditions, list):
                    conditions = []

        return daily_kcal, macro, conditions

    def _load_day_events(self, user_id: str, day) -> List[Tuple[str, str]]:
        with get_db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT event_type, severity
                FROM event_log
                WHERE user_id = :u AND event_date = :d
                """,
                {"u": user_id, "d": day},
            )
            return cur.fetchall()

    def _apply_condition_filters(self, df: pd.DataFrame, conditions: List[str]) -> pd.DataFrame:
        dff = df.copy()

        cond_text = " ".join(conditions).lower()
        has_diabetes = any(k in cond_text for k in ["당뇨", "diabetes"])
        has_htn = any(k in cond_text for k in ["고혈압", "hypertension"])

        if has_diabetes:
            bad = dff["mid_cat"].str.contains("고GI", case=False, na=False) | dff["mid_cat"].str.contains(
                "정제", case=False, na=False
            )
            if (~bad).sum() >= 50:
                dff = dff[~bad].copy()

        if has_htn:
            bad = dff["sodium_mg"].fillna(0) > 700
            if (~bad).sum() >= 50:
                dff = dff[~bad].copy()

        return dff

    def build_weekly_plan(self, user_id: str, week_days: List) -> Dict:
        daily_kcal, macro, conditions = self._load_user_context(user_id)
        if daily_kcal <= 0:
            daily_kcal = 1800

        df = self._load_food_df()
        df = self._apply_condition_filters(df, conditions)

        pools = {
            "carb": df[df["big_cat"].str.contains("탄수", na=False)].copy(),
            "protein": df[df["big_cat"].str.contains("단백", na=False)].copy(),
            "fat": df[df["big_cat"].str.contains("지방", na=False)].copy(),
            "veg": df[df["big_cat"].str.contains("채소|야채|veget", case=False, na=False)].copy(),
        }

        for k, p in list(pools.items()):
            if len(p) < 10:
                pools[k] = df.copy()

        out: Dict = {}

        for d in week_days:
            events = self._load_day_events(user_id, d)
            day_kcal = daily_kcal

            sev_text = " ".join([str(s or "") for _, s in events]).lower()
            if "높음" in sev_text:
                day_kcal = int(day_kcal * 0.90)
            elif "보통" in sev_text:
                day_kcal = int(day_kcal * 0.95)

            plan = self._build_day_plan(pools, day_kcal, macro)
            plan["daily_target_kcal"] = day_kcal
            plan["events"] = [{"type": et, "severity": sv} for et, sv in events]
            out[d] = plan

        return out

    def _build_day_plan(self, pools: Dict[str, pd.DataFrame], day_kcal: int, macro: MacroTargets) -> Dict:
        meal_splits = {
            "아침": 0.25,
            "점심": 0.35,
            "저녁": 0.30,
            "간식": 0.10,
        }

        meals: Dict[str, List[Dict[str, Any]]] = {}
        totals = {"kcal": 0, "carb_g": 0.0, "protein_g": 0.0, "fat_g": 0.0}

        for meal, ratio in meal_splits.items():
            target_kcal = int(day_kcal * ratio)
            meal_items = self._build_meal(pools, target_kcal, macro, meal_name=meal)
            meals[meal] = meal_items

            for it in meal_items:
                totals["kcal"] += int(it["kcal"])
                totals["carb_g"] += float(it.get("carb_g", 0))
                totals["protein_g"] += float(it.get("protein_g", 0))
                totals["fat_g"] += float(it.get("fat_g", 0))

        totals["carb_g"] = round(totals["carb_g"], 1)
        totals["protein_g"] = round(totals["protein_g"], 1)
        totals["fat_g"] = round(totals["fat_g"], 1)

        return {"meals": meals, "planned_totals": totals}

    def _pick_row(self, df: pd.DataFrame) -> pd.Series:
        idx = random.randrange(0, len(df))
        return df.iloc[idx]

    def _scale_item(self, row: pd.Series, target_kcal: int, clamp_g: Tuple[int, int]) -> Dict[str, Any]:
        base_g = float(row["grams_base"] or 100.0)
        base_kcal = float(row["kcal"] or 1.0)

        # ✅ 스케일 cap 강화(폭주 방지)
        factor = target_kcal / base_kcal if base_kcal > 0 else 1.0
        factor = max(0.6, min(2.0, factor))

        grams = int(round(base_g * factor))
        grams = max(clamp_g[0], min(clamp_g[1], grams))

        def scale(v):
            v = float(v or 0.0)
            return v * (grams / base_g)

        kcal = int(round(base_kcal * (grams / base_g)))

        return {
            "food": str(row["food_name"]),
            "grams": grams,
            "kcal": kcal,
            "carb_g": round(scale(row.get("carb_g", 0.0)), 1),
            "protein_g": round(scale(row.get("protein_g", 0.0)), 1),
            "fat_g": round(scale(row.get("fat_g", 0.0)), 1),
            "big_cat": str(row.get("big_cat", "")),
            "mid_cat": str(row.get("mid_cat", "")),
        }

    def _build_meal(
        self,
        pools: Dict[str, pd.DataFrame],
        meal_kcal: int,
        macro: MacroTargets,
        meal_name: str,
    ) -> List[Dict[str, Any]]:
        if meal_name == "간식":
            parts = {"carb": 0.55, "protein": 0.25, "veg": 0.15, "fat": 0.05}
        else:
            parts = {"carb": 0.45, "protein": 0.35, "veg": 0.15, "fat": 0.05}

        # ✅ 현실 g 제한(강화)
        clamp = {
            "carb": (80, 250),
            "protein": (80, 250),
            "veg": (60, 250),
            "fat": (5, 35),
        }

        items: List[Dict[str, Any]] = []

        for cat, r in parts.items():
            target = int(meal_kcal * r)
            row = self._pick_row(pools[cat])
            it = self._scale_item(row, target_kcal=target, clamp_g=clamp[cat])

            tag = cat
            if cat == "carb":
                tag = "carb_low_gi" if ("저GI" in it["mid_cat"] or "저gi" in it["mid_cat"].lower()) else "carb"
            if cat == "protein":
                tag = "protein"
            if cat == "veg":
                tag = "vegetable"
            if cat == "fat":
                tag = "fat"

            it["tag"] = tag
            items.append(it)

        # ✅ kcal 미세 보정: carb만 범위 내에서
        total = sum(i["kcal"] for i in items)
        if total > 0:
            low = int(meal_kcal * 0.85)
            high = int(meal_kcal * 1.15)
            if total < low or total > high:
                for i in items:
                    if i.get("tag") in ("carb", "carb_low_gi"):
                        diff = meal_kcal - total
                        per_g = max(0.5, i["kcal"] / max(1, i["grams"]))
                        add_g = int(round(diff / per_g))
                        i["grams"] = max(clamp["carb"][0], min(clamp["carb"][1], i["grams"] + add_g))
                        i["kcal"] = int(round(i["grams"] * per_g))
                        break

        return items
