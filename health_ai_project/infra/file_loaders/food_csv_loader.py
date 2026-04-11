# infra/file_loaders/food_csv_loader.py
from __future__ import annotations

from pathlib import Path
from functools import lru_cache
from typing import Optional

import pandas as pd

from config import BASE_DIR


# 프로젝트 기본 경로 (원하면 여기만 바꾸면 됨)
DEFAULT_XLSX = BASE_DIR / "data" / "food_nutrition.xlsx"


def _norm_str(x) -> str:
    if x is None:
        return ""
    s = str(x)
    return s.strip()


def _to_num(s: pd.Series) -> pd.Series:
    # -99 같은 값은 NaN으로 만들고, 최종적으로 0.0으로 채움(계산 안전)
    out = pd.to_numeric(s, errors="coerce")
    out = out.replace(-99, pd.NA)
    return out.fillna(0.0)


@lru_cache(maxsize=1)
def load_food_nutrition_db(path: Optional[str | Path] = None) -> pd.DataFrame:
    """
    food_nutrition.xlsx 로드.
    ✅ 핵심: L/M/N(대분류/중분류/권장도)이 비어있는 행은 제외 (800번 이전 제거 목적)
    """
    xlsx_path = Path(path) if path else DEFAULT_XLSX
    if not xlsx_path.exists():
        raise FileNotFoundError(f"food nutrition xlsx not found: {xlsx_path}")

    df = pd.read_excel(xlsx_path)

    # 원본 컬럼명(사용자 파일 기준)
    # A: class_id, B: 음식명, C: 중량(g), D: 에너지(kcal),
    # E: 탄수화물(g), F: 당류(g), G: 지방(g), H: 단백질(g),
    # I: 나트륨(mg), J: 콜레스테롤(mg), K: 트랜스지방(g),
    # L: 대분류, M: 중분류, N: 권장도
    # (혹시 다른 이름이 섞여도 안전하게 처리)
    col_food = "음식명" if "음식명" in df.columns else ("food_name" if "food_name" in df.columns else None)
    if not col_food:
        raise ValueError(f"cannot find food name column in {list(df.columns)}")

    # 1) 이름 없는 행 제거
    df = df.copy()
    df[col_food] = df[col_food].astype(str).map(_norm_str)
    df = df[df[col_food] != ""]

    # 2) ✅ 라벨링(대/중/권장도) 없는 행 제거 -> 800번 이전 데이터 자동 제거
    col_major = "대분류" if "대분류" in df.columns else None
    col_mid = "중분류" if "중분류" in df.columns else None
    col_rec = "권장도" if "권장도" in df.columns else None

    # 대분류/중분류/권장도 중 하나라도 없으면(컬럼 자체가 없다면) 필터링 불가 → 그대로 반환
    # (하지만 지금 사용자 파일에는 존재)
    if col_major and col_mid and col_rec:
        df[col_major] = df[col_major].astype(str).map(_norm_str)
        df[col_mid] = df[col_mid].astype(str).map(_norm_str)
        df[col_rec] = df[col_rec].astype(str).map(_norm_str)

        df = df[(df[col_major] != "") & (df[col_mid] != "") & (df[col_rec] != "")]

    # 3) 영양 컬럼 정리 (없으면 0 처리)
    def _get(colname: str) -> pd.Series:
        return df[colname] if colname in df.columns else pd.Series([0] * len(df))

    df["serving_g"] = _to_num(_get("중량(g)")) if "중량(g)" in df.columns else _to_num(_get("중량"))
    df["calorie"] = _to_num(_get("에너지(kcal)")) if "에너지(kcal)" in df.columns else _to_num(_get("에너지"))
    df["carb"] = _to_num(_get("탄수화물(g)"))
    df["protein"] = _to_num(_get("단백질(g)"))
    df["fat"] = _to_num(_get("지방(g)"))
    df["sodium_mg"] = _to_num(_get("나트륨(mg)"))

    # 4) 공통 접근용 별칭 컬럼
    df["food_name"] = df[col_food]

    # 원본 분류 컬럼 유지 + 별칭
    if col_major:
        df["category_kr"] = df[col_major]
    else:
        df["category_kr"] = ""

    if col_mid:
        df["sub_category_kr"] = df[col_mid]
    else:
        df["sub_category_kr"] = ""

    if col_rec:
        df["recommend_level"] = df[col_rec]
    else:
        df["recommend_level"] = ""

    return df.reset_index(drop=True)
