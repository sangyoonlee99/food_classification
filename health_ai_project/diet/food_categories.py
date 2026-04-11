# diet/food_categories.py
from __future__ import annotations

from typing import Optional
import pandas as pd


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def _find_col(
    df: pd.DataFrame,
    candidates: list[str],
    contains: Optional[str] = None,
) -> Optional[str]:
    cols = list(df.columns)

    # 1) 정확 일치
    for c in candidates:
        if c in cols:
            return c

    # 2) 부분 문자열
    if contains:
        for col in cols:
            if contains in str(col):
                return col

    return None


def filter_food_candidates(
    nutrition_df: pd.DataFrame,
    category: str,
    *,
    sub_category: Optional[str] = None,
    profile: Optional[dict] = None,
    recommend_only: bool | None = None,
) -> pd.DataFrame:
    """
    ✔ L/M/N 기반 엄격 필터
    ✔ 실패 시 fallback 제거 (현실성 보장)
    """

    df = _normalize_columns(nutrition_df)

    # ==================================================
    # 🔹 대분류
    # ==================================================
    big_col = _find_col(
        df,
        candidates=["대분류", "food_group"],
        contains="대분류",
    )
    if not big_col:
        raise KeyError(f"'대분류' 컬럼 없음: {list(df.columns)}")

    filtered = df[df[big_col].astype(str).str.strip() == str(category).strip()]

    # ==================================================
    # 🔹 중분류
    # ==================================================
    if sub_category:
        mid_col = _find_col(
            df,
            candidates=["중분류", "food_sub_group"],
            contains="중분류",
        )
        if mid_col:
            filtered = filtered[
                filtered[mid_col].astype(str).str.strip()
                == str(sub_category).strip()
            ]

    # ==================================================
    # 🔹 권장도 필터 (핵심)
    # ==================================================
    if recommend_only:
        rec_col = _find_col(
            df,
            candidates=["권장도"],
            contains="권장",
        )
        if rec_col:
            filtered = filtered[
                filtered[rec_col].astype(str).str.contains("권장")
            ]

    # ==================================================
    # ❌ fallback 제거 (중요)
    # ==================================================
    if filtered.empty:
        # 추천 가능한 음식이 없다는 것을 명확히 표현
        return df.iloc[0:0]

    return filtered
