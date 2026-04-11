# common/schemas.py
from __future__ import annotations

from datetime import date, datetime
from typing import List, Optional, Literal, Dict, Any, Tuple

from pydantic import (
    BaseModel,
    Field,
    AliasChoices,
    field_validator,
    model_validator,
)

from events.event_types import EventType


# =====================================================
# Common Coercions
# =====================================================
def _coerce_bytes16(v) -> bytes:
    """
    Accept:
      - bytes (len=16)
      - hex str (32 chars) -> bytes(16)
    """
    if isinstance(v, (bytes, bytearray)):
        b = bytes(v)
        if len(b) != 16:
            raise ValueError(f"user_id must be 16 bytes (RAW16). got len={len(b)}")
        return b

    if isinstance(v, str):
        s = v.strip()
        if len(s) == 32 and all(c in "0123456789abcdefABCDEF" for c in s):
            b = bytes.fromhex(s)
            if len(b) != 16:
                raise ValueError("hex user_id must decode to 16 bytes")
            return b

    raise ValueError("user_id must be bytes(16) or 32-length hex string")


def _coerce_date(v) -> date:
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, str):
        return date.fromisoformat(v[:10])
    raise ValueError(f"cannot coerce to date: {type(v)}")


def _normalize_activity_level(v: Any) -> Literal["low", "medium", "high"]:
    """
    DB/기존 코드에서 들어오는 값들 보정:
    - 'sedentary' -> 'low'
    - 'moderate'/'normal' -> 'medium'
    - 'active' -> 'high'
    """
    if v is None:
        return "medium"
    s = str(v).strip().lower()
    if s in ("low", "l", "1"):
        return "low"
    if s in ("medium", "mid", "moderate", "normal", "m", "2"):
        return "medium"
    if s in ("high", "h", "active", "3"):
        return "high"
    if s in ("sedentary",):
        return "low"
    return "medium"


def _safe_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


# =====================================================
# USER & PROFILE
# =====================================================
class UserProfile(BaseModel):
    """
    ✅ UI/E2E 안정성 우선:
    - 과거 코드/DB에서 누락되던 gender/age/weight_kg/activity_level 때문에 크래시 나던 부분 방지
    - birth_year가 있으면 age 자동 계산
    - activity_level은 sedentary 등도 자동 보정
    """

    user_id: bytes

    # DB/온보딩 상태에 따라 아직 없을 수 있음 -> Optional
    gender: Optional[Literal["male", "female"]] = None

    # birth_year 기반 계산 권장 (사용자 요구)
    birth_year: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("birth_year", "birthYear", "byear"),
    )

    # age는 직접 저장 안 해도 되지만(없을 수 있음) Loader/Validator가 참조할 수 있어 Optional로 둠
    age: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("age", "user_age"),
    )

    height_cm: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("height_cm", "height"),
    )

    # 기존 DB 컬럼명/코드 혼재 대비
    weight_kg: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("weight_kg", "weight_kg_baseline", "weight", "weight_kg_current"),
    )

    activity_level: Literal["low", "medium", "high"] = Field(
        default="medium",
        validation_alias=AliasChoices("activity_level", "activity", "activityLevel"),
    )

    has_diabetes: bool = False
    has_hypertension: bool = False

    # 파생/입력 모두 허용
    is_elderly: Optional[bool] = None

    banned_foods: List[str] = Field(default_factory=list)

    meal_priority: Literal[
        "balanced",
        "breakfast_high",
        "lunch_high",
        "dinner_high",
    ] = "balanced"

    # ✅ STEP A 요구: 선호 운동 저장 필드(스키마에 반영)
    preferred_exercise: List[str] = Field(default_factory=list)

    _coerce_uid = field_validator("user_id", mode="before")(_coerce_bytes16)

    @field_validator("activity_level", mode="before")
    @classmethod
    def _coerce_activity_level(cls, v):
        return _normalize_activity_level(v)

    @model_validator(mode="after")
    def _fill_age_and_elderly(self):
        # age 채우기
        if self.age is None and self.birth_year:
            self.age = date.today().year - int(self.birth_year)

        # is_elderly 채우기
        if self.is_elderly is None and self.age is not None:
            self.is_elderly = bool(self.age >= 65)

        return self

    @property
    def has_disease(self) -> bool:
        return bool(self.has_diabetes or self.has_hypertension)


class UserGoal(BaseModel):
    user_id: bytes

    goal_type: Literal["weight_loss", "maintenance", "muscle_gain"] = "maintenance"

    target_weight_kg: Optional[float] = None
    target_weeks: Optional[int] = None

    # DB 기반 goal 카드에서도 쓰는 값들(있으면 쓰고 없으면 None)
    kcal_target: Optional[float] = None
    macro_target: Optional[Dict[str, Any]] = None

    _coerce_uid = field_validator("user_id", mode="before")(_coerce_bytes16)


class UserPreferences(BaseModel):
    user_id: bytes
    preferred_foods: List[str] = Field(default_factory=list)
    disliked_foods: List[str] = Field(default_factory=list)
    banned_foods: List[str] = Field(default_factory=list)

    # ✅ 선호 운동(온보딩/설정/가이드 공통)
    preferred_exercise: List[str] = Field(default_factory=list)

    exercise_limitations: List[str] = Field(default_factory=list)

    _coerce_uid = field_validator("user_id", mode="before")(_coerce_bytes16)


# =====================================================
# EVENTS
# =====================================================
class Event(BaseModel):
    user_id: bytes
    event_type: EventType

    datetime_start: datetime
    datetime_end: Optional[datetime] = None

    intensity: Literal["light", "normal", "heavy"] = "normal"

    @field_validator("event_type", mode="before")
    @classmethod
    def coerce_event_type(cls, v):
        return EventType.from_any(v)

    _coerce_uid = field_validator("user_id", mode="before")(_coerce_bytes16)


# =====================================================
# FOOD / DIET PLAN
# =====================================================
class MealItem(BaseModel):
    food_name: str
    category: Optional[str] = ""
    portion_gram: float = 0.0
    calorie: float = 0.0
    carb: float = 0.0
    protein: float = 0.0
    fat: float = 0.0

    # diet_validator.py에서 item.recommend_level 체크함
    recommend_level: Optional[str] = None


class DailyDietPlan(BaseModel):
    """
    ✅ diet_validator.py 호환을 위해 이 필드들이 반드시 존재해야 함:
      - target_kcal
      - total_kcal
      - total_macro (protein_g 등)
      - meals (breakfast/lunch/dinner/snack 키)
    또한 과거 형태(breakfast/lunch/dinner/snacks 리스트)로 들어와도 자동 변환.
    """

    user_id: bytes
    date: date

    # 핵심(validator가 참조)
    target_kcal: float = 0.0
    total_kcal: float = 0.0
    total_macro: Dict[str, float] = Field(default_factory=dict)

    # 표준 표현: meals dict
    meals: Dict[str, List[MealItem]] = Field(default_factory=dict)

    # 과거/대안 표현(있으면 meals로 흡수)
    breakfast: Optional[List[MealItem]] = None
    lunch: Optional[List[MealItem]] = None
    dinner: Optional[List[MealItem]] = None
    snacks: Optional[List[MealItem]] = None

    _coerce_uid = field_validator("user_id", mode="before")(_coerce_bytes16)
    _coerce_d = field_validator("date", mode="before")(_coerce_date)

    @model_validator(mode="after")
    def _normalize_meals(self):
        # meals가 비어 있고, 개별 리스트가 있으면 dict로 구성
        if not self.meals:
            m: Dict[str, List[MealItem]] = {}
            if self.breakfast is not None:
                m["breakfast"] = self.breakfast
            if self.lunch is not None:
                m["lunch"] = self.lunch
            if self.dinner is not None:
                m["dinner"] = self.dinner
            if self.snacks is not None:
                m["snack"] = self.snacks
            self.meals = m

        # total_macro 키 표준화(프로젝트 내에서 carbs_g/protein_g/fat_g 혼재 가능)
        tm = dict(self.total_macro or {})
        # 보정: carb_g -> carbs_g
        if "carb_g" in tm and "carbs_g" not in tm:
            tm["carbs_g"] = float(tm.get("carb_g") or 0.0)
        self.total_macro = {k: float(v or 0.0) for k, v in tm.items()}

        # total_kcal / target_kcal 안전 보정
        self.total_kcal = float(self.total_kcal or 0.0)
        self.target_kcal = float(self.target_kcal or 0.0)

        return self


class WeeklyDietPlan(BaseModel):
    user_id: bytes
    week_start: date
    week_end: date
    days: List[DailyDietPlan] = Field(default_factory=list)

    _coerce_uid = field_validator("user_id", mode="before")(_coerce_bytes16)


# =====================================================
# EXERCISE
# =====================================================
class ExerciseItem(BaseModel):
    name: str
    category: Literal["strength", "cardio"]
    intensity: Optional[str] = None

    minutes: int = Field(
        0,
        validation_alias=AliasChoices("minutes", "duration_min", "duration_minutes"),
    )

    met_value: float = 0.0
    calorie_burn: float = 0.0


class DailyExercisePlan(BaseModel):
    """
    ✅ 홈에서 크래시 났던 지점 보완:
    - user_id/date/exercises/total_calorie_burn가 없으면 ValidationError → 이제 기본값으로 흡수
    - ExerciseAdapter가 완전한 형태로 리턴하면 그대로 사용
    """

    user_id: bytes
    date: date

    exercises: List[ExerciseItem] = Field(default_factory=list)
    total_minutes: int = 0
    total_calorie_burn: float = 0.0

    _coerce_uid = field_validator("user_id", mode="before")(_coerce_bytes16)
    _coerce_d = field_validator("date", mode="before")(_coerce_date)

    @field_validator("total_minutes", mode="before")
    @classmethod
    def _coerce_total_minutes(cls, v):
        return int(v or 0)

    @field_validator("total_calorie_burn", mode="before")
    @classmethod
    def _coerce_total_calorie(cls, v):
        return float(v or 0.0)


# =====================================================
# ROUTINE
# =====================================================
class DailyRoutine(BaseModel):
    user_id: bytes
    date: date
    diet: DailyDietPlan
    exercise: DailyExercisePlan

    _coerce_uid = field_validator("user_id", mode="before")(_coerce_bytes16)


class WeeklyRoutine(BaseModel):
    user_id: bytes
    week_start: date
    week_end: date
    daily_routines: List[DailyRoutine] = Field(default_factory=list)

    _coerce_uid = field_validator("user_id", mode="before")(_coerce_bytes16)


# =====================================================
# MEAL CONTEXT
# =====================================================
class MealContext(BaseModel):
    meal_type: Optional[Literal["breakfast", "lunch", "dinner", "snack", "manual"]] = None
    source: Optional[str] = None

    elderly: bool = False
    diabetes: bool = False

    diet: Optional[Literal["weight_loss", "maintenance", "muscle_gain"]] = None
    fitness: Optional[Literal["pre_workout", "post_workout"]] = None


# =====================================================
# MEAL LOG & SUMMARY
# =====================================================
class NutritionSummary(BaseModel):
    total: Dict[str, float] = Field(default_factory=dict)
    items_count: int = 0


class MealEvaluation(BaseModel):
    meal_score: int = Field(0, ge=0, le=100)
    grade: str = "미평가"
    flags: Dict[str, bool] = Field(default_factory=dict)
    advice: List[str] = Field(default_factory=list)


class MealRecord(BaseModel):
    meal_id: str
    user_id: bytes

    image: Optional[str] = None
    created_at: datetime

    primary_food: Optional[Dict[str, Any]] = None
    foods: List[Dict[str, Any]] = Field(default_factory=list)

    nutrition_summary: NutritionSummary = Field(default_factory=NutritionSummary)
    meal_evaluation: Optional[MealEvaluation] = None
    context: Optional[MealContext] = None

    _coerce_uid = field_validator("user_id", mode="before")(_coerce_bytes16)


class DailyMealSummary(BaseModel):
    user_id: bytes
    date: date

    meals: List[MealRecord] = Field(default_factory=list)

    daily_nutrition: NutritionSummary = Field(default_factory=NutritionSummary)
    daily_score: int = 0
    daily_grade: str = "미평가"

    feedback: Optional[List[str]] = None

    _coerce_uid = field_validator("user_id", mode="before")(_coerce_bytes16)


class WeeklyMealSummary(BaseModel):
    user_id: bytes
    week_start: date
    week_end: date

    days: List[DailyMealSummary] = Field(default_factory=list)

    weekly_nutrition: NutritionSummary = Field(default_factory=NutritionSummary)
    weekly_score: int = 0
    weekly_grade: str = "미평가"

    feedback: List[str] = Field(default_factory=list)
    nutrient_boost: List[str] = Field(default_factory=list)

    _coerce_uid = field_validator("user_id", mode="before")(_coerce_bytes16)


class MonthlyMealSummary(BaseModel):
    user_id: bytes
    month: str  # "2025-12"

    days: List[DailyMealSummary] = Field(default_factory=list)

    monthly_nutrition: NutritionSummary = Field(default_factory=NutritionSummary)
    monthly_score: int = 0
    monthly_grade: str = "미평가"

    feedback: Optional[List[str]] = None

    _coerce_uid = field_validator("user_id", mode="before")(_coerce_bytes16)


# =====================================================
# EXERCISE RECORD (LOG)
# =====================================================
class ExerciseRecord(BaseModel):
    """
    주의:
    - 현재 exercise_service.py는 exercise_id를 uuid.bytes로 넣고 있음.
      (DB 컬럼이 RAW(16)이라면 bytes(16)로 두는 게 맞음)
    """

    exercise_id: bytes = Field(..., validation_alias=AliasChoices("exercise_id", "id"))
    user_id: bytes

    exercise_type: str
    minutes: int = 0
    intensity: str = "normal"
    date: date

    _coerce_uid = field_validator("user_id", mode="before")(_coerce_bytes16)

    @field_validator("exercise_id", mode="before")
    @classmethod
    def _coerce_exercise_id(cls, v):
        # bytes(16) or hex(32) 모두 허용
        return _coerce_bytes16(v)

    _coerce_d = field_validator("date", mode="before")(_coerce_date)
