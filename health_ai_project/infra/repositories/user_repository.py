from __future__ import annotations

from typing import Optional, Dict, Any
import json
from infra.db_server import get_db_conn


class UserRepository:
    # ==================================================
    # 기본 CRUD
    # ==================================================
    def create_user(
        self,
        *,
        user_id: bytes,
        email: str,
        password_hash: str,
    ) -> None:
        sql = """
        INSERT INTO users (user_id, email, password_hash)
        VALUES (:user_id, :email, :password_hash)
        """
        with get_db_conn() as conn:
            conn.cursor().execute(
                sql,
                {
                    "user_id": user_id,
                    "email": email,
                    "password_hash": password_hash,
                },
            )
            conn.commit()

    def get_user(self, *, user_id: bytes) -> Optional[tuple]:
        sql = "SELECT * FROM users WHERE user_id = :user_id"
        with get_db_conn() as conn:
            cur = conn.cursor()
            cur.execute(sql, {"user_id": user_id})
            return cur.fetchone()

    # ==================================================
    # 🔥 사용자 초기화
    # ==================================================
    def ensure_user_initialized(self, *, user_id: bytes) -> None:
        with get_db_conn() as conn:
            cur = conn.cursor()

            cur.execute(
                """
                MERGE INTO users u
                USING (SELECT :user_id AS user_id FROM dual) src
                ON (u.user_id = src.user_id)
                WHEN NOT MATCHED THEN
                  INSERT (
                    user_id,
                    email,
                    password_hash,
                    status,
                    created_at
                  )
                  VALUES (
                    :user_id,
                    'guest_' || RAWTOHEX(:user_id) || '@local',
                    'TEMP',
                    'active',
                    CURRENT_TIMESTAMP
                  )
                """,
                {"user_id": user_id},
            )

            cur.execute(
                """
                MERGE INTO user_settings s
                USING (SELECT :user_id AS user_id FROM dual) src
                ON (s.user_id = src.user_id)
                WHEN NOT MATCHED THEN
                  INSERT (
                    user_id,
                    allow_menu_chg_plt_days,
                    allow_routine_chg_plt_days,
                    est_delta_kcal
                  )
                  VALUES (
                    :user_id,
                    14,
                    14,
                    400
                  )
                """,
                {"user_id": user_id},
            )

            conn.commit()

    # ==================================================
    # 🔧 Utils
    # ==================================================
    @staticmethod
    def _load_clob(val):
        if val is None:
            return []
        if hasattr(val, "read"):
            val = val.read()
        return json.loads(val) if val else []

    @staticmethod
    def _list_to_str(val):
        if val is None:
            return None
        if isinstance(val, list):
            return ",".join(val)
        return val

    # ==================================================
    # 👤 사용자 프로필 조회
    # ==================================================
    def get_user_profile(self, *, user_id: bytes) -> dict | None:
        import json
        from infra.db_server import get_db_conn

        with get_db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT
                    sex,
                    birth_year,
                    height_cm,
                    weight_kg_baseline,
                    activity_level,
                    conditions,
                    preferred_cardio,
                    preferred_strength
                FROM user_profile
                WHERE user_id = :u
                """,
                {"u": user_id},
            )
            row = cur.fetchone()

        if not row:
            return None

        sex, birth, height, weight, activity, cond, cardio, strength = row

        def _loads(v):
            if not v:
                return []
            try:
                return json.loads(v)
            except Exception:
                return []

        return {
            "sex": sex,
            "birth_year": birth,
            "height_cm": height,
            "weight_kg_baseline": weight,
            "activity_level": activity,
            "conditions": _loads(cond),
            "preferred_cardio": _loads(cardio),
            "preferred_strength": _loads(strength),
        }

    # ==================================================
    # ✅ 온보딩 완료 여부
    # ==================================================
    def has_profile(self, user_id: bytes) -> bool:
        sql = """
        SELECT 1
        FROM user_profile
        WHERE user_id = :p_user_id
          AND birth_year IS NOT NULL
          AND height_cm IS NOT NULL
          AND weight_kg_baseline IS NOT NULL
          AND activity_level IS NOT NULL
        """
        with get_db_conn() as conn:
            cur = conn.cursor()
            cur.execute(sql, {"p_user_id": user_id})
            return cur.fetchone() is not None

    # ==================================================
    # 💾 프로필 저장 (🔥 FIXED)
    # ==================================================
    def save_user_profile(
            self,
            user_id,
            sex,
            birth_year,
            height_cm,
            weight_kg,
            activity_level,
            conditions,
            preferences,
            preferred_cardio,
            preferred_strength,
    ):
        import json
        from infra.db_server import get_db_conn

        with get_db_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                MERGE INTO user_profile p
                USING (SELECT :u AS user_id FROM dual) s
                ON (p.user_id = s.user_id)
                WHEN MATCHED THEN
                  UPDATE SET
                    sex = :sex,
                    birth_year = :birth,
                    height_cm = :height,
                    weight_kg_baseline = :weight,
                    activity_level = :activity,
                    conditions = :conditions,
                    preferred_cardio = :preferred_cardio,
                    preferred_strength = :preferred_strength
                WHEN NOT MATCHED THEN
                  INSERT (
                    user_id, sex, birth_year, height_cm,
                    weight_kg_baseline, activity_level,
                    conditions, preferred_cardio, preferred_strength
                  )
                  VALUES (
                    :u, :sex, :birth, :height,
                    :weight, :activity,
                    :conditions, :preferred_cardio, :preferred_strength
                  )
                """,
                {
                    "u": user_id,
                    "sex": sex,
                    "birth": birth_year,
                    "height": height_cm,
                    "weight": weight_kg,
                    "activity": activity_level,
                    "conditions": json.dumps(conditions, ensure_ascii=False),
                    "preferred_cardio": json.dumps(preferred_cardio, ensure_ascii=False),
                    "preferred_strength": json.dumps(preferred_strength, ensure_ascii=False),
                },
            )
            conn.commit()

    # ==================================================
    # 🎯 목표 저장
    # ==================================================
    def save_user_goal(
        self,
        *,
        user_id: bytes,
        goal_type: str,
        kcal_target: int,
        macro_target: dict,
    ) -> None:
        with get_db_conn() as conn:
            cur = conn.cursor()

            cur.execute(
                """
                UPDATE user_goal
                SET is_active = 'N'
                WHERE user_id = :p_user_id
                  AND is_active = 'Y'
                """,
                {"p_user_id": user_id},
            )

            cur.execute(
                """
                INSERT INTO user_goal (
                    goal_id, user_id, goal_type, start_date,
                    kcal_target, macro_target, is_active
                ) VALUES (
                    SYS_GUID(), :p_user_id, :goal, TRUNC(SYSDATE),
                    :kcal, :macro, 'Y'
                )
                """,
                {
                    "p_user_id": user_id,
                    "goal": goal_type,
                    "kcal": kcal_target,
                    "macro": json.dumps(macro_target),
                },
            )

            conn.commit()

    # ==================================================
    # ⚙️ 사용자 설정 조회 (Recommendation / Scheduler 공용)
    # ==================================================
    def get_user_settings(self, *, user_id: bytes) -> Dict[str, Any]:
        """
        recommendation.py / replan_scheduler 에서 사용하는 사용자 설정
        DB(user_settings) 기준 + 안전한 기본값 fallback
        """
        sql = """
        SELECT
            allow_menu_chg_plt_days,
            allow_routine_chg_plt_days,
            est_delta_kcal,
            preferred_cardio,
            preferred_strength,
            exercise_experience,
            other_flags
        FROM user_settings
        WHERE user_id = :user_id
        """
        with get_db_conn() as conn:
            cur = conn.cursor()
            cur.execute(sql, {"user_id": user_id})
            row = cur.fetchone()

        # DB에 없으면 (이론상 ensure_user_initialized로 거의 없음)
        if not row:
            return {
                "allow_menu_chg_plt_days": 14,
                "allow_routine_chg_plt_days": 14,
                "est_delta_kcal": 400,
                "preferred_cardio": [],
                "preferred_strength": [],
                "exercise_experience": None,
                "flags": {},
            }

        (
            allow_menu_days,
            allow_routine_days,
            est_delta_kcal,
            preferred_cardio,
            preferred_strength,
            exercise_experience,
            other_flags,
        ) = row

        # CLOB / JSON 안전 처리
        def _load(v, default):
            if v is None:
                return default
            if hasattr(v, "read"):
                v = v.read()
            try:
                return json.loads(v)
            except Exception:
                return default

        return {
            "allow_menu_chg_plt_days": allow_menu_days,
            "allow_routine_chg_plt_days": allow_routine_days,
            "est_delta_kcal": est_delta_kcal,
            "preferred_cardio": _load(preferred_cardio, []),
            "preferred_strength": _load(preferred_strength, []),
            "exercise_experience": exercise_experience,
            "flags": _load(other_flags, {}),
        }
