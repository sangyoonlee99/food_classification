# infra/repositories/meal_repository.py
from infra.db_server import get_db_conn

class MealRepository:

    def insert_meal(self, meal):
        sql = """
        INSERT INTO meal_record
        (meal_id, user_id, eaten_at, meal_type, food_name, amount_g, nutrition_snapshot)
        VALUES
        (:meal_id, :user_id, :eaten_at, :meal_type, :food_name, :amount_g, :nutrition)
        """
        with get_db_conn() as conn:
            conn.cursor().execute(sql, {
                "meal_id": meal["meal_id"],
                "user_id": meal["user_id"],
                "eaten_at": meal["eaten_at"],
                "meal_type": meal["meal_type"],
                "food_name": meal["food_name"],
                "amount_g": meal["amount_g"],
                "nutrition": meal["nutrition_snapshot"]
            })
            conn.commit()

    def load_meals_by_date(self, user_id, start_date, end_date):
        sql = """
        SELECT *
        FROM meal_record
        WHERE user_id = :user_id
          AND eaten_at BETWEEN :start_date AND :end_date
        ORDER BY eaten_at
        """
        with get_db_conn() as conn:
            cur = conn.cursor()
            cur.execute(sql, {
                "user_id": user_id,
                "start_date": start_date,
                "end_date": end_date
            })
            return cur.fetchall()
