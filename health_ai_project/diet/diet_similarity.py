# diet/diet_similarity.py

def similarity_penalty(food_name: str, used_foods: dict) -> float:
    """
    주간 동일 음식 반복에 대한 soft penalty
    """
    count = used_foods.get(food_name, 0)

    if count <= 1:
        return 1.0        # 자유
    if count == 2:
        return 0.7        # 약한 패널티
    if count == 3:
        return 0.5
    return 0.3            # 강한 패널티 (완전 차단 ❌)
