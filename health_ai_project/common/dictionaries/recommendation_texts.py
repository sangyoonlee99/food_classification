# common/dictionaries/recommendation_texts.py

RECOMMENDATION_TEXTS = {
    "applied": {
        "macro": [
            "정체가 길어져 식단이나 운동 구성을 한 번 바꿔보는 것이 좋아요.",
            "새로운 메뉴나 루틴으로 몸에 새로운 자극을 주세요.",
        ],
        "default": [
            "오늘은 현재 계획을 유지해 주세요."
        ],
    },

    "partial": {
        "micro": [
            "오늘은 섭취 열량을 {kcal}kcal 정도 조정해보세요.",
            "여유가 된다면 가볍게 유산소 {cardio}분 정도만 해도 충분해요.",
        ],
    },

    "failed": {
        "default": [
            "오늘은 무리한 조정보다는 현재 루틴을 유지하는 것이 좋아요."
        ],
    },
}
