from events.event_types import EventType

EVENT_POLICY = {
    EventType.DINNER_OUT: {
        "diet": {
            "same_day": {
                "dinner": {"mode": "free", "replace": True},
                "lunch": {"carb_ratio": -0.4},
            },
            "next_day": {
                "kcal_delta": -200,
            },
        },
        "exercise": {
            "same_day": {"skip": True},
            "next_day": {"cardio_min": +20},
        },
    },

    EventType.EXTRA_EXERCISE: {
        "diet": {
            "same_day": {"carb_ratio": +0.2},
        },
        "exercise": {
            "same_day": {"intensity": "up"},
            "next_day": {"rest": True},
        },
    },

    EventType.TRAVEL: {
        "diet": {
            "during": {"free_ratio_up": True},
            "after": {"normalize": True},
        },
        "exercise": {
            "during": {"walk_based": True},
            "after": {"light_cardio": True},
        },
    },
}
