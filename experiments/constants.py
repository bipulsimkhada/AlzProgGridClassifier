RANDOM_STATE = 42

LOSS_SEARCH_STAGES = {
    "stage_1_severity": [
        {
            "name": "s1_severity_000",
            "time_weights": [1.0, 1.0, 1.0, 1.0],
            "severity_weight": 0.0,
            "transition_weight": 0.0,
            "transition_loss": "huber",
            "huber_delta": 1.0,
        },
        {
            "name": "s1_severity_025",
            "time_weights": [1.0, 1.0, 1.0, 1.0],
            "severity_weight": 0.25,
            "transition_weight": 0.0,
            "transition_loss": "huber",
            "huber_delta": 1.0,
        },
        {
            "name": "s1_severity_050",
            "time_weights": [1.0, 1.0, 1.0, 1.0],
            "severity_weight": 0.5,
            "transition_weight": 0.0,
            "transition_loss": "huber",
            "huber_delta": 1.0,
        },
        {
            "name": "s1_severity_075",
            "time_weights": [1.0, 1.0, 1.0, 1.0],
            "severity_weight": 0.75,
            "transition_weight": 0.0,
            "transition_loss": "huber",
            "huber_delta": 1.0,
        },
        {
            "name": "s1_severity_100",
            "time_weights": [1.0, 1.0, 1.0, 1.0],
            "severity_weight": 1.0,
            "transition_weight": 0.0,
            "transition_loss": "huber",
            "huber_delta": 1.0,
        },

    ],
    "stage_2_transition": [
        {
            "name": "s2_transition_000",
            "time_weights": [1.0, 1.0, 1.0, 1.0],
            "severity_weight": 0.0, # s1_severity_000 is better overall and for converter
            "transition_weight": 0.0,
            "transition_loss": "huber",
            "huber_delta": 1.0,
        },
        {
            "name": "s2_transition_025",
            "time_weights": [1.0, 1.0, 1.0, 1.0],
            "severity_weight": 0.0, # s1_severity_000 is better overall and for converter
            "transition_weight": 0.25,
            "transition_loss": "huber",
            "huber_delta": 1.0,
        },
        {
            "name": "s2_transition_050",
            "time_weights": [1.0, 1.0, 1.0, 1.0],
            "severity_weight": 0.0, # s1_severity_000 is better overall and for converter
            "transition_weight": 0.50,
            "transition_loss": "huber",
            "huber_delta": 1.0,
        },
        {
            "name": "s2_transition_075",
            "time_weights": [1.0, 1.0, 1.0, 1.0],
            "severity_weight": 0.0, # s1_severity_000 is better overall and for converter
            "transition_weight": 0.75,
            "transition_loss": "huber",
            "huber_delta": 1.0,
        },
        {
            "name": "s2_transition_100",
            "time_weights": [1.0, 1.0, 1.0, 1.0],
            "severity_weight": 0.0, # s1_severity_000 is better overall and for converter
            "transition_weight": 1.0,
            "transition_loss": "huber",
            "huber_delta": 1.0,
        },
        {
            "name": "s2_transition_150",
            "time_weights": [1.0, 1.0, 1.0, 1.0],
            "severity_weight": 0.0, # s1_severity_000 is better overall and for converter
            "transition_weight": 1.5,
            "transition_loss": "huber",
            "huber_delta": 1.0,
        },
        {
            "name": "s2_transition_200",
            "time_weights": [1.0, 1.0, 1.0, 1.0],
            "severity_weight": 0.0, # s1_severity_000 is better overall and for converter
            "transition_weight": 2.0,
            "transition_loss": "huber",
            "huber_delta": 1.0,
        },
        {
            "name": "s2_transition_300",
            "time_weights": [1.0, 1.0, 1.0, 1.0],
            "severity_weight": 0.0, # s1_severity_000 is better overall and for converter
            "transition_weight": 3.0,
            "transition_loss": "huber",
            "huber_delta": 1.0,
        },
    ]
}