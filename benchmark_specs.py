"""Minimal benchmark specs for reproducible method comparison."""

BENCHMARK_SPECS = {
    "toy_v1": {
        "theorem_set": "toy_search",
        "action_space": "core_v1",
        "rollout": {"max_steps": 5},
        "search": {"beam_width": 16, "max_depth": 4},
    },
    "nat_single_v1": {
        "theorem_set": "nat_single",
        "action_space": "core_v1",
        "rollout": {"max_steps": 5},
        "search": {"beam_width": 16, "max_depth": 4},
    },
    "nat_more_v2": {
        "theorem_set": "nat_more",
        "action_space": "search_v2",
        "rollout": {"max_steps": 8},
        "search": {"beam_width": 24, "max_depth": 6},
    },
    "mixed_easy_v2": {
        "theorem_set": "mixed_easy_v2",
        "action_space": "search_v2",
        "rollout": {"max_steps": 8},
        "search": {"beam_width": 24, "max_depth": 6},
    },
}
