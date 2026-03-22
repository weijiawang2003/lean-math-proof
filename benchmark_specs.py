"""Minimal benchmark specs for reproducible method comparison."""

BENCHMARK_SPECS = {
    "toy_v1": {
        "theorem_set": "toy_search",
        "rollout": {"max_steps": 5},
        "search": {"beam_width": 16, "max_depth": 4},
    },
    "nat_single_v1": {
        "theorem_set": "nat_single",
        "rollout": {"max_steps": 5},
        "search": {"beam_width": 16, "max_depth": 4},
    },
}
