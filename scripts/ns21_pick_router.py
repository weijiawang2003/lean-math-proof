"""NS21 — choose the best Finset/aesop checkpoint for the router.

Reads raw_ckpt eval results for the three NS21 candidates on the
held-out Finset surfaces and writes ``project/evolve/routing/
ns21_router.json`` with the winning checkpoint plugged in.

The scoring rule (deliberately simple, surfaced in the report):

    score = raw_finset_wins − 0.5 × set_regressions
                                    − 0.5 × demo_regressions
                                    − 0.5 × nat_regressions

Where regression deltas are vs ``gen_v5_ns12_balanced`` on Set/demo,
and vs the routed NS15 baseline on Nat (Nat is routed through NS15,
so a single-checkpoint Nat regression doesn't necessarily mean
trouble — but we still penalize it as a sanity check).

Ties go to the smaller-oversample variant (less risk of
over-memorization).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


CANDIDATE_CKPTS = [
    "gen_v5_ns21_finset_aesop_10x",
    "gen_v5_ns21_finset_aesop_20x",
    "gen_v5_ns21_finset_aesop_minimal",
]

FINSET_SETS = [
    "ns17_finset_extra",
    "cx1_finset_image_filter",
    "ns20_finset_aesop_extra_easy",
    "ns20_finset_aesop_extra_medium",
    "ns20_finset_aesop_extra_hard",
]

SET_SETS = ["ns17_set_extra", "ns14_set_finset_extra"]
DEMO_SETS = ["demo_v1"]
NAT_SETS = ["nat_defs_medium", "nat_defs_large_v5"]

BASELINE_CKPTS = {
    "set": "gen_v5_ns12_balanced",
    "demo": "gen_v5_ns12_balanced",
}


def first_match(pat: str) -> str | None:
    ms = sorted(glob.glob(pat))
    return ms[0] if ms else None


def load_solved(p: str | None) -> set[str]:
    if not p: return set()
    return {t["full_name"] for t in
            json.loads(Path(p).read_text(encoding="utf-8")).get(
                "per_theorem", []) if t.get("finished")}


def count(ckpt: str, set_name: str) -> int | None:
    p = first_match(
        f"project/evolve/eval_runs/ns21_rawckpt_{ckpt}_{set_name}/"
        "eval-*/metrics.json"
    )
    if not p: return None
    return len(load_solved(p))


def score(ckpt: str) -> dict:
    detail = {"ckpt": ckpt, "finset": {}, "set": {}, "demo": {}, "nat": {}}
    finset_wins = 0
    for s in FINSET_SETS:
        n = count(ckpt, s)
        detail["finset"][s] = n
        if n is not None:
            finset_wins += n

    set_reg = 0
    for s in SET_SETS:
        n = count(ckpt, s)
        baseline_n = count(BASELINE_CKPTS["set"], s)
        detail["set"][s] = {"ckpt": n, "baseline": baseline_n}
        if n is not None and baseline_n is not None:
            set_reg += max(0, baseline_n - n)

    demo_reg = 0
    for s in DEMO_SETS:
        n = count(ckpt, s)
        baseline_n = count(BASELINE_CKPTS["demo"], s)
        detail["demo"][s] = {"ckpt": n, "baseline": baseline_n}
        if n is not None and baseline_n is not None:
            demo_reg += max(0, baseline_n - n)

    nat_reg = 0
    for s in NAT_SETS:
        n = count(ckpt, s)
        # Compare vs NS15 nat_oversample baseline if available.
        baseline_n = count("gen_v5_ns15_nat_oversample", s)
        detail["nat"][s] = {"ckpt": n, "baseline": baseline_n}
        if n is not None and baseline_n is not None:
            nat_reg += max(0, baseline_n - n)

    detail["finset_total_wins"] = finset_wins
    detail["set_regressions"] = set_reg
    detail["demo_regressions"] = demo_reg
    detail["nat_regressions"] = nat_reg
    detail["score"] = (
        finset_wins
        - 0.5 * set_reg
        - 0.5 * demo_reg
        - 0.5 * nat_reg
    )
    return detail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",
                    default="project/evolve/routing/ns21_router.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    scored = [score(c) for c in CANDIDATE_CKPTS]
    print("=== NS21 router candidate scoring ===")
    for s in scored:
        print(f"\n[{s['ckpt']}]  score={s['score']}")
        print(f"  finset wins        = {s['finset_total_wins']} "
              f"({s['finset']})")
        print(f"  set regressions    = {s['set_regressions']} "
              f"({s['set']})")
        print(f"  demo regressions   = {s['demo_regressions']} "
              f"({s['demo']})")
        print(f"  nat regressions    = {s['nat_regressions']} "
              f"({s['nat']})")

    # Sort by score desc, then by oversample factor asc (10 < 20 < minimal
    # because minimal has weak replay).
    rank = {
        "gen_v5_ns21_finset_aesop_10x": 0,
        "gen_v5_ns21_finset_aesop_20x": 1,
        "gen_v5_ns21_finset_aesop_minimal": 2,
    }
    scored.sort(key=lambda d: (-d["score"], rank.get(d["ckpt"], 99)))
    winner = scored[0]["ckpt"]
    print(f"\n=== chosen: {winner} ===")

    router = {
        "_about": (
            "NS21 domain-aware router. Routes Finset goals to the chosen "
            "NS21 Finset/aesop checkpoint, Nat goals to NS15 nat_oversample, "
            "and Set/default to NS12 balanced. Selection by "
            "scripts/ns21_pick_router.py."
        ),
        "_chosen_ckpt_score_detail": scored[0],
        "_all_candidates": scored,
        "routes": [
            {"pattern": "^Nat\\.",
             "ckpt_dir": "project/models/gen_v5_ns15_nat_oversample"},
            {"pattern": "^Finset\\.",
             "ckpt_dir": f"project/models/{winner}"},
            {"pattern": "^Set\\.",
             "ckpt_dir": "project/models/gen_v5_ns12_balanced"},
        ],
        "default_ckpt_dir": "project/models/gen_v5_ns12_balanced",
    }
    if args.dry_run:
        print(json.dumps(router, indent=2))
        return
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(router, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
