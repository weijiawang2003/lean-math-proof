"""NS13 Stage 3 — per-theorem solved-set comparison and oracle union.

Reads per-theorem ``finished`` flags from every eval run we care
about, computes:

  - set of theorems solved by each model on each set
  - solved-by-all
  - per-model exclusive wins ("only X proves this")
  - lost-by-router (theorems solved by some single model but not
    by the routed policy)
  - oracle union (any-model-proves-it)

This is a pure offline analysis script — it does NOT call any
model. It only reads existing eval artifacts under
``project/evolve/eval_runs/``.

Output: a markdown report at
``project/evolve/reports/ns13_model_union_analysis.md``.
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from pathlib import Path


# (model tag, glob for eval dir containing metrics.json)
MODELS_BY_SET: dict[str, dict[str, str]] = {
    "nat_defs_medium": {
        "gen_v5":          "project/evolve/eval_runs/gen_v5_raw_medium/*/metrics.json",
        "ns11_combined":   "project/evolve/eval_runs/gen_v5_ns11_combined_raw_nat_defs_medium/*/metrics.json",
        "ns12_balanced":   "project/evolve/eval_runs/gen_v5_ns12_balanced_raw_medium/*/metrics.json",
        "ns12_replay":     "project/evolve/eval_runs/gen_v5_ns12_replay_raw_medium/*/metrics.json",
        "ns12_low_lr":     "project/evolve/eval_runs/gen_v5_ns12_low_lr_raw_medium/*/metrics.json",
        "routed":          "project/evolve/eval_runs/gen_v5_routed_raw_medium/*/metrics.json",
    },
    "nat_defs_large_v5": {
        "ns11_combined":   "project/evolve/eval_runs/gen_v5_ns11_combined_raw_nat_defs_large_v5/*/metrics.json",
        "ns12_balanced":   "project/evolve/eval_runs/gen_v5_ns12_balanced_raw_large/*/metrics.json",
        "ns12_replay":     "project/evolve/eval_runs/gen_v5_ns12_replay_raw_large/*/metrics.json",
        "routed":          "project/evolve/eval_runs/gen_v5_routed_raw_large/*/metrics.json",
    },
    "demo_v1": {
        "gen_v5":          "project/evolve/eval_runs/gen_v5_raw_demo_v1/*/metrics.json",
        "ns11_combined":   "project/evolve/eval_runs/gen_v5_ns11_combined_raw_demo_v1/*/metrics.json",
        "ns12_balanced":   "project/evolve/eval_runs/gen_v5_ns12_balanced_raw_demo_v1/*/metrics.json",
        "ns12_replay":     "project/evolve/eval_runs/gen_v5_ns12_replay_raw_demo_v1/*/metrics.json",
        "ns12_low_lr":     "project/evolve/eval_runs/gen_v5_ns12_low_lr_raw_demo_v1/*/metrics.json",
        "routed":          "project/evolve/eval_runs/gen_v5_routed_raw_demo_v1/*/metrics.json",
    },
}


def load_solved(metrics_path: str) -> tuple[set[str], int]:
    m = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    total = m.get("total_theorems") or len(m.get("per_theorem", []))
    solved = {t["full_name"] for t in m.get("per_theorem", [])
              if t.get("finished")}
    return solved, total


def first_match(pattern: str) -> str | None:
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def analyze_set(set_name: str, models: dict[str, str]) -> dict:
    by_model: dict[str, set[str]] = {}
    totals: dict[str, int] = {}
    missing: list[str] = []
    for tag, pattern in models.items():
        p = first_match(pattern)
        if not p:
            missing.append(tag)
            continue
        solved, total = load_solved(p)
        by_model[tag] = solved
        totals[tag] = total

    if not by_model:
        return {"set": set_name, "missing": missing, "models": {}}

    all_models = list(by_model.keys())
    total = max(totals.values())

    # Build per-theorem solver bitmap, only over the union of solved.
    union: set[str] = set()
    for s in by_model.values():
        union |= s
    intersection: set[str] = set.intersection(*by_model.values()) if by_model else set()

    # Per-theorem solver lists.
    solvers_by_thm: dict[str, list[str]] = {}
    for thm in union:
        solvers_by_thm[thm] = sorted(
            tag for tag in all_models if thm in by_model[tag]
        )

    # Per-model exclusive wins: thm solved by exactly this model.
    exclusive: dict[str, list[str]] = defaultdict(list)
    for thm, ss in solvers_by_thm.items():
        if len(ss) == 1:
            exclusive[ss[0]].append(thm)

    # router-specific lens: which thms does any single model solve
    # but the router does not?
    if "routed" in by_model:
        router = by_model["routed"]
        any_single = set()
        for tag, s in by_model.items():
            if tag == "routed":
                continue
            any_single |= s
        router_missed = sorted(any_single - router)
        router_wins_alone = sorted(
            t for t in router if all(t not in by_model.get(tag, set())
                                     for tag in all_models if tag != "routed")
        )
    else:
        router_missed = []
        router_wins_alone = []

    return {
        "set": set_name,
        "total": total,
        "missing_models": missing,
        "models": {tag: len(s) for tag, s in by_model.items()},
        "by_model_solved": {tag: sorted(s) for tag, s in by_model.items()},
        "union_size": len(union),
        "intersection_size": len(intersection),
        "intersection": sorted(intersection),
        "exclusive": {tag: sorted(v) for tag, v in exclusive.items()},
        "router_missed_some_single_solved": router_missed,
        "router_wins_alone": router_wins_alone,
    }


def render_md(reports: list[dict]) -> str:
    out: list[str] = []
    out.append("# NS13 — Per-theorem model union analysis\n")
    out.append("Offline diff of which theorems each base-model variant proves "
               "on each theorem set. Includes the oracle union (any-model-"
               "proves-it) as an upper bound for a perfect router.\n")

    for r in reports:
        out.append(f"## `{r['set']}` (total {r.get('total','?')})\n")
        if r.get("missing_models"):
            out.append("Missing models: " + ", ".join(
                f"`{m}`" for m in r["missing_models"]) + "\n")
        out.append("| model | proved | union? | intersection? |")
        out.append("|---|---:|---:|---:|")
        for tag, n in r["models"].items():
            out.append(f"| `{tag}` | {n} | {r['union_size']} | {r['intersection_size']} |")
        out.append("")

        out.append("**Oracle union (upper bound for a perfect router): "
                   f"{r['union_size']}/{r.get('total','?')}**\n")

        out.append("Solved by every model in the comparison "
                   f"(intersection size {r['intersection_size']}):")
        if r["intersection"]:
            for t in r["intersection"]:
                out.append(f"- `{t}`")
        else:
            out.append("- (none)")
        out.append("")

        out.append("Exclusive wins (this model proves it, no other does):")
        any_excl = False
        for tag, ts in r["exclusive"].items():
            if not ts:
                continue
            any_excl = True
            out.append(f"- `{tag}` ({len(ts)}): " +
                       ", ".join(f"`{t}`" for t in ts))
        if not any_excl:
            out.append("- (none)")
        out.append("")

        if "router_missed_some_single_solved" in r:
            missed = r["router_missed_some_single_solved"]
            if missed:
                out.append(f"**Router gap** — theorems some single model proves "
                           f"but routed does not ({len(missed)}):")
                for t in missed:
                    out.append(f"- `{t}`")
            else:
                out.append("**Router gap**: none — routed matches the union "
                           "of all single-model wins.")
            out.append("")
            wa = r["router_wins_alone"]
            if wa:
                out.append(f"Router-only wins ({len(wa)}): " +
                           ", ".join(f"`{t}`" for t in wa))
            out.append("")

    return "\n".join(out) + "\n"


def main() -> None:
    reports = [analyze_set(name, models)
               for name, models in MODELS_BY_SET.items()]
    md = render_md(reports)
    out_path = Path("project/evolve/reports/ns13_model_union_analysis.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(md)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
