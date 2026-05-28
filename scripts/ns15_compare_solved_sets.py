"""NS15 — per-theorem solved-set comparison across NS11/NS12/NS13/NS15.

Reads ``metrics.json`` from every eval-run we care about and emits a
markdown report covering:

  - per-set per-model proved count and oracle union
  - intersection across all listed models per set
  - exclusive wins per model (only X proves this)
  - NS15 transfer analysis on ns14_nat_extra:
      * which of the 8 NS14 wrapper-only Nat wins became raw NS15 wins?
      * which wrapper-only iff-omega patterns made it into the model?
  - demo_v1 retention check: did simp [Set.subset_def] survive?
  - ns14_set_finset_extra retention check

Mirrors ``scripts/ns13_compare_model_union.py`` but with NS15
checkpoints added and the NS14 sets folded in.

Run::

    python scripts/ns15_compare_solved_sets.py

Output markdown: ``project/evolve/reports/ns15_model_union_analysis.md``
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path


def _glob_metrics(prefix: str) -> str:
    return f"project/evolve/eval_runs/{prefix}/*/metrics.json"


# (model tag, glob for metrics.json). Each set lists every model we
# expect a metrics file for; missing files are reported as gaps.
MODELS_BY_SET: dict[str, dict[str, str]] = {
    "nat_defs_medium": {
        "gen_v5":                    _glob_metrics("gen_v5_raw_medium"),
        "ns11_combined":             _glob_metrics("gen_v5_ns11_combined_raw_nat_defs_medium"),
        "ns12_balanced":             _glob_metrics("gen_v5_ns12_balanced_raw_medium"),
        "ns13_routed":               _glob_metrics("gen_v5_routed_raw_medium"),
        "ns15_combined_all":         _glob_metrics("gen_v5_ns15_combined_all_raw_medium"),
        "ns15_nat_oversample":       _glob_metrics("gen_v5_ns15_nat_oversample_raw_medium"),
        "ns15_balanced_namespace":   _glob_metrics("gen_v5_ns15_balanced_namespace_raw_medium"),
        "ns15_curriculum":           _glob_metrics("gen_v5_ns15_curriculum_raw_medium"),
        "ns15_routed":               _glob_metrics("gen_v5_ns15_routed_raw_medium"),
    },
    "nat_defs_large_v5": {
        "ns11_combined":             _glob_metrics("gen_v5_ns11_combined_raw_nat_defs_large_v5"),
        "ns12_balanced":             _glob_metrics("gen_v5_ns12_balanced_raw_large"),
        "ns13_routed":               _glob_metrics("gen_v5_routed_raw_large"),
        "ns15_combined_all":         _glob_metrics("gen_v5_ns15_combined_all_raw_large"),
        "ns15_nat_oversample":       _glob_metrics("gen_v5_ns15_nat_oversample_raw_large"),
        "ns15_balanced_namespace":   _glob_metrics("gen_v5_ns15_balanced_namespace_raw_large"),
        "ns15_curriculum":           _glob_metrics("gen_v5_ns15_curriculum_raw_large"),
        "ns15_routed":               _glob_metrics("gen_v5_ns15_routed_raw_large"),
    },
    "demo_v1": {
        "gen_v5":                    _glob_metrics("gen_v5_raw_demo_v1"),
        "ns11_combined":             _glob_metrics("gen_v5_ns11_combined_raw_demo_v1"),
        "ns12_balanced":             _glob_metrics("gen_v5_ns12_balanced_raw_demo_v1"),
        "ns13_routed":               _glob_metrics("gen_v5_routed_raw_demo_v1"),
        "ns15_combined_all":         _glob_metrics("gen_v5_ns15_combined_all_raw_demo_v1"),
        "ns15_nat_oversample":       _glob_metrics("gen_v5_ns15_nat_oversample_raw_demo_v1"),
        "ns15_balanced_namespace":   _glob_metrics("gen_v5_ns15_balanced_namespace_raw_demo_v1"),
        "ns15_curriculum":           _glob_metrics("gen_v5_ns15_curriculum_raw_demo_v1"),
        "ns15_routed":               _glob_metrics("gen_v5_ns15_routed_raw_demo_v1"),
    },
    "ns14_nat_extra": {
        "ns13_routed":               _glob_metrics("ns14_routed_raw_nat"),
        "ns15_combined_all":         _glob_metrics("gen_v5_ns15_combined_all_raw_ns14_nat_extra"),
        "ns15_nat_oversample":       _glob_metrics("gen_v5_ns15_nat_oversample_raw_ns14_nat_extra"),
        "ns15_balanced_namespace":   _glob_metrics("gen_v5_ns15_balanced_namespace_raw_ns14_nat_extra"),
        "ns15_curriculum":           _glob_metrics("gen_v5_ns15_curriculum_raw_ns14_nat_extra"),
        "ns15_routed":               _glob_metrics("gen_v5_ns15_routed_raw_ns14_nat_extra"),
    },
    "ns14_set_finset_extra": {
        "ns13_routed":               _glob_metrics("ns14_routed_raw_set_finset"),
        "ns15_combined_all":         _glob_metrics("gen_v5_ns15_combined_all_raw_ns14_set_finset_extra"),
        "ns15_nat_oversample":       _glob_metrics("gen_v5_ns15_nat_oversample_raw_ns14_set_finset_extra"),
        "ns15_balanced_namespace":   _glob_metrics("gen_v5_ns15_balanced_namespace_raw_ns14_set_finset_extra"),
        "ns15_curriculum":           _glob_metrics("gen_v5_ns15_curriculum_raw_ns14_set_finset_extra"),
        "ns15_routed":               _glob_metrics("gen_v5_ns15_routed_raw_ns14_set_finset_extra"),
    },
    "ns14_mixed_easy": {
        "ns15_combined_all":         _glob_metrics("gen_v5_ns15_combined_all_raw_ns14_mixed_easy"),
        "ns15_nat_oversample":       _glob_metrics("gen_v5_ns15_nat_oversample_raw_ns14_mixed_easy"),
        "ns15_balanced_namespace":   _glob_metrics("gen_v5_ns15_balanced_namespace_raw_ns14_mixed_easy"),
        "ns15_curriculum":           _glob_metrics("gen_v5_ns15_curriculum_raw_ns14_mixed_easy"),
        "ns15_routed":               _glob_metrics("gen_v5_ns15_routed_raw_ns14_mixed_easy"),
    },
    "ns14_mixed_medium": {
        "ns15_combined_all":         _glob_metrics("gen_v5_ns15_combined_all_raw_ns14_mixed_medium"),
        "ns15_nat_oversample":       _glob_metrics("gen_v5_ns15_nat_oversample_raw_ns14_mixed_medium"),
        "ns15_balanced_namespace":   _glob_metrics("gen_v5_ns15_balanced_namespace_raw_ns14_mixed_medium"),
        "ns15_curriculum":           _glob_metrics("gen_v5_ns15_curriculum_raw_ns14_mixed_medium"),
        "ns15_routed":               _glob_metrics("gen_v5_ns15_routed_raw_ns14_mixed_medium"),
    },
}

# NS14 wrapper-only Nat wins (the 8 theorems from NS14 report — these
# are closed only by the wrapper + routed combo on NS13, never by raw
# routed). The NS15 transfer question: how many of these does any
# NS15 raw model close?
WRAPPER_ONLY_NAT_WINS_NS14 = {
    "Nat.pred_eq_succ_iff",
    "Nat.pred_sub",
    "Nat.lt_of_lt_pred",
    "Nat.lt_sub_iff_add_lt'",
    "Nat.sub_sub_sub_cancel_right",
    "Nat.add_sub_sub_cancel",
    "Nat.sub_add_sub_cancel",
    "Nat.sub_lt_sub_iff_right",
}

# Demo regression theorems (NS11 → NS12 lost these; NS12 balanced
# recovered them). We want to keep them solved by every NS15 variant.
DEMO_REGRESSION_TARGETS = {
    "Set.subset_univ",
    "Set.empty_subset",
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
        return {"set": set_name, "missing_models": missing, "models": {}}

    total = max(totals.values()) if totals else 0
    union: set[str] = set()
    for s in by_model.values():
        union |= s
    intersection: set[str] = (
        set.intersection(*by_model.values()) if by_model else set()
    )

    solvers_by_thm: dict[str, list[str]] = {}
    for thm in union:
        solvers_by_thm[thm] = sorted(
            tag for tag, s in by_model.items() if thm in s
        )
    exclusive: dict[str, list[str]] = defaultdict(list)
    for thm, ss in solvers_by_thm.items():
        if len(ss) == 1:
            exclusive[ss[0]].append(thm)

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
        "solvers_by_thm": {t: ss for t, ss in solvers_by_thm.items()},
    }


def analyze_transfer(reports: list[dict]) -> dict:
    """NS14 wrapper-only Nat transfer + demo retention."""
    nat_extra = next((r for r in reports if r["set"] == "ns14_nat_extra"),
                    None)
    demo = next((r for r in reports if r["set"] == "demo_v1"), None)
    set_finset = next((r for r in reports
                       if r["set"] == "ns14_set_finset_extra"), None)

    out: dict = {}

    if nat_extra and "by_model_solved" in nat_extra:
        out["wrapper_only_nat_transfer"] = {}
        for tag, solved in nat_extra["by_model_solved"].items():
            sset = set(solved)
            won = sorted(WRAPPER_ONLY_NAT_WINS_NS14 & sset)
            out["wrapper_only_nat_transfer"][tag] = {
                "wins": won,
                "count": len(won),
                "target_size": len(WRAPPER_ONLY_NAT_WINS_NS14),
            }

    if demo and "by_model_solved" in demo:
        out["demo_regression_retention"] = {}
        for tag, solved in demo["by_model_solved"].items():
            sset = set(solved)
            held = sorted(DEMO_REGRESSION_TARGETS & sset)
            out["demo_regression_retention"][tag] = {
                "retained": held,
                "count": len(held),
                "target_size": len(DEMO_REGRESSION_TARGETS),
            }

    if set_finset and "by_model_solved" in set_finset:
        out["set_finset_solved_counts"] = set_finset["models"]

    return out


def render_md(reports: list[dict], transfer: dict) -> str:
    out: list[str] = []
    out.append("# NS15 — Per-theorem solved-set comparison\n")
    out.append(
        "Offline analysis of which theorems each base / routed / "
        "NS15 model proves on each eval set. Includes the oracle "
        "union as an upper bound for a router restricted to the "
        "listed candidates.\n")

    for r in reports:
        out.append(f"## `{r['set']}` (total {r.get('total','?')})\n")
        if r.get("missing_models"):
            out.append("Missing metrics for: " + ", ".join(
                f"`{m}`" for m in r["missing_models"]) + "\n")
        if not r.get("models"):
            out.append("(no metrics found for any listed model)\n")
            continue
        out.append("| model | proved | union? | intersection? |")
        out.append("|---|---:|---:|---:|")
        for tag, n in r["models"].items():
            out.append(
                f"| `{tag}` | {n} | {r['union_size']} | {r['intersection_size']} |"
            )
        out.append("")

        out.append(
            f"**Oracle union (perfect router upper bound): "
            f"{r['union_size']}/{r.get('total','?')}**\n"
        )

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

    # transfer + retention
    out.append("## NS14 wrapper-only Nat transfer\n")
    if transfer.get("wrapper_only_nat_transfer"):
        out.append("Did any NS15 raw model learn the 8 NS14 wrapper-only "
                   "Nat wins?\n")
        out.append("| model | learned | / target |")
        out.append("|---|---:|---:|")
        for tag, info in transfer["wrapper_only_nat_transfer"].items():
            out.append(
                f"| `{tag}` | {info['count']} | {info['target_size']} |"
            )
        out.append("")
        out.append("Wins per model:")
        for tag, info in transfer["wrapper_only_nat_transfer"].items():
            ws = ", ".join(f"`{t}`" for t in info["wins"]) or "(none)"
            out.append(f"- `{tag}`: {ws}")
        out.append("")
    else:
        out.append("(no NS14 metrics — run Stage 4 evals first)\n")

    out.append("## demo_v1 regression retention\n")
    if transfer.get("demo_regression_retention"):
        out.append(
            "Are `Set.subset_univ` and `Set.empty_subset` still proved?\n"
        )
        out.append("| model | retained | / target |")
        out.append("|---|---:|---:|")
        for tag, info in transfer["demo_regression_retention"].items():
            out.append(
                f"| `{tag}` | {info['count']} | {info['target_size']} |"
            )
        out.append("")
    else:
        out.append("(no demo_v1 metrics)\n")

    if transfer.get("set_finset_solved_counts"):
        out.append("## ns14_set_finset_extra retention\n")
        out.append("| model | proved |")
        out.append("|---|---:|")
        for tag, n in transfer["set_finset_solved_counts"].items():
            out.append(f"| `{tag}` | {n} |")
        out.append("")

    return "\n".join(out) + "\n"


def main() -> None:
    reports = [analyze_set(name, models)
               for name, models in MODELS_BY_SET.items()]
    transfer = analyze_transfer(reports)
    md = render_md(reports, transfer)
    out_path = Path("project/evolve/reports/ns15_model_union_analysis.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(md)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
