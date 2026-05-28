"""NS18 Stage 5 + 6 — wrapper-variant signal extraction and comparison.

For each NS18 wrapper variant, compares against:
  - raw NS15 routed (baseline raw),
  - NS9 best wrapper + NS15 routed (baseline wrapper).

For each (variant, set) pair, emits:
  - variant_solved theorems
  - wrapper-only-new = variant_solved − raw_solved (new wrapper signal)
  - regressed = baseline_wrapper_solved − variant_solved (lost theorems)
  - tactic family of each new win (greedy family classifier reuse)

Aggregates per variant into project/data/ns18_wrapper_signal_meta.json
and writes a markdown summary at
project/evolve/reports/ns18_wrapper_variants_comparison.md.
"""
from __future__ import annotations

import glob
import json
import re
from collections import defaultdict
from pathlib import Path


VARIANTS = (
    "constructor_omega", "split_ifs_omega", "nat_simp_arith",
    "aesop_wrapper", "bool_option_cases", "combined_safe",
)

# All sets we evaluated, in display order.
SETS = (
    "nat_defs_medium",
    "nat_defs_large_v5",
    "demo_v1",
    "ns14_nat_extra",
    "ns14_set_finset_extra",
    "ns16_nat_iff_extra",
    "ns16_nat_div_mod_extra",
    "ns16_nat_order_extra",
    "ns16_nat_mixed_extra",
    "ns17_set_extra",
    "ns17_finset_extra",
    "ns17_list_multiset",
    "ns17_nat_remaining",
)


def first_match(pattern: str) -> str | None:
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def load_per_thm(metrics_path: str) -> dict[str, dict]:
    m = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    return {t["full_name"]: t for t in m.get("per_theorem", [])}


def load_solved(metrics_path: str) -> set[str]:
    m = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    return {t["full_name"] for t in m.get("per_theorem", [])
            if t.get("finished")}


def baseline_raw_path(set_name: str) -> str | None:
    """Path to raw NS15 routed metrics for a given set."""
    candidates = []
    if set_name.startswith("ns16_"):
        candidates.append(f"project/evolve/eval_runs/ns16_ns15routed_raw_{set_name}/eval-*/metrics.json")
    if set_name.startswith("ns17_"):
        candidates.append(f"project/evolve/eval_runs/ns17_ns15routed_raw_{set_name}/eval-*/metrics.json")
    # The NS15 raw eval runs use the full set name in some cases.
    candidates.append(f"project/evolve/eval_runs/gen_v5_ns15_routed_raw_{set_name}/eval-*/metrics.json")
    # NS13 routed has shorter names for the canonical sets.
    if set_name == "nat_defs_medium":
        candidates.append("project/evolve/eval_runs/gen_v5_routed_raw_medium/eval-*/metrics.json")
    if set_name == "nat_defs_large_v5":
        candidates.append("project/evolve/eval_runs/gen_v5_routed_raw_large/eval-*/metrics.json")
    for pat in candidates:
        m = first_match(pat)
        if m:
            return m
    return None


def baseline_wrapper_path(set_name: str) -> str | None:
    """Path to NS9 wrapper + NS15 routed metrics for a given set."""
    candidates = []
    if set_name.startswith("ns16_"):
        candidates.append(f"project/evolve/eval_runs/ns16_ns15routed_wrapper_{set_name}/eval-*/metrics.json")
    if set_name.startswith("ns17_"):
        candidates.append(f"project/evolve/eval_runs/ns17_ns15routed_wrapper_{set_name}/eval-*/metrics.json")
    # The NS15 wrapper eval runs.
    candidates.append(f"project/evolve/eval_runs/gen_v5_ns15_routed_wrapper_{set_name}/eval-*/metrics.json")
    for pat in candidates:
        m = first_match(pat)
        if m:
            return m
    return None


def variant_path(variant: str, set_name: str) -> str | None:
    return first_match(
        f"project/evolve/eval_runs/ns18_{variant}_wrapper_{set_name}/eval-*/metrics.json"
    )


# Minimal family classifier reused from NS17.
def family_of(t: str) -> str:
    t = re.sub(r"\s+", " ", (t or "").strip())
    if not t: return "empty"
    if t == "omega": return "fallback_omega"
    if t == "aesop": return "fallback_aesop"
    if t == "decide": return "fallback_decide"
    if t == "rfl": return "fallback_rfl"
    if t.startswith("constructor") and "omega" in t:
        return "constructor_omega"
    if t.startswith("split_ifs"):
        return "split_ifs"
    if "fun h => by omega" in t and t.count("by omega") >= 2:
        return "iff_omega_pair"
    if t.startswith("simp_all"):
        return "simp_all"
    if t.startswith("simp"):
        return "simp_other"
    if t.startswith("rw"):
        return "rw_named"
    if t.startswith("exact"):
        return "exact_named"
    if t.startswith("apply"):
        return "apply_named"
    return "other"


def analyze() -> dict:
    out: dict[str, dict] = {}
    for variant in VARIANTS:
        var_summary: dict[str, dict] = {}
        for s in SETS:
            vp = variant_path(variant, s)
            rp = baseline_raw_path(s)
            wp = baseline_wrapper_path(s)
            if not vp:
                continue
            v_thms = load_per_thm(vp)
            v_solved = {n for n, t in v_thms.items() if t.get("finished")}
            raw_solved = load_solved(rp) if rp else set()
            wrap_solved = load_solved(wp) if wp else set()
            new_wins = sorted(v_solved - raw_solved)  # wrapper-only
            regressed = sorted(wrap_solved - v_solved)
            new_wins_families: dict[str, list[str]] = defaultdict(list)
            for thm in new_wins:
                tac = (v_thms.get(thm) or {}).get("winning_tactic") or \
                      (v_thms.get(thm) or {}).get("last_tactic") or ""
                # eval_rollout_all stores the winning tactic somewhere
                # less consistent; fall back to scanning the per-theorem
                # blob.
                if not tac:
                    blob = v_thms.get(thm) or {}
                    for k in ("tactic", "winning", "proof", "result_tactic"):
                        if blob.get(k):
                            tac = blob[k]; break
                fam = family_of(tac) if tac else "unknown"
                new_wins_families[fam].append(thm)
            var_summary[s] = {
                "variant_solved": len(v_solved),
                "raw_solved": len(raw_solved),
                "wrap_solved": len(wrap_solved),
                "wrapper_only_new_count": len(new_wins),
                "wrapper_only_new": new_wins,
                "regressed_count": len(regressed),
                "regressed": regressed,
                "new_wins_by_family": dict(new_wins_families),
            }
        out[variant] = var_summary
    return out


def render_md(report: dict) -> str:
    lines: list[str] = []
    lines.append("# NS18 — wrapper-variant signal comparison\n")
    lines.append("Per-variant, per-set summary of:\n"
                 "- `proved`: this variant + NS15 routed\n"
                 "- `vs raw`: variant proved − raw NS15 routed proved\n"
                 "- `vs wrap`: variant proved − NS9 wrapper baseline proved\n"
                 "- wrapper-only-new = theorems variant proves that "
                 "raw NS15 routed does not\n")

    for variant in VARIANTS:
        lines.append(f"## `{variant}`\n")
        if variant not in report:
            lines.append("(no metrics)\n")
            continue
        v = report[variant]
        lines.append("| set | proved | raw | wrap | Δraw | Δwrap | new wrapper-only |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        new_wins_total = 0
        for s in SETS:
            row = v.get(s)
            if not row:
                continue
            d_raw = row["variant_solved"] - row["raw_solved"]
            d_wrap = row["variant_solved"] - row["wrap_solved"]
            new = row["wrapper_only_new"]
            new_wins_total += len(new)
            new_s = ", ".join(f"`{t}`" for t in new) or ""
            lines.append(
                f"| `{s}` | {row['variant_solved']} | {row['raw_solved']} | "
                f"{row['wrap_solved']} | {d_raw:+d} | {d_wrap:+d} | {new_s} |"
            )
        lines.append(f"\n**Total wrapper-only-new (vs raw) across sets: {new_wins_total}**\n")
    return "\n".join(lines) + "\n"


def main() -> None:
    report = analyze()
    Path("project/data").mkdir(parents=True, exist_ok=True)
    meta_path = Path("project/data/ns18_wrapper_signal_meta.json")
    meta_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = render_md(report)
    md_path = Path("project/evolve/reports/ns18_wrapper_variants_comparison.md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md, encoding="utf-8")

    # CLI summary.
    print(f"{'variant':>20} {'fresh_wo':>10} {'regressions':>12}")
    for variant, info in report.items():
        new = sum(len(v["wrapper_only_new"]) for v in info.values())
        reg = sum(len(v["regressed"]) for v in info.values())
        print(f"{variant:>20} {new:>10} {reg:>12}")
    print()
    print(f"wrote {meta_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
