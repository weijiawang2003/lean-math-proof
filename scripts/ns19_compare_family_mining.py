"""NS19 Stage 6 — wrapper-variant signal extraction and comparison.

For each NS19 variant (`finset_aesop_only`, `nat_simp_arith_targeted`)
and each surface (the two NS19 surfaces plus the preservation sets),
compares against the raw NS15 routed baseline and the NS9 best
wrapper baseline. For each (variant, set) pair, emits:
  - variant_solved theorems
  - wrapper-only-new = variant_solved − raw_solved
  - newly closed beyond NS9 wrapper = variant_solved − wrap_solved
  - regressed = wrap_solved − variant_solved (lost theorems)
  - tactic family of each new win (greedy family classifier reuse)

Aggregates per variant into project/data/ns19_wrapper_signal_meta.json
and writes a markdown summary at
project/evolve/reports/ns19_wrapper_variants_comparison.md.

Reads NS15 raw + NS9 wrapper baselines from earlier NS18 eval dirs
when available; falls back to the NS19 baseline eval dirs that
ns19_run_matrix.sh produces.
"""
from __future__ import annotations

import glob
import json
import re
from collections import defaultdict
from pathlib import Path


VARIANTS = (
    "finset_aesop_only",
    "nat_simp_arith_targeted",
)

# Sets evaluated per variant.
VARIANT_SETS = {
    "finset_aesop_only": (
        "ns19_finset_aesop_surface",
        "ns17_finset_extra",
        "ns17_set_extra",
        "nat_defs_medium",
        "demo_v1",
        "ns14_set_finset_extra",
    ),
    "nat_simp_arith_targeted": (
        "ns19_nat_simp_arith_replay",
        "nat_defs_medium",
        "demo_v1",
        "ns16_nat_div_mod_extra",
    ),
}


def first_match(pattern: str) -> str | None:
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def load_per_thm(path: str) -> dict[str, dict]:
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    return {t["full_name"]: t for t in m.get("per_theorem", [])}


def load_solved(path: str) -> set[str]:
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    return {t["full_name"] for t in m.get("per_theorem", [])
            if t.get("finished")}


def raw_metrics_path(set_name: str) -> str | None:
    # Prefer NS19's own raw run (built for the new surfaces), fall
    # back to NS15/NS17 raw runs for the preservation sets.
    candidates = [
        f"project/evolve/eval_runs/ns19_raw_{set_name}/eval-*/metrics.json",
    ]
    if set_name.startswith("ns17_"):
        candidates.append(
            f"project/evolve/eval_runs/ns17_ns15routed_raw_{set_name}/eval-*/metrics.json"
        )
    if set_name.startswith("ns16_"):
        candidates.append(
            f"project/evolve/eval_runs/ns16_ns15routed_raw_{set_name}/eval-*/metrics.json"
        )
    candidates.append(
        f"project/evolve/eval_runs/gen_v5_ns15_routed_raw_{set_name}/eval-*/metrics.json"
    )
    if set_name == "nat_defs_medium":
        candidates.append(
            "project/evolve/eval_runs/gen_v5_routed_raw_medium/eval-*/metrics.json"
        )
    for pat in candidates:
        m = first_match(pat)
        if m:
            return m
    return None


def wrap_metrics_path(set_name: str) -> str | None:
    # Prefer NS19's NS9-wrapper baseline (built for new surfaces),
    # fall back to NS15/NS17 wrapper runs for preservation sets.
    candidates = [
        f"project/evolve/eval_runs/ns19_ns9wrap_{set_name}/eval-*/metrics.json",
    ]
    if set_name.startswith("ns17_"):
        candidates.append(
            f"project/evolve/eval_runs/ns17_ns15routed_wrapper_{set_name}/eval-*/metrics.json"
        )
    if set_name.startswith("ns16_"):
        candidates.append(
            f"project/evolve/eval_runs/ns16_ns15routed_wrapper_{set_name}/eval-*/metrics.json"
        )
    candidates.append(
        f"project/evolve/eval_runs/gen_v5_ns15_routed_wrapper_{set_name}/eval-*/metrics.json"
    )
    for pat in candidates:
        m = first_match(pat)
        if m:
            return m
    return None


def variant_metrics_path(variant: str, set_name: str) -> str | None:
    return first_match(
        f"project/evolve/eval_runs/ns19_{variant}_wrapper_{set_name}/eval-*/metrics.json"
    )


# Minimal family classifier reused from NS17 / NS18.
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


def winning_tactic(blob: dict) -> str:
    return (
        blob.get("winning_tactic")
        or blob.get("last_tactic")
        or blob.get("tactic")
        or blob.get("proof")
        or blob.get("result_tactic")
        or ""
    )


def analyze() -> dict:
    out: dict[str, dict] = {}
    for variant in VARIANTS:
        var_summary: dict[str, dict] = {}
        for s in VARIANT_SETS[variant]:
            vp = variant_metrics_path(variant, s)
            rp = raw_metrics_path(s)
            wp = wrap_metrics_path(s)
            if not vp:
                continue
            v_thms = load_per_thm(vp)
            v_solved = {n for n, t in v_thms.items() if t.get("finished")}
            raw_solved = load_solved(rp) if rp else set()
            wrap_solved = load_solved(wp) if wp else set()

            # Wrapper-only (vs raw): theorems the variant proves that
            # raw NS15 routed does not.
            wrapper_only = sorted(v_solved - raw_solved)
            # New beyond NS9 wrap baseline.
            new_vs_wrap = sorted(v_solved - wrap_solved)
            regressed = sorted(wrap_solved - v_solved)

            new_wins_families: dict[str, list[str]] = defaultdict(list)
            for thm in new_vs_wrap:
                tac = winning_tactic(v_thms.get(thm) or {})
                fam = family_of(tac) if tac else "unknown"
                new_wins_families[fam].append(thm)

            var_summary[s] = {
                "variant_solved": len(v_solved),
                "raw_solved": len(raw_solved),
                "wrap_solved": len(wrap_solved),
                "wrapper_only_count": len(wrapper_only),
                "wrapper_only": wrapper_only,
                "new_vs_wrap_count": len(new_vs_wrap),
                "new_vs_wrap": new_vs_wrap,
                "regressed_count": len(regressed),
                "regressed": regressed,
                "new_wins_by_family": dict(new_wins_families),
                "raw_path": rp,
                "wrap_path": wp,
                "variant_path": vp,
            }
        out[variant] = var_summary
    return out


def render_md(report: dict) -> str:
    lines: list[str] = []
    lines.append("# NS19 — wrapper-variant signal comparison\n")
    lines.append("Per-variant, per-set summary:\n"
                 "- `proved`: this NS19 variant + NS15 routed\n"
                 "- `raw`: raw NS15 routed only\n"
                 "- `wrap`: NS9 best wrapper + NS15 routed\n"
                 "- Δraw = proved − raw (wrapper-only signal vs raw)\n"
                 "- Δwrap = proved − wrap (genuinely new beyond NS9)\n")

    for variant in VARIANTS:
        lines.append(f"## `{variant}`\n")
        if variant not in report or not report[variant]:
            lines.append("(no metrics available yet)\n")
            continue
        v = report[variant]
        lines.append("| set | proved | raw | wrap | Δraw | Δwrap | new beyond NS9 |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        new_vs_wrap_total = 0
        regressions_total = 0
        for s in VARIANT_SETS[variant]:
            row = v.get(s)
            if not row:
                continue
            d_raw = row["variant_solved"] - row["raw_solved"]
            d_wrap = row["variant_solved"] - row["wrap_solved"]
            new_vs_wrap_total += row["new_vs_wrap_count"]
            regressions_total += row["regressed_count"]
            new_s = ", ".join(f"`{t}`" for t in row["new_vs_wrap"]) or ""
            lines.append(
                f"| `{s}` | {row['variant_solved']} | {row['raw_solved']} | "
                f"{row['wrap_solved']} | {d_raw:+d} | {d_wrap:+d} | {new_s} |"
            )
        lines.append(
            f"\n**Total Δwrap (new beyond NS9) across sets: "
            f"{new_vs_wrap_total} | total regressions: {regressions_total}**\n"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    report = analyze()
    Path("project/data").mkdir(parents=True, exist_ok=True)
    meta_path = Path("project/data/ns19_wrapper_signal_meta.json")
    meta_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = render_md(report)
    md_path = Path("project/evolve/reports/ns19_wrapper_variants_comparison.md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md, encoding="utf-8")

    print(f"{'variant':>26} {'Δwrap':>6} {'regress':>8}")
    for variant, info in report.items():
        nv = sum(row["new_vs_wrap_count"] for row in info.values())
        reg = sum(row["regressed_count"] for row in info.values())
        print(f"{variant:>26} {nv:>6} {reg:>8}")
    print()
    print(f"wrote {meta_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
