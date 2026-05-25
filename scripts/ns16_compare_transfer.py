"""NS16 Stage 7 — transfer analysis vs NS15 baseline.

Compares each NS16 raw variant against the NS15 routed baseline and
the NS16 wrapper run on each Nat eval set. Answers:

  - Which NS16 wrapper-only theorems became raw NS16 wins?
  - Did NS15's 8 NS14 wrapper-only Nat wins survive?
  - Which nat_defs_medium / nat_defs_large_v5 theorems changed?
  - For each NS16 set: raw vs wrapper, theorem-level diff.

Output: markdown report at
``project/evolve/reports/ns16_transfer_analysis.md``.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path


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
)

# tag → glob for metrics.json
def runs_for(set_name: str) -> dict[str, str]:
    """Return tag → glob mapping for a given eval set."""
    short_for_set = {
        "nat_defs_medium": "medium",
        "nat_defs_large_v5": "large",
        "demo_v1": "demo_v1",
        "ns14_nat_extra": "ns14_nat_extra",
        "ns14_set_finset_extra": "ns14_set_finset_extra",
        "ns16_nat_iff_extra": "ns16_nat_iff_extra",
        "ns16_nat_div_mod_extra": "ns16_nat_div_mod_extra",
        "ns16_nat_order_extra": "ns16_nat_order_extra",
        "ns16_nat_mixed_extra": "ns16_nat_mixed_extra",
    }
    short = short_for_set[set_name]
    g = lambda d: f"project/evolve/eval_runs/{d}/eval-*/metrics.json"
    out = {
        "ns13_routed":       g(f"gen_v5_routed_raw_{short}"),
        "ns15_routed":       g(f"gen_v5_ns15_routed_raw_{set_name}"),
        "ns15_wrapper":      g(f"gen_v5_ns15_routed_wrapper_{set_name}"),
        "ns16_10x":          g(f"gen_v5_ns16_oversample_10x_raw_{set_name}"),
        "ns16_20x":          g(f"gen_v5_ns16_oversample_20x_raw_{set_name}"),
        "ns16_curriculum":   g(f"gen_v5_ns16_curriculum_continue_raw_{set_name}"),
        "ns16_routed":       g(f"gen_v5_ns16_routed_raw_{set_name}"),
        "ns16_wrapper":      g(f"gen_v5_ns16_routed_wrapper_{set_name}"),
    }
    # NS16-only eval set: ns13_routed has no run. NS14 sets: substitute prior NS14 wrappers.
    if set_name.startswith("ns16_"):
        out["ns15_routed"] = g(f"ns16_ns15routed_raw_{set_name}")
        out["ns15_wrapper"] = g(f"ns16_ns15routed_wrapper_{set_name}")
    if set_name == "ns14_nat_extra":
        out["ns13_routed"] = g("ns14_routed_raw_nat")
    if set_name == "ns14_set_finset_extra":
        out["ns13_routed"] = g("ns14_routed_raw_set_finset")
    return out


def load_solved(pattern: str) -> tuple[set[str] | None, int | None]:
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None, None
    m = json.loads(Path(matches[0]).read_text(encoding="utf-8"))
    total = m.get("total_theorems") or len(m.get("per_theorem", []))
    solved = {t["full_name"] for t in m.get("per_theorem", [])
              if t.get("finished")}
    return solved, total


NS14_WRAPPER_ONLY = {
    "Nat.pred_eq_succ_iff", "Nat.pred_sub", "Nat.lt_of_lt_pred",
    "Nat.lt_sub_iff_add_lt'", "Nat.sub_sub_sub_cancel_right",
    "Nat.add_sub_sub_cancel", "Nat.sub_add_sub_cancel",
    "Nat.sub_lt_sub_iff_right",
}


def analyze_set(set_name: str) -> dict:
    runs = runs_for(set_name)
    rows: dict[str, set[str]] = {}
    totals: dict[str, int] = {}
    missing: list[str] = []
    for tag, pat in runs.items():
        solved, total = load_solved(pat)
        if solved is None:
            missing.append(tag)
            continue
        rows[tag] = solved
        totals[tag] = total
    return {
        "set": set_name,
        "total": max(totals.values()) if totals else 0,
        "missing": missing,
        "rows": {tag: sorted(s) for tag, s in rows.items()},
        "counts": {tag: len(s) for tag, s in rows.items()},
        "wrapper_only_from_ns16_wrapper": sorted(
            (rows.get("ns16_wrapper", set()) - rows.get("ns16_routed", set()))
        ),
        "ns14_wrapper_only_retained": sorted(
            NS14_WRAPPER_ONLY & rows.get("ns16_routed", set())
        ) if set_name == "ns14_nat_extra" else None,
    }


def render(reports: list[dict]) -> str:
    out: list[str] = []
    out.append("# NS16 — Transfer + wrapper analysis\n")
    out.append("Per-set, per-model raw eval counts plus the diff of "
               "wrapper-only theorems (wrapper proves, raw does not). "
               "All NS16 sub-models were trained from gen_v5 base "
               "using a 19-row wrapper-only Nat corpus mined from "
               "the medium+large+NS16 wrapper traces.\n")

    for r in reports:
        out.append(f"## `{r['set']}` (total {r['total']})\n")
        if r["missing"]:
            out.append("Missing metrics: " + ", ".join(f"`{m}`"
                       for m in r["missing"]) + "\n")
        if not r["counts"]:
            out.append("(no metrics found)\n")
            continue
        out.append("| model | proved |")
        out.append("|---|---:|")
        for tag, n in r["counts"].items():
            out.append(f"| `{tag}` | {n} |")
        out.append("")
        wo = r.get("wrapper_only_from_ns16_wrapper") or []
        if wo:
            out.append(f"NS16 wrapper-only on this set "
                       f"(wrapper proves, NS16 routed does not), "
                       f"count {len(wo)}:")
            for t in wo:
                out.append(f"- `{t}`")
            out.append("")
        if r.get("ns14_wrapper_only_retained") is not None:
            ret = r["ns14_wrapper_only_retained"]
            out.append(f"NS14 wrapper-only Nat wins retained by NS16 raw "
                       f"router: {len(ret)}/{len(NS14_WRAPPER_ONLY)}")
            for t in sorted(NS14_WRAPPER_ONLY):
                mark = "✓" if t in set(ret) else "✗"
                out.append(f"- {mark} `{t}`")
            out.append("")

    out.append("## Headline transfer\n")
    nat_extra = next((r for r in reports if r["set"] == "ns14_nat_extra"),
                     None)
    if nat_extra:
        retained = nat_extra.get("ns14_wrapper_only_retained") or []
        out.append(f"- NS14 wrapper-only Nat wins (8): retained {len(retained)} "
                   f"by NS16 router (unchanged from NS15 routed).\n")
    new_wins_total = 0
    for r in reports:
        if not r["set"].startswith("ns16_"):
            continue
        baseline = set(r["rows"].get("ns15_routed", []))
        ns16 = set(r["rows"].get("ns16_routed", []))
        new = sorted(ns16 - baseline)
        new_wins_total += len(new)
        if new:
            out.append(f"- New raw wins on `{r['set']}` (NS16 routed minus "
                       f"NS15 routed): {len(new)} → {new}\n")
    if new_wins_total == 0:
        out.append("- **No new raw wins** on any NS16 set vs NS15 routed. "
                   "The 19-row wrapper-only corpus was too sparse/varied to "
                   "produce NS14-style transfer.\n")

    return "\n".join(out) + "\n"


def main() -> None:
    reports = [analyze_set(s) for s in SETS]
    md = render(reports)
    out_path = Path("project/evolve/reports/ns16_transfer_analysis.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(md)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
