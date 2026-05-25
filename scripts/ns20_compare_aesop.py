"""NS20 Stage 2 — Finset/aesop signal extraction and comparison.

For each NS20 surface, compares finset_aesop_only against:
  - raw NS15 routed
  - NS9 best wrapper + NS15 routed

Emits per-set:
  - proved counts
  - wrapper-only-new = variant_solved − raw_solved
  - new beyond NS9 = variant_solved − wrap_solved
  - regressed = wrap_solved − variant_solved
  - winning tactic for each new theorem (greedy family classifier)

Aggregates per surface into
project/data/ns20_aesop_signal_meta.json and writes
project/evolve/reports/ns20_finset_aesop_comparison.md.
"""
from __future__ import annotations

import glob
import json
import re
from collections import defaultdict
from pathlib import Path


SURFACES = (
    "ns20_finset_aesop_extra_easy",
    "ns20_finset_aesop_extra_medium",
    "ns20_finset_aesop_extra_hard",
)


def first_match(pattern: str) -> str | None:
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def load_per_thm(path: str) -> dict[str, dict]:
    return {t["full_name"]: t for t in
            json.loads(Path(path).read_text(encoding="utf-8")).get("per_theorem", [])}


def load_solved(path: str) -> set[str]:
    return {t["full_name"] for t in
            json.loads(Path(path).read_text(encoding="utf-8")).get("per_theorem", [])
            if t.get("finished")}


def family_of(t: str) -> str:
    t = re.sub(r"\s+", " ", (t or "").strip())
    if not t: return "empty"
    if t == "omega": return "fallback_omega"
    if t == "aesop": return "aesop"
    if t == "decide": return "fallback_decide"
    if t == "rfl": return "fallback_rfl"
    if t.startswith("constructor") and "omega" in t: return "constructor_omega"
    if t.startswith("split_ifs"): return "split_ifs"
    if "fun h => by omega" in t and t.count("by omega") >= 2: return "iff_omega_pair"
    if t.startswith("simp_all"): return "simp_all"
    if t.startswith("simp"): return "simp_other"
    if t.startswith("rw"): return "rw_named"
    if t.startswith("exact"): return "exact_named"
    if t.startswith("apply"): return "apply_named"
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
    for s in SURFACES:
        vp = first_match(f"project/evolve/eval_runs/ns20_finset_aesop_only_wrapper_{s}/eval-*/metrics.json")
        rp = first_match(f"project/evolve/eval_runs/ns20_raw_{s}/eval-*/metrics.json")
        wp = first_match(f"project/evolve/eval_runs/ns20_ns9wrap_{s}/eval-*/metrics.json")
        if not vp:
            continue
        v_thms = load_per_thm(vp)
        v_solved = {n for n, t in v_thms.items() if t.get("finished")}
        raw_solved = load_solved(rp) if rp else set()
        wrap_solved = load_solved(wp) if wp else set()

        wrapper_only = sorted(v_solved - raw_solved)
        new_vs_wrap = sorted(v_solved - wrap_solved)
        regressed = sorted(wrap_solved - v_solved)

        new_wins_families: dict[str, list[str]] = defaultdict(list)
        for thm in new_vs_wrap:
            tac = winning_tactic(v_thms.get(thm) or {})
            fam = family_of(tac) if tac else "unknown"
            new_wins_families[fam].append(thm)

        out[s] = {
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
    return out


def render_md(report: dict) -> str:
    lines: list[str] = []
    lines.append("# NS20 — Finset/aesop signal comparison\n")
    lines.append("Variant: `ns19_finset_aesop_only` (unchanged from NS19).\n")
    lines.append("| set | proved | raw | wrap | Δraw | Δwrap | new beyond NS9 (family) |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    new_total = 0
    reg_total = 0
    for s in SURFACES:
        row = report.get(s)
        if not row:
            continue
        d_raw = row["variant_solved"] - row["raw_solved"]
        d_wrap = row["variant_solved"] - row["wrap_solved"]
        new_total += row["new_vs_wrap_count"]
        reg_total += row["regressed_count"]
        if row["new_vs_wrap"]:
            entries = []
            for fam, thms in row["new_wins_by_family"].items():
                for thm in thms:
                    entries.append(f"`{thm}` ({fam})")
            new_s = ", ".join(entries)
        else:
            new_s = ""
        lines.append(
            f"| `{s}` | {row['variant_solved']} | {row['raw_solved']} | "
            f"{row['wrap_solved']} | {d_raw:+d} | {d_wrap:+d} | {new_s} |"
        )
    lines.append(f"\n**Total Δwrap across NS20 surfaces: {new_total} | regressions: {reg_total}**\n")
    return "\n".join(lines) + "\n"


def main() -> None:
    report = analyze()
    Path("project/data").mkdir(parents=True, exist_ok=True)
    meta_path = Path("project/data/ns20_aesop_signal_meta.json")
    meta_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = render_md(report)
    md_path = Path("project/evolve/reports/ns20_finset_aesop_comparison.md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md, encoding="utf-8")
    print(f"{'set':>40} {'proved':>6} {'Δwrap':>6} {'regress':>8}")
    for s, row in report.items():
        d_wrap = row["variant_solved"] - row["wrap_solved"]
        print(f"{s:>40} {row['variant_solved']:>6} {d_wrap:>+6} {row['regressed_count']:>8}")
    print()
    print(f"wrote {meta_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
