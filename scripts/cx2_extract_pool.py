"""CX2 Stage 5 — extract Int iff_omega + fallback_omega pool metadata.

Aggregates the wrapper-only-vs-NS9 wins for the Int namespace from
CX1 (cx1_bool_option_int probe) and CX2 (all 4 cx2_int_* sets).
Classifies tactic family (iff_omega_pair vs fallback_omega vs other).

Writes:
  - project/data/cx2_int_iff_omega_pool_meta.json
  - project/evolve/reports/cx2_pool_summary.md
"""
from __future__ import annotations

import glob
import json
import re
from collections import defaultdict
from pathlib import Path


def fam(t: str) -> str:
    t = re.sub(r"\s+", " ", (t or "").strip())
    if not t:
        return "empty"
    if "fun h => by omega" in t and t.count("by omega") >= 2:
        return "iff_omega_pair"
    if t == "omega":
        return "fallback_omega"
    if t == "aesop":
        return "aesop"
    if t == "decide":
        return "fallback_decide"
    if t == "rfl":
        return "fallback_rfl"
    if t.startswith("simp_all"):
        return "simp_all"
    if t.startswith("simp"):
        return "simp_other"
    return "other"


def first_match(pat: str) -> str | None:
    ms = sorted(glob.glob(pat))
    return ms[0] if ms else None


def per_thm(p: str | None) -> dict[str, dict]:
    if not p:
        return {}
    return {t["full_name"]: t for t in json.load(open(p)).get(
        "per_theorem", [])}


def solved(per_thm_dict: dict[str, dict]) -> set[str]:
    return {n for n, t in per_thm_dict.items() if t.get("finished")}


# (arc, set_name, raw_pat, wrap_pat)
SOURCES = [
    # CX1 — bool_option_int probe was the only one that produced
    # Int wrapper-only wins.
    ("CX1", "cx1_bool_option_int",
     "project/evolve/eval_runs/cx1_raw_cx1_bool_option_int/eval-*/metrics.json",
     "project/evolve/eval_runs/cx1_ns9wrap_cx1_bool_option_int/eval-*/metrics.json"),
    # CX2 — 4 sets
    ("CX2", "cx2_int_iff_omega_easy",
     "project/evolve/eval_runs/cx2_raw_cx2_int_iff_omega_easy/eval-*/metrics.json",
     "project/evolve/eval_runs/cx2_ns9wrap_cx2_int_iff_omega_easy/eval-*/metrics.json"),
    ("CX2", "cx2_int_iff_omega_medium",
     "project/evolve/eval_runs/cx2_raw_cx2_int_iff_omega_medium/eval-*/metrics.json",
     "project/evolve/eval_runs/cx2_ns9wrap_cx2_int_iff_omega_medium/eval-*/metrics.json"),
    ("CX2", "cx2_int_order_arith",
     "project/evolve/eval_runs/cx2_raw_cx2_int_order_arith/eval-*/metrics.json",
     "project/evolve/eval_runs/cx2_ns9wrap_cx2_int_order_arith/eval-*/metrics.json"),
    ("CX2", "cx2_int_mixed",
     "project/evolve/eval_runs/cx2_raw_cx2_int_mixed/eval-*/metrics.json",
     "project/evolve/eval_runs/cx2_ns9wrap_cx2_int_mixed/eval-*/metrics.json"),
]


def main() -> None:
    pool: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"thms": {}, "sources": defaultdict(int)}
    )
    per_set_summary: list[dict] = []
    for arc, s, rp, wp in SOURCES:
        raw_match = first_match(rp)
        wrap_match = first_match(wp)
        if not raw_match or not wrap_match:
            continue
        raw_pt = per_thm(raw_match)
        wrap_pt = per_thm(wrap_match)
        raw_s = solved(raw_pt)
        wrap_s = solved(wrap_pt)
        wrap_only = wrap_s - raw_s
        only_int = {n for n in wrap_only if n.startswith("Int.")}
        per_set_summary.append({
            "arc": arc, "set": s,
            "raw_solved": len(raw_s), "wrap_solved": len(wrap_s),
            "wrap_only": len(wrap_only),
            "wrap_only_int": len(only_int),
        })
        for thm in only_int:
            blob = wrap_pt.get(thm) or {}
            tac = (blob.get("winning_tactic") or blob.get("last_tactic")
                   or blob.get("tactic") or "")
            f = fam(tac)
            ns = "Int"
            key = (f, ns)
            if thm not in pool[key]["thms"]:
                pool[key]["thms"][thm] = {
                    "winning_tactic": tac,
                    "first_seen_arc": arc,
                    "first_seen_set": s,
                }
            pool[key]["sources"][f"{arc}:{s}"] += 1

    out: dict = {
        "training_gate_unique_required": 5,
        "per_set_summary": per_set_summary,
        "families": {},
    }
    for (f, ns), info in pool.items():
        unique = len(info["thms"])
        if unique <= 1:
            osf = 20
        elif unique <= 3:
            osf = 15
        elif unique <= 6:
            osf = 10
        elif unique <= 12:
            osf = 5
        else:
            osf = 2
        out["families"][f"{f}|{ns}"] = {
            "family": f, "namespace": ns,
            "unique_count": unique,
            "trainable": unique >= 5,
            "recommended_oversample_factor": osf,
            "theorems": info["thms"],
            "source_breakdown": dict(info["sources"]),
        }
    out["families"] = dict(sorted(
        out["families"].items(),
        key=lambda kv: -kv[1]["unique_count"]
    ))

    out_path = Path("project/data/cx2_int_iff_omega_pool_meta.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    md = ["# CX2 — Int wrapper-only pool", ""]
    md.append("## Per-set mining summary")
    md.append("")
    md.append("| arc | set | raw | wrap | wrap-only | Int wrap-only |")
    md.append("|---|---|---:|---:|---:|---:|")
    for r in per_set_summary:
        md.append(f"| {r['arc']} | {r['set']} | {r['raw_solved']} | "
                  f"{r['wrap_solved']} | {r['wrap_only']} | "
                  f"**{r['wrap_only_int']}** |")
    md.append("")
    md.append("## Combined pool (Int namespace)")
    md.append("")
    md.append("| family | unique wins | gate met? | "
              "recommended oversample |")
    md.append("|---|---:|:---:|---:|")
    for k, info in out["families"].items():
        gate = "✓" if info["trainable"] else "✗"
        md.append(f"| `{info['family']}` | **{info['unique_count']}** | "
                  f"{gate} | {info['recommended_oversample_factor']}× |")
    md.append("")
    md.append("## Theorem detail")
    md.append("")
    for k, info in out["families"].items():
        md.append(f"### `{info['family']}` ({info['unique_count']} unique)")
        md.append("")
        md.append("| theorem | winning tactic | first seen |")
        md.append("|---|---|---|")
        for thm, m in info["theorems"].items():
            md.append(f"| `{thm}` | `{m['winning_tactic'][:60]}` | "
                      f"{m['first_seen_arc']}:{m['first_seen_set']} |")
        md.append("")

    rep = Path("project/evolve/reports/cx2_pool_summary.md")
    rep.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"wrote {out_path}")
    print(f"wrote {rep}")
    for k, info in out["families"].items():
        gate = "TRAIN" if info["trainable"] else " --  "
        print(f"  [{gate}] {k}: {info['unique_count']} unique")


if __name__ == "__main__":
    main()
