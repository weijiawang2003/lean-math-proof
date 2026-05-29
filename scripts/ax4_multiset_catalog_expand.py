"""AX4 Stage 1 — broader Multiset catalog expansion.

AX3 exhausted the *confirmed-available* Multiset surface: of the 260 available
Multiset theorems (cx1_available_theorems.json), 259 are already consumed by
WX3 + AX3 theorem sets. To reach Green we must go beyond that to the broader
*discovered* catalog (project/discovered_theorems_cx1.json: 573 Multiset names
scanned from the pinned mathlib source), whose availability was never probed.

This script computes the frontier:

    frontier = discovered_Multiset
               - already_available_260   (all consumed by WX3/AX3)
               - every prior theorem set  (WX3/AX3/CX/WX/ns17/demo/...)
               - already-labelled names

Availability of frontier names is UNCONFIRMED here (the cx1 probe sampled only
a subset of files). We do not have a cheap offline availability oracle, so the
final availability check happens at mine time in Stage 3 (LeanDojo load), which
is timeout-guarded after the AX3 `eq_of_mem_map_const` REPL-hang incident.
What this stage does is rank the frontier by *induction-likelihood* so Stage 2
mines the highest-yield names first.

Confidence buckets (precedence top→bottom; a name lands in the first it hits):

  negative   — structurally NOT an induction_on+simp_all target: ext / count /
               nodup / dedup / antisymm / Rel / le-antisymm / quotient-eq.
  high       — strong recursive-structure signal that simp_all closes after
               cons-induction: cons, bind, fold(r/l), sum, prod, join, pmap,
               attach, scanl, foldr, foldl.
  medium     — plausible but often simp-closable without induction: map, mem,
               card, length, replicate, filterMap.
  hard       — induction-shaped but hard difficulty / mixed eq signal.

`cross_surface` is a flag, not a bucket: True when the name's file was barely
touched by prior mining (anything other than Multiset/Basic.lean, which WX3/AX3
mined heavily). Stage 2 uses it to carve the cross-surface set.

Read-only audit; writes a small meta JSON + a report. No training, no eval.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DISCOVERED = ROOT / "project/discovered_theorems_cx1.json"
AVAIL = ROOT / "project/data/cx1_available_theorems.json"
WX3_LABELS = ROOT / "project/data/wx3_minimal_multiset_labels.json"
AX3_LABELS = ROOT / "project/data/ax3_minimal_multiset_symbolic_labels.json"
OUT_META = ROOT / "project/data/ax4_multiset_catalog_expand_meta.json"
OUT_REPORT = ROOT / "project/evolve/reports/ax4_multiset_catalog_expand.md"

# heavily-mined file → anything else is "cross surface"
MINED_FILE = "Mathlib/Data/Multiset/Basic.lean"

# precedence-ordered keyword buckets (lowercased substring match on full_name)
NEGATIVE_KW = ("ext", "count", "nodup", "dedup", "antisymm", ".rel", "rel_",
               "_rel", "le_iff", "lt_iff", "subset", "disjoint")
HIGH_KW = ("cons", "bind", "foldr", "foldl", "fold", "sum", "prod", "join",
           "pmap", "attach", "scanl")
MEDIUM_KW = ("map", "mem", "card", "length", "replicate", "filtermap",
             "erase", "singleton", "add", "zero", "cons")


def bucket(full_name: str) -> str:
    s = full_name.lower()
    if any(k in s for k in NEGATIVE_KW):
        return "negative"
    if any(k in s for k in HIGH_KW):
        return "high"
    if any(k in s for k in MEDIUM_KW):
        return "medium"
    return "hard"


def _label_names(path: Path) -> set[str]:
    if not path.exists():
        return set()
    j = json.loads(path.read_text(encoding="utf-8"))
    rows = (j if isinstance(j, list) else
            j.get("relabel_results") or j.get("labels") or
            j.get("clean_labels") or j.get("rows") or [])
    out = set()
    for r in rows:
        if isinstance(r, dict):
            n = r.get("full_name") or r.get("theorem") or r.get("name")
            if n:
                out.add(n)
    return out


def main() -> None:
    sys.path.insert(0, str(ROOT))
    import tasks

    disc = json.loads(DISCOVERED.read_text(encoding="utf-8"))["theorems"]
    ms = {t["full_name"]: t for t in disc
          if "Multiset." in str(t.get("full_name", ""))}

    avail = json.loads(AVAIL.read_text(encoding="utf-8"))["theorems"]
    avail260 = {t["full_name"] for t in avail
                if "Multiset." in str(t.get("full_name", ""))}

    prior: set[str] = set()
    for thms in tasks.THEOREM_SETS.values():
        for t in thms:
            fn = getattr(t, "full_name", None) or (
                t.get("full_name") if isinstance(t, dict) else None)
            if fn:
                prior.add(fn)

    labeled = _label_names(WX3_LABELS) | _label_names(AX3_LABELS)

    frontier = [t for fn, t in ms.items()
                if fn not in avail260 and fn not in prior
                and fn not in labeled]

    for t in frontier:
        t["_bucket"] = bucket(t["full_name"])
        t["_cross_surface"] = t.get("file_path") != MINED_FILE

    bucket_counts = Counter(t["_bucket"] for t in frontier)
    file_counts = Counter(t.get("file_path", "?") for t in frontier)
    diff_counts = Counter(t.get("difficulty", "?") for t in frontier)
    cross_counts = Counter("cross" if t["_cross_surface"] else "basic"
                           for t in frontier)

    candidates = sorted(
        ({"full_name": t["full_name"],
          "file": t.get("file_path"),
          "difficulty": t.get("difficulty", "?"),
          "num_tactics_approx": t.get("num_tactics_approx"),
          "family_tags": t.get("family_tags", []),
          "bucket": t["_bucket"],
          "cross_surface": t["_cross_surface"]}
         for t in frontier),
        key=lambda c: (c["bucket"], not c["cross_surface"], c["full_name"]))

    meta = {
        "arc": "AX4",
        "stage": 1,
        "sources": {
            "discovered": str(DISCOVERED.relative_to(ROOT)),
            "available": str(AVAIL.relative_to(ROOT)),
        },
        "availability": "UNCONFIRMED for frontier; verified at mine time "
                        "(Stage 3 LeanDojo load, timeout-guarded).",
        "total_multiset_discovered": len(ms),
        "available_260_consumed_by_wx3_ax3": len(avail260),
        "prior_set_names": len(prior),
        "already_labeled": len(labeled),
        "frontier_count": len(frontier),
        "bucket_counts": dict(bucket_counts),
        "difficulty_counts": dict(diff_counts),
        "cross_surface_counts": dict(cross_counts),
        "file_counts": file_counts.most_common(20),
        "frontier_candidates": candidates,
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    lines = [
        "# AX4 Stage 1 — Multiset catalog expansion (broader discovered "
        "frontier)",
        "",
        f"AX3 consumed the confirmed-available Multiset surface "
        f"({len(avail260)} available, {len(avail260 & prior)} already in prior "
        f"sets). To reach Green we mine the broader discovered catalog.",
        "",
        f"- discovered Multiset names: **{len(ms)}**",
        f"- minus available-260 (consumed) + prior sets + labeled",
        f"- → **frontier = {len(frontier)}** availability-unconfirmed candidates",
        "",
        "## Buckets (induction-likelihood)",
        "",
        "| bucket | n |",
        "|---|---:|",
    ]
    for b in ("high", "medium", "hard", "negative"):
        lines.append(f"| {b} | {bucket_counts.get(b, 0)} |")
    lines += [
        "",
        f"- cross-surface (non-Basic file): "
        f"**{cross_counts.get('cross', 0)}**, basic: {cross_counts.get('basic', 0)}",
        f"- difficulty: {dict(diff_counts)}",
        "",
        "## Frontier by file",
        "",
        "| file | n |",
        "|---|---:|",
    ]
    for f, n in file_counts.most_common(20):
        lines.append(f"| `{f}` | {n} |")
    lines += [
        "",
        "## Caveat",
        "",
        "Frontier availability is UNCONFIRMED — these names were discovered by "
        "source scan but never probe-loaded. Stage 3 confirms availability at "
        "mine time (LeanDojo load), timeout-guarded per the AX3 REPL-hang "
        "incident. Expect attrition: some names will be unavailable / private "
        "/ deprecated and silently drop out.",
        "",
    ]
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote {OUT_META.relative_to(ROOT)}")
    print(f"wrote {OUT_REPORT.relative_to(ROOT)}")
    print(f"frontier={len(frontier)}  buckets={dict(bucket_counts)}  "
          f"cross={cross_counts.get('cross', 0)}")


if __name__ == "__main__":
    main()
