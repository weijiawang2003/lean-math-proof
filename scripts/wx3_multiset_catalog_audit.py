"""WX3 Stage 1 — Multiset catalog audit.

Audits the fresh, *confirmed-available* Multiset surface and classifies each
candidate by likely proof shape, so Stage 2 can carve disjoint theorem sets
targeted at the new state-aware Multiset wrapper actions
(MULTISET_INDUCTION_SIMP / EXT_SIMP / MULTISET_CASES_SIMP).

Sources:
  - project/data/cx1_available_theorems.json  (authoritative availability;
    every theorem here was actually probe-loaded under the pinned mathlib)
  - project/discovered_theorems_cx1.json       (broader discovered catalog,
    used only for the total-discovered count)
  - tasks.THEOREM_SETS                          (to exclude already-probed
    Multiset theorems and avoid duplicates)

We keep only `full_name` beginning with `Multiset.` (some live in the root
namespace, e.g. `_root_.Multiset.toFinset_card_eq_one_iff`, but the head is
still `Multiset.`). Availability is taken from cx1_available_theorems.json so
Stage 2 never emits an unavailable theorem.

Classification precedence (a name can match several keyword buckets; the
strongest structural signal wins):

  quotient  — Quot/induction_on territory: 'induction', 'rec', 'strongInduction'
  induction — recursive-structure lemmas:  cons, attach, bind, fold, sum, prod,
              foldr, foldl, join, pmap
  ext       — extensionality / count reasoning: ext, count, nodup, dedup,
              erase, union, inter, sub, le (antisymm-style)
  simp      — likely closed by simp/simp_all: zero, nil, empty, singleton,
              map, mem, card, coe, replicate, range, toList, toFinset
  hard      — everything else (unknown shape)

No neural training. Read-only audit; writes a small meta JSON + a report.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AVAIL = ROOT / "project/data/cx1_available_theorems.json"
DISCOVERED = ROOT / "project/discovered_theorems_cx1.json"
OUT_META = ROOT / "project/data/wx3_multiset_catalog_audit_meta.json"
OUT_REPORT = ROOT / "project/evolve/reports/wx3_multiset_catalog_audit.md"

# ----- classification keyword buckets (checked in this precedence) -----
QUOTIENT_KW = ("induction", "rec", "stronginduction", "quot")
INDUCTION_KW = ("cons", "attach", "bind", "foldr", "foldl", "fold", "sum",
                "prod", "join", "pmap", "scanl")
EXT_KW = ("ext", "count", "nodup", "dedup", "erase", "union", "inter",
          "antisymm")
SIMP_KW = ("zero", "nil", "empty", "singleton", "map", "mem", "card", "coe",
           "replicate", "range", "tolist", "tofinset", "le_", "filter",
           "add", "cons_zero")


def classify(short: str) -> str:
    s = short.lower()
    if any(k in s for k in QUOTIENT_KW):
        return "quotient"
    if any(k in s for k in INDUCTION_KW):
        return "induction"
    if any(k in s for k in EXT_KW):
        return "ext"
    if any(k in s for k in SIMP_KW):
        return "simp"
    return "hard"


def main() -> None:
    sys.path.insert(0, str(ROOT))
    import tasks

    avail = json.loads(AVAIL.read_text(encoding="utf-8"))
    theorems = avail["theorems"]

    discovered = json.loads(DISCOVERED.read_text(encoding="utf-8"))
    total_discovered = discovered.get("by_namespace_total", {}).get(
        "Multiset", 0)

    # Multiset candidates = available theorems whose head namespace is
    # Multiset (root-namespaced ones included via full_name prefix).
    ms = [t for t in theorems if str(t.get("full_name", "")).startswith(
        "Multiset.")]
    # de-dup by full_name (availability file can list a name twice)
    by_name: dict[str, dict] = {}
    for t in ms:
        by_name.setdefault(t["full_name"], t)
    ms = list(by_name.values())

    # Already-probed Multiset theorems (any existing theorem set).
    already: set[str] = set()
    for thms in tasks.THEOREM_SETS.values():
        for t in thms:
            if t.full_name.startswith("Multiset."):
                already.add(t.full_name)

    fresh = [t for t in ms if t["full_name"] not in already]

    # Classify the fresh, available surface.
    for t in fresh:
        short = t["full_name"].split(".")[-1]
        t["_shape"] = classify(short)

    cat_counts = Counter(t["_shape"] for t in fresh)
    diff_counts = Counter(t.get("difficulty", "?") for t in fresh)
    # cross-tab shape x difficulty
    shape_diff: dict[str, Counter] = {}
    for t in fresh:
        shape_diff.setdefault(t["_shape"], Counter())[
            t.get("difficulty", "?")] += 1

    # files
    file_counts = Counter(t.get("file_path", "?") for t in fresh)

    meta = {
        "source_available": str(AVAIL.relative_to(ROOT)),
        "mathlib_commit": discovered.get("mathlib_commit"),
        "total_multiset_discovered": total_discovered,
        "total_multiset_available": len(ms),
        "already_probed_multiset": sorted(already),
        "already_probed_count": len(already),
        "fresh_available_count": len(fresh),
        "category_counts": dict(cat_counts),
        "difficulty_counts": dict(diff_counts),
        "shape_x_difficulty": {k: dict(v) for k, v in shape_diff.items()},
        "top_files": file_counts.most_common(12),
        "fresh_candidates": sorted(
            ({"full_name": t["full_name"],
              "file": t.get("file_path"),
              "difficulty": t.get("difficulty", "?"),
              "num_tactics_approx": t.get("num_tactics_approx"),
              "family_tags": t.get("family_tags", []),
              "shape": t["_shape"]}
             for t in fresh),
            key=lambda c: (c["shape"], c["full_name"])),
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    # ---- recommended sets (by shape, easy-leaning where possible) ----
    def pick(shape, diffs):
        return [c for c in meta["fresh_candidates"]
                if c["shape"] == shape and c["difficulty"] in diffs]

    rec = {
        "wx3_multiset_simp_easy": len(pick("simp", ("easy", "medium"))),
        "wx3_multiset_induction_easy": len(pick("induction",
                                                ("easy", "medium"))),
        "wx3_multiset_ext_medium": len(pick("ext", ("easy", "medium",
                                                    "hard"))),
        "wx3_multiset_quotient_medium": len(pick("quotient",
                                                 ("easy", "medium", "hard"))),
    }

    lines = []
    lines.append("# WX3 Stage 1 — Multiset catalog audit\n")
    lines.append(f"- mathlib commit: `{meta['mathlib_commit']}`")
    lines.append(f"- total Multiset discovered: "
                 f"**{total_discovered}**")
    lines.append(f"- total Multiset confirmed-available: "
                 f"**{len(ms)}**")
    lines.append(f"- already probed (prior sets, excluded): "
                 f"**{len(already)}** "
                 f"({', '.join(sorted(already)) or 'none'})")
    lines.append(f"- **fresh available candidates: {len(fresh)}**\n")
    lines.append("## Category counts (by likely proof shape)\n")
    for k, v in cat_counts.most_common():
        lines.append(f"- `{k}`: {v}")
    lines.append("\n## Difficulty\n")
    for k, v in diff_counts.most_common():
        lines.append(f"- {k}: {v}")
    lines.append("\n## Shape × difficulty\n")
    lines.append("| shape | easy | medium | hard | ? |")
    lines.append("|---|---|---|---|---|")
    for shape in ("simp", "induction", "ext", "quotient", "hard"):
        c = shape_diff.get(shape, Counter())
        lines.append(f"| {shape} | {c.get('easy',0)} | {c.get('medium',0)} "
                     f"| {c.get('hard',0)} | {c.get('?',0)} |")
    lines.append("\n## Top source files\n")
    for f, n in file_counts.most_common(12):
        lines.append(f"- {n:>3}  {f}")
    lines.append("\n## Recommended theorem sets (target sizes)\n")
    for k, v in rec.items():
        lines.append(f"- `{k}`: ~{v} candidates")
    lines.append("\n_Generated by scripts/wx3_multiset_catalog_audit.py "
                 "(read-only; no training)._")
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {OUT_META.relative_to(ROOT)}")
    print(f"wrote {OUT_REPORT.relative_to(ROOT)}")
    print(f"Multiset: discovered={total_discovered} available={len(ms)} "
          f"already={len(already)} fresh={len(fresh)}")
    print("categories:", dict(cat_counts))
    print("recommended:", rec)


if __name__ == "__main__":
    main()
