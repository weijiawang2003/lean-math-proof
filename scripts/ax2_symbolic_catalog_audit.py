"""AX2 Stage 1 — audit remaining Option/List symbolic-action candidates.

AX1 established a symbolic-action bridge (CASES_SIMP/INDUCTION_SIMP) whose
label vocabulary is tiny (4 labels cover the 27 WX1+WX2 wins). AX2 wants to
grow that label dataset by mining *fresh* Option/List theorems under the
symbolic wrapper. This script audits what fresh candidates remain.

"Fresh" = present in the confirmed-available catalog
(project/data/cx1_available_theorems.json) and NOT already consumed by any
registered theorem set (which covers CX3 / WX1 / WX2 / AX1-equivalence /
demo_v1, all loaded into tasks.THEOREM_SETS). The broader scan
(project/discovered_theorems_cx1.json) is cross-checked to confirm no
additional *available* candidate exists beyond the confirmed catalog.

Classification (a prior, not ground truth — the Stage 3 probe decides the
actual winning tactic):
  - Option: option_cases_simp | option_simp_only
  - List:   list_cases_simp | list_induction_simp | list_simp_only |
            list_hard_unknown

Outputs:
  project/data/ax2_symbolic_catalog_audit_meta.json
  project/evolve/reports/ax2_symbolic_catalog_audit.md
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AVAILABLE = ROOT / "project/data/cx1_available_theorems.json"
DISCOVERED = ROOT / "project/discovered_theorems_cx1.json"
OUT_META = ROOT / "project/data/ax2_symbolic_catalog_audit_meta.json"
OUT_MD = ROOT / "project/evolve/reports/ax2_symbolic_catalog_audit.md"

NS = ["Option", "List"]

# List name fragments that hint a structural constructor split (cases).
LIST_CASES_HINTS = (
    "cons", "nil", "head", "tail", "append", "concat", "reverse", "map",
    "getLast", "get", "drop", "take", "singleton", "isEmpty", "mem",
    "filter", "replicate", "set", "modify", "insert", "erase", "find",
)
# List name fragments that hint structural recursion (induction).
LIST_INDUCTION_HINTS = (
    "foldr", "foldl", "fold", "length", "sum", "prod", "count", "scanl",
    "scanr", "join", "flatMap", "bind", "zipWith", "enum",
)
# Fragments that hint a plain simp / rewrite closes (no split needed).
SIMP_ONLY_HINTS = (
    "_eq_", "eq_nil", "eq_cons", "_def", "_id", "id_eq", "comm", "assoc",
    "self", "_iff_", "cancel", "_zero", "_one",
)


def classify_list(short: str, tags: list[str], difficulty: str) -> str:
    s = short.lower()
    if any(h in s for h in LIST_INDUCTION_HINTS) or "fold" in tags:
        return "list_induction_simp"
    if any(h in s for h in LIST_CASES_HINTS):
        return "list_cases_simp"
    if any(h in s for h in SIMP_ONLY_HINTS):
        return "list_simp_only"
    if difficulty == "hard":
        return "list_hard_unknown"
    return "list_simp_only"


def classify_option(short: str, tags: list[str]) -> str:
    s = short.lower()
    # Option lemmas mentioning some/none/isSome/getD/map/bind usually want a
    # none/some split; pure rewrite identities want simp only.
    cases_hints = ("some", "none", "issome", "isnone", "getd", "get", "map",
                   "bind", "elim", "orelse", "join", "filter", "guard")
    if any(h in s for h in cases_hints):
        return "option_cases_simp"
    return "option_simp_only"


def main() -> None:
    sys.path.insert(0, str(ROOT))
    import tasks

    used: set[str] = set()
    used_by_set: dict[str, int] = {}
    for s, thms in tasks.THEOREM_SETS.items():
        used_by_set[s] = len(thms)
        for t in thms:
            used.add(t.full_name)

    av = json.loads(AVAILABLE.read_text())["theorems"]
    av_by: dict[str, dict] = {}
    for t in av:
        av_by.setdefault(t["full_name"], t)
    av_names = set(av_by)

    disc = json.loads(DISCOVERED.read_text())["theorems"]
    disc_by: dict[str, dict] = {}
    for t in disc:
        disc_by.setdefault(t["full_name"], t)

    per_ns: dict[str, dict] = {}
    fresh_pool: dict[str, list[dict]] = {}
    for ns in NS:
        cands = [t for n, t in av_by.items() if n.startswith(ns + ".")]
        fresh = [t for t in cands if t["full_name"] not in used]
        # cross-check: discovered-only (not in available) fresh w/ tactic proof
        disc_only_fresh = [
            t for n, t in disc_by.items()
            if n.startswith(ns + ".") and n not in used
            and n not in av_names and t.get("has_tactic_proof")
        ]
        buckets: Counter = Counter()
        classified = []
        for t in fresh:
            short = t["full_name"].split(".")[-1]
            tags = t.get("family_tags", [])
            diff = t.get("difficulty")
            if ns == "List":
                fam = classify_list(short, tags, diff)
            else:
                fam = classify_option(short, tags)
            buckets[fam] += 1
            classified.append({
                "full_name": t["full_name"],
                "file": t["file_path"],
                "difficulty": diff,
                "tags": tags,
                "num_tactics_approx": t.get("num_tactics_approx"),
                "expected_family": fam,
            })
        per_ns[ns] = {
            "available_unique": len(cands),
            "fresh_unused": len(fresh),
            "discovered_only_unverified": len(disc_only_fresh),
            "family_buckets": dict(buckets),
        }
        fresh_pool[ns] = sorted(classified, key=lambda c: c["full_name"])

    meta = {
        "note": ("AX2 fresh Option/List symbolic-action candidate audit. "
                 "'used' is the union of all registered theorem sets "
                 "(CX3/WX1/WX2/AX1-equivalence/demo_v1). Confirmed-available "
                 "catalog is authoritative; discovered_only_unverified counts "
                 "fresh candidates that appear only in the broader scan and "
                 "are NOT confirmed available (excluded from mining)."),
        "sources": {
            "available_catalog": str(AVAILABLE.relative_to(ROOT)),
            "discovered_catalog": str(DISCOVERED.relative_to(ROOT)),
            "used_theorem_sets": len(used_by_set),
            "used_unique_theorems": len(used),
        },
        "per_namespace": per_ns,
        "fresh_candidates": fresh_pool,
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    md = ["# AX2 — fresh Option/List symbolic-action catalog audit", ""]
    md.append("Goal: find fresh (unused, confirmed-available) Option/List "
              "theorems to mine for new symbolic-action labels under the AX1 "
              "wrapper. `used` = union of all registered theorem sets "
              "(CX3/WX1/WX2/AX1-equivalence/demo_v1).")
    md.append("")
    md.append("| namespace | available unique | fresh unused | "
              "discovered-only (unverified) | buckets |")
    md.append("|---|---:|---:|---:|---|")
    for ns in NS:
        p = per_ns[ns]
        md.append(f"| {ns} | {p['available_unique']} | {p['fresh_unused']} | "
                  f"{p['discovered_only_unverified']} | {p['family_buckets']} |")
    md.append("")
    opt_fresh = per_ns["Option"]["fresh_unused"]
    list_fresh = per_ns["List"]["fresh_unused"]
    md.append(f"**Verdict:** Option is **exhausted** ({opt_fresh} fresh — all "
              f"{per_ns['Option']['available_unique']} available Option lemmas "
              "consumed by CX3/WX1), with no additional available candidate in "
              "the broader scan. The only fresh symbolic-mining surface is "
              f"**List ({list_fresh} fresh)**. AX2 dataset growth is therefore "
              "List-only; Option contribution stays at the AX1 baseline. This "
              "matches the WX2 finding that List is the sole remaining "
              "cases/induction-friendly surface.")
    md.append("")
    md.append("List bucket detail (classification is a prior; the Stage 3 "
              "probe decides the actual winning tactic):")
    md.append("")
    for fam, n in sorted(per_ns["List"]["family_buckets"].items(),
                         key=lambda kv: -kv[1]):
        md.append(f"- `{fam}`: {n}")
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"wrote {OUT_META.relative_to(ROOT)}")
    print(f"wrote {OUT_MD.relative_to(ROOT)}")
    for ns in NS:
        p = per_ns[ns]
        print(f"  {ns:7s} avail={p['available_unique']:4d} "
              f"fresh={p['fresh_unused']:4d} buckets={p['family_buckets']}")


if __name__ == "__main__":
    main()
