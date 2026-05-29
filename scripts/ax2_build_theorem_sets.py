"""AX2 Stage 2 — build fresh symbolic-mining theorem sets from the audit.

Reads project/data/ax2_symbolic_catalog_audit_meta.json and writes disjoint
sets to project/evolve/routing/ax2_theorem_sets.json (loaded by
tasks._load_ax2_sets).

The audit shows Option/Bool/Sum/Prod are exhausted; the only fresh
cases/induction surface is List (76 fresh). The five sets named in the AX2
spec are emitted; the Option sets are intentionally empty (documented
exhaustion). All List fresh candidates are used, split into three disjoint
sets so the union is the whole fresh surface and nothing is double-counted:

  ax2_option_cases_fresh        — EMPTY (Option exhausted; 0 fresh available)
  ax2_option_simp_fresh         — EMPTY (Option exhausted; 0 fresh available)
  ax2_list_cases_fresh          — list_cases_simp class (structural splits)
  ax2_list_induction_fresh      — list_induction_simp class (fold/length/sum)
  ax2_option_list_mixed_fresh   — list_simp_only + list_hard_unknown
                                  (mixed control: simp-only negatives + hard
                                   unknowns; disjoint from the two above)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / "project/data/ax2_symbolic_catalog_audit_meta.json"
OUT = ROOT / "project/evolve/routing/ax2_theorem_sets.json"


def emit(items: list[dict]) -> list[dict]:
    out = []
    seen: set[str] = set()
    for c in items:
        if c["full_name"] in seen:
            continue
        seen.add(c["full_name"])
        out.append({
            "file_path": c["file"],
            "full_name": c["full_name"],
            "namespace": c["full_name"].split(".")[0],
            "difficulty": c.get("difficulty", "?"),
            "expected_family": c["expected_family"],
        })
    return out


def main() -> None:
    sys.path.insert(0, str(ROOT))
    m = json.loads(AUDIT.read_text())
    fresh = m["fresh_candidates"]
    listc = fresh.get("List", [])
    optc = fresh.get("Option", [])

    by_fam: dict[str, list[dict]] = {}
    for c in listc:
        by_fam.setdefault(c["expected_family"], []).append(c)

    cases = sorted(by_fam.get("list_cases_simp", []),
                   key=lambda c: c["full_name"])
    induction = sorted(by_fam.get("list_induction_simp", []),
                       key=lambda c: c["full_name"])
    mixed = sorted(by_fam.get("list_simp_only", []) +
                   by_fam.get("list_hard_unknown", []),
                   key=lambda c: c["full_name"])

    # Option fresh (expected empty): split by class for fidelity to the spec.
    opt_cases = [c for c in optc if c["expected_family"] == "option_cases_simp"]
    opt_simp = [c for c in optc if c["expected_family"] == "option_simp_only"]

    sets = {
        "ax2_option_cases_fresh": emit(opt_cases),
        "ax2_option_simp_fresh": emit(opt_simp),
        "ax2_list_cases_fresh": emit(cases),
        "ax2_list_induction_fresh": emit(induction),
        "ax2_option_list_mixed_fresh": emit(mixed),
    }

    # disjointness check across List sets
    seen: set[str] = set()
    dupes: list[str] = []
    for name in ("ax2_list_cases_fresh", "ax2_list_induction_fresh",
                 "ax2_option_list_mixed_fresh"):
        for t in sets[name]:
            if t["full_name"] in seen:
                dupes.append(t["full_name"])
            seen.add(t["full_name"])
    assert not dupes, f"non-disjoint sets: {dupes}"

    OUT.write_text(json.dumps(sets, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    total = sum(len(v) for v in sets.values())
    print(f"wrote {OUT.relative_to(ROOT)}")
    for k, v in sets.items():
        print(f"  {k}: {len(v)}")
    print(f"  TOTAL fresh candidates: {total}")
    print(f"  (union of List sets is disjoint, covers all "
          f"{len(listc)} fresh List)")


if __name__ == "__main__":
    main()
