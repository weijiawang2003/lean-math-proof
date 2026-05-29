"""WX2 Stage 3 — audit cases/induction-friendly surfaces beyond Option.

The WX1 state-aware Option cases wrapper added +19 Option wins. WX2 asks
whether the same `cases/induction <var> <;> simp` pattern generalizes.
This audits the *fresh* (unused) candidates in the inductive namespaces
the wrapper can target with a constructor split.

Namespaces considered: Option, List, Bool, Sum, Prod (Multiset is a
quotient type — `cases`/`induction` on a raw Multiset var does not apply,
so it is reported but excluded from the cases-friendly buckets).

Classifies each candidate by likely tactic:
  - list_cases / list_induction  (List structural lemmas)
  - bool_cases                   (Bool, decidable)
  - prod_cases / sum_cases       (×, ⊕ destructuring)
  - option_cases                 (Option none/some)

Outputs:
  project/data/wx2_cases_catalog_audit_meta.json
  project/evolve/reports/wx2_cases_catalog_audit.md
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

NS = ["Option", "List", "Bool", "Sum", "Prod", "Multiset"]

# name fragments that hint a constructor split closes the goal
CASES_HINTS = ("cons", "nil", "some", "none", "isSome", "isNone", "getD",
               "map", "bind", "head", "tail", "reverse", "append",
               "replicate", "fst", "snd", "inl", "inr", "elim", "cases",
               "rec", "swap", "isEmpty", "singleton", "mem")
INDUCTION_HINTS = ("foldr", "foldl", "fold", "length", "sum", "prod",
                   "count", "reverse", "append", "map")


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import tasks
    used: set[str] = set()
    for _s, thms in tasks.THEOREM_SETS.items():
        for t in thms:
            used.add(t.full_name)

    av = json.load(open("project/data/cx1_available_theorems.json"))["theorems"]
    by_name: dict[str, dict] = {}
    for t in av:
        by_name.setdefault(t["full_name"], t)  # dedup repeated entries

    per_ns: dict[str, dict] = {}
    fresh_pool: dict[str, list[dict]] = {}
    for ns in NS:
        cands = [t for n, t in by_name.items() if n.startswith(ns + ".")]
        fresh = [t for t in cands if t["full_name"] not in used]
        # classify fresh
        buckets: Counter = Counter()
        classified = []
        for t in fresh:
            short = t["full_name"].split(".")[-1].lower()
            tags = t.get("family_tags", [])
            fam = None
            if ns == "List":
                if any(h in short for h in ("foldr", "foldl", "fold")) or \
                        "fold" in tags:
                    fam = "list_induction"
                else:
                    fam = "list_cases"
            elif ns == "Option":
                fam = "option_cases"
            elif ns == "Bool":
                fam = "bool_cases"
            elif ns == "Prod":
                fam = "prod_cases"
            elif ns == "Sum":
                fam = "sum_cases"
            elif ns == "Multiset":
                fam = "multiset_quotient_excluded"
            buckets[fam] += 1
            classified.append({
                "full_name": t["full_name"], "file": t["file_path"],
                "difficulty": t.get("difficulty"), "tags": tags,
                "expected_family": fam,
            })
        per_ns[ns] = {
            "available": len(cands),
            "fresh_unused": len(fresh),
            "family_buckets": dict(buckets),
            "cases_friendly": ns not in ("Multiset",),
        }
        fresh_pool[ns] = classified

    meta = {
        "note": ("WX1 Option cases wrapper generalization audit. List is "
                 "the only large fresh cases-friendly surface; Option/Bool "
                 "are exhausted (consumed by CX3), Sum is absent, Prod is "
                 "tiny, Multiset is a quotient type (cases/induction on a "
                 "raw Multiset var does not apply) and is excluded."),
        "per_namespace": per_ns,
        "fresh_candidates": fresh_pool,
    }
    Path("project/data/wx2_cases_catalog_audit_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    md = ["# WX2 — cases/induction catalog audit", ""]
    md.append("Does the WX1 state-aware Option cases pattern generalize? "
              "Audit of fresh (unused) candidates in inductive namespaces.")
    md.append("")
    md.append("| namespace | available | fresh unused | cases-friendly | buckets |")
    md.append("|---|---:|---:|:---:|---|")
    for ns in NS:
        p = per_ns[ns]
        md.append(f"| {ns} | {p['available']} | {p['fresh_unused']} | "
                  f"{'yes' if p['cases_friendly'] else 'NO (quotient)'} | "
                  f"{p['family_buckets']} |")
    md.append("")
    md.append("**Verdict:** the fresh cases-friendly surface is dominated "
              "by **List** ({} fresh). Option and Bool are exhausted (0 "
              "fresh — consumed by CX3); Sum is absent; Prod is tiny; "
              "Multiset is a quotient type and excluded. WX2 generalization "
              "is therefore primarily a **List** test."
              .format(per_ns["List"]["fresh_unused"]))
    Path("project/evolve/reports/wx2_cases_catalog_audit.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8")

    print("wrote project/data/wx2_cases_catalog_audit_meta.json")
    print("wrote project/evolve/reports/wx2_cases_catalog_audit.md")
    for ns in NS:
        p = per_ns[ns]
        print(f"  {ns:9s} avail={p['available']:4d} fresh={p['fresh_unused']:4d} "
              f"buckets={p['family_buckets']}")


if __name__ == "__main__":
    main()
