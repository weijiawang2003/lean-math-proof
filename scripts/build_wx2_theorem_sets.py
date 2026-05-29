"""WX2 Stage 4 — build fresh cases-test theorem sets from the audit.

Reads project/data/wx2_cases_catalog_audit_meta.json and writes five
disjoint sets to project/evolve/routing/wx2_theorem_sets.json (loaded by
tasks._load_wx2_sets). The fresh cases-friendly surface is essentially
List (Option/Bool exhausted, Sum absent, Prod tiny, Multiset excluded),
so the List sets carry the generalization test; Bool reuses the CX3
fresh-Bool theorems as a negative control.

Sets:
  wx2_list_cases_easy    — medium-difficulty List structural lemmas
  wx2_list_cases_medium  — remaining medium + a sample of hard List
  wx2_list_induction     — List fold/length/sum lemmas (induction-leaning)
  wx2_prod_cases         — the fresh Prod candidate(s)
  wx2_bool_cases_control — CX3 fresh Bool (negative control; Bool exhausted)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

AUDIT = "project/data/wx2_cases_catalog_audit_meta.json"
OUT = "project/evolve/routing/wx2_theorem_sets.json"

STRUCTURAL = ("cons", "nil", "append", "map", "replicate", "reverse",
              "head", "tail", "mem", "singleton", "isEmpty", "concat",
              "getLast", "filter", "drop", "take")
EASY_CAP = 40
MEDIUM_CAP = 35


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import tasks
    m = json.load(open(AUDIT))
    fresh = m["fresh_candidates"]

    list_cases = [c for c in fresh["List"]
                  if c["expected_family"] == "list_cases"]
    list_ind = [c for c in fresh["List"]
                if c["expected_family"] == "list_induction"]

    def structural(c):
        s = c["full_name"].split(".")[-1].lower()
        return any(h in s for h in STRUCTURAL)

    easy = [c for c in list_cases
            if c["difficulty"] == "medium" and structural(c)]
    easy = sorted(easy, key=lambda c: c["full_name"])[:EASY_CAP]
    easy_names = {c["full_name"] for c in easy}

    rest = [c for c in list_cases if c["full_name"] not in easy_names]
    medium = sorted(rest, key=lambda c: (c["difficulty"] != "medium",
                                         c["full_name"]))[:MEDIUM_CAP]

    prod = fresh.get("Prod", [])

    # Bool control: reuse CX3 fresh-Bool theorems (Bool is exhausted, so
    # there is no fresh Bool; the control confirms the generalized config's
    # Bool tactics neither break nor newly help).
    bool_ctrl = []
    for t in tasks.THEOREM_SETS.get("cx3_bool_decide_easy", []):
        bool_ctrl.append({"full_name": t.full_name, "file": t.file_path,
                          "difficulty": "?", "expected_family": "bool_cases"})

    def emit(items):
        out = []
        seen = set()
        for c in items:
            if c["full_name"] in seen:
                continue
            seen.add(c["full_name"])
            out.append({
                "file_path": c["file"], "full_name": c["full_name"],
                "namespace": c["full_name"].split(".")[0],
                "difficulty": c.get("difficulty", "?"),
                "expected_family": c["expected_family"],
            })
        return out

    sets = {
        "wx2_list_cases_easy": emit(easy),
        "wx2_list_cases_medium": emit(medium),
        "wx2_list_induction": emit(list_ind),
        "wx2_prod_cases": emit(prod),
        "wx2_bool_cases_control": emit(bool_ctrl),
    }
    Path(OUT).write_text(json.dumps(sets, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    total = sum(len(v) for v in sets.values())
    print(f"wrote {OUT}")
    for k, v in sets.items():
        print(f"  {k}: {len(v)}")
    print(f"  TOTAL: {total}")


if __name__ == "__main__":
    main()
