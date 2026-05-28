"""NS23 Stage 1 — collect all wrapper-only-vs-NS9 wins across arcs.

Reads:
  - project/data/cx1_combined_pool_meta.json — NS18 + NS19 + NS20 + CX1
  - project/data/cx2_int_iff_omega_pool_meta.json — CX1 + CX2 Int

Resolves file_path from tasks.THEOREM_SETS (the source of truth for
mining configs).

Output:
  project/data/ns23_wrapper_only_wins_raw_meta.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import tasks
    name_to_file: dict[str, str] = {}
    for set_name, thms in tasks.THEOREM_SETS.items():
        for t in thms:
            name_to_file.setdefault(t.full_name, t.file_path)

    out: dict = {"theorems": []}
    seen: set[str] = set()

    def add(thm: str, fam: str, ns: str, arc: str, set_name: str,
            tactic: str) -> None:
        if thm in seen:
            return
        seen.add(thm)
        fp = name_to_file.get(thm)
        if not fp:
            print(f"WARN: no file_path for {thm}")
            return
        out["theorems"].append({
            "full_name": thm,
            "file_path": fp,
            "namespace": ns,
            "original_family": fam,
            "wrapper_tactic": tactic,
            "first_seen_arc": arc,
            "first_seen_set": set_name,
        })

    # CX1 combined pool (NS18+NS19+NS20+CX1)
    c1 = json.load(open("project/data/cx1_combined_pool_meta.json"))
    for key, info in c1["families"].items():
        fam, ns = key.split("|", 1)
        for thm, det in info["theorems"].items():
            add(thm, fam, ns, det.get("first_seen_in_arc", "?"),
                det.get("first_seen_in_set", "?"), det.get("winning_tactic", ""))

    # CX2 Int pool (CX1 + CX2 Int)
    c2 = json.load(open("project/data/cx2_int_iff_omega_pool_meta.json"))
    for key, info in c2["families"].items():
        fam, ns = key.split("|", 1)
        for thm, det in info["theorems"].items():
            add(thm, fam, ns, det.get("first_seen_arc", "?"),
                det.get("first_seen_set", "?"), det.get("winning_tactic", ""))

    # Summarize.
    by_family = {}
    by_namespace = {}
    by_arc = {}
    for t in out["theorems"]:
        by_family[t["original_family"]] = by_family.get(
            t["original_family"], 0) + 1
        by_namespace[t["namespace"]] = by_namespace.get(
            t["namespace"], 0) + 1
        by_arc[t["first_seen_arc"]] = by_arc.get(t["first_seen_arc"], 0) + 1
    out["summary"] = {
        "total_unique_wrapper_only_theorems": len(out["theorems"]),
        "by_original_family": by_family,
        "by_namespace": by_namespace,
        "by_arc": by_arc,
    }

    Path("project/data/ns23_wrapper_only_wins_raw_meta.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(f"wrote project/data/ns23_wrapper_only_wins_raw_meta.json")
    print(f"total theorems: {out['summary']['total_unique_wrapper_only_theorems']}")
    print(f"by family: {by_family}")
    print(f"by namespace: {by_namespace}")
    print(f"by arc: {by_arc}")


if __name__ == "__main__":
    main()
