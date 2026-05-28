"""NS14 Stage 1 — wider theorem-set construction.

Reads ``project/discovered_theorems.json`` (which was produced
offline from a Mathlib scan), filters out theorems already
appearing in the prior eval sets (``nat_defs_medium``,
``nat_defs_large_v5``, ``demo_v1``, ``set_small``, ``finset_small``,
``mixed_easy_v2``), and emits four new sets:

  - ``ns14_nat_extra``         — 20 easy Nat.Defs lemmas
  - ``ns14_set_finset_extra``  — 10 Set + 10 Finset easy lemmas
  - ``ns14_mixed_easy``        — 5 from each namespace
  - ``ns14_mixed_medium``      — medium-difficulty mix

Output: ``project/evolve/routing/ns14_theorem_sets.json``.
``tasks.py`` reads this file at import time and registers the
four sets.

Selection is deterministic: discovered-theorems insertion order
within each (namespace, difficulty) bucket; first N entries kept.

Usage::

    python scripts/build_ns14_theorem_sets.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Re-use the existing tasks module to find theorems we already test.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tasks import THEOREM_SETS  # noqa: E402


DISCOVERED_PATH = Path("project/discovered_theorems.json")
OUT_PATH = Path("project/evolve/routing/ns14_theorem_sets.json")

EXISTING_SETS = (
    "nat_defs_medium",
    "nat_defs_large_v5",
    "demo_v1",
    "nat_defs_subset",
    "nat_more",
    "set_small",
    "finset_small",
    "mixed_easy_v2",
)


def main() -> None:
    if not DISCOVERED_PATH.exists():
        raise FileNotFoundError(DISCOVERED_PATH)
    catalog = json.loads(DISCOVERED_PATH.read_text(encoding="utf-8"))

    used: set[str] = set()
    for s in EXISTING_SETS:
        for cfg in THEOREM_SETS.get(s, []):
            used.add(cfg.full_name)

    by_ns_diff: dict[tuple[str, str], list[dict]] = {}
    for t in catalog["theorems"]:
        if t["full_name"] in used:
            continue
        if not t.get("has_tactic_proof"):
            continue
        name = t["full_name"]
        if "." not in name:
            continue
        ns = name.split(".", 1)[0]
        if ns not in ("Nat", "Set", "Finset"):
            continue
        diff = t.get("difficulty") or "?"
        by_ns_diff.setdefault((ns, diff), []).append(
            {"file_path": t["file_path"], "full_name": name,
             "difficulty": diff}
        )

    def take(ns: str, diff: str, k: int) -> list[dict]:
        return list(by_ns_diff.get((ns, diff), []))[:k]

    nat_easy = take("Nat", "easy", 20)
    set_easy = take("Set", "easy", 10)
    finset_easy = take("Finset", "easy", 10)
    nat_med = take("Nat", "medium", 5)
    set_med = take("Set", "medium", 5)
    finset_med = take("Finset", "medium", 5)

    out: dict[str, list[dict]] = {
        "ns14_nat_extra": nat_easy,
        "ns14_set_finset_extra": set_easy + finset_easy,
        "ns14_mixed_easy": (
            nat_easy[:5] + set_easy[:5] + finset_easy[:5]
        ),
        "ns14_mixed_medium": nat_med + set_med + finset_med,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"wrote {OUT_PATH}")
    for name, items in out.items():
        ns_counts: dict[str, int] = {}
        for t in items:
            ns = t["full_name"].split(".", 1)[0]
            ns_counts[ns] = ns_counts.get(ns, 0) + 1
        ns_str = ", ".join(f"{k}={v}" for k, v in sorted(ns_counts.items()))
        print(f"  {name}: {len(items)} theorems ({ns_str})")


if __name__ == "__main__":
    main()
