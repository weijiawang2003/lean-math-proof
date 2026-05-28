"""NS16 Stage 1 — expanded Nat theorem-set construction.

Builds four new Nat-focused theorem sets from
``project/discovered_theorems.json``, excluding theorems already
present in any prior eval set.  Selection is naming-heuristic based:

  - ``ns16_nat_iff_extra``        — names containing ``_iff`` /
    ``_iff_``: wrapper's ``exact ⟨fun h => by omega,
    fun h => by omega⟩`` template should fire often.
  - ``ns16_nat_div_mod_extra``    — names mentioning ``div``,
    ``mod``, or ``dvd``: NS9 family tactics for divisibility +
    `omega` after substitution.
  - ``ns16_nat_order_extra``      — `_lt_`, `_le_`, `lt_`, `le_`
    in name: omega / linarith / norm_num close many.
  - ``ns16_nat_mixed_extra``      — everything else (rounded up).

Buckets are deterministic (discovered-theorems insertion order).

Output: ``project/evolve/routing/ns16_theorem_sets.json``.
``tasks.py`` reads this at import time and registers each ``ns16_*``
set.

Usage:
    python scripts/build_ns16_theorem_sets.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Re-use tasks.py to deduplicate against prior eval sets.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tasks import THEOREM_SETS  # noqa: E402


DISCOVERED_PATH = Path("project/discovered_theorems.json")
OUT_PATH = Path("project/evolve/routing/ns16_theorem_sets.json")

EXISTING_SETS = (
    "nat_defs_medium",
    "nat_defs_large_v5",
    "demo_v1",
    "nat_defs_subset",
    "nat_more",
    "set_small",
    "finset_small",
    "mixed_easy_v2",
    "ns14_nat_extra",
    "ns14_set_finset_extra",
    "ns14_mixed_easy",
    "ns14_mixed_medium",
)

# Heuristic naming buckets. Order matters: a name is assigned to the
# first matching bucket. We greedily attach iff first so that
# ``div_lt_iff`` lands in iff (the shape that drives the template).
IFF_TOKENS = ("_iff_", "_iff",)
DIVMOD_TOKENS = ("div", "mod", "dvd", "gcd", "lcm")
ORDER_TOKENS = ("_lt_", "_le_", "lt_", "le_", "_pos", "pos_iff",
                "succ_lt", "pred_lt")


def classify(name: str) -> str:
    short = name.split(".", 1)[1] if "." in name else name
    sl = short.lower()
    for tok in IFF_TOKENS:
        if tok in sl:
            return "iff"
    for tok in DIVMOD_TOKENS:
        if tok in sl:
            return "div_mod"
    for tok in ORDER_TOKENS:
        if tok in sl:
            return "order"
    return "mixed"


def main() -> None:
    if not DISCOVERED_PATH.exists():
        raise FileNotFoundError(DISCOVERED_PATH)
    catalog = json.loads(DISCOVERED_PATH.read_text(encoding="utf-8"))

    used: set[str] = set()
    for s in EXISTING_SETS:
        for cfg in THEOREM_SETS.get(s, []):
            used.add(cfg.full_name)

    buckets: dict[str, list[dict]] = {
        "iff": [], "div_mod": [], "order": [], "mixed": [],
    }
    for t in catalog["theorems"]:
        name = t.get("full_name", "")
        if not name.startswith("Nat."):
            continue
        if name in used:
            continue
        if not t.get("has_tactic_proof"):
            continue
        # Skip hard for now — wrapper templates target easy/medium shapes.
        if t.get("difficulty") == "hard":
            continue
        bucket = classify(name)
        buckets[bucket].append({
            "file_path": t["file_path"],
            "full_name": name,
            "difficulty": t.get("difficulty", "?"),
        })

    # Caps per bucket. iff is the priority bucket so we take more.
    def take(bucket: str, k: int) -> list[dict]:
        return list(buckets[bucket])[:k]

    out: dict[str, list[dict]] = {
        "ns16_nat_iff_extra":     take("iff", 40),
        "ns16_nat_div_mod_extra": take("div_mod", 40),
        "ns16_nat_order_extra":   take("order", 40),
        "ns16_nat_mixed_extra":   take("mixed", 30),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"wrote {OUT_PATH}")
    total = 0
    for name, items in out.items():
        difficulties: dict[str, int] = {}
        for t in items:
            d = t.get("difficulty", "?")
            difficulties[d] = difficulties.get(d, 0) + 1
        d_str = ", ".join(f"{k}={v}" for k, v in sorted(difficulties.items()))
        print(f"  {name}: {len(items)} theorems ({d_str})")
        total += len(items)
    print(f"  TOTAL: {total} theorems")


if __name__ == "__main__":
    main()
