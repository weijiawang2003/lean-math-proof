"""NS17 Stage 4 — targeted theorem-surface construction.

The NS17 audit showed that the per-family pool sizes for the
likely-NS18 candidates (``iff_omega_pair``, ``constructor_omega``,
``nat_simp_arith``, ``set_subset_simp``) are small.

This script enumerates fresh theorem surface from
``project/discovered_theorems.json``, excluding every set already
used in NS14/NS16/etc., and emits four NS17 surfaces:

  - ``ns17_nat_remaining``  — all unused Nat theorems
                              (a small, narrow remainder).
  - ``ns17_set_extra``      — up to 30 fresh Set theorems.
  - ``ns17_finset_extra``   — up to 30 fresh Finset theorems.
  - ``ns17_list_multiset``  — fresh List + Multiset theorems.

The List/Multiset surface is unexplored — the model has never seen
any wrapper trace on these namespaces. Running wrapper on them
should reveal which (if any) new pattern families produce
wrapper-only wins.

Output: ``project/evolve/routing/ns17_theorem_sets.json``.
``tasks.py`` is patched to load these at import time.

Usage:
    python scripts/build_ns17_theorem_sets.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tasks import THEOREM_SETS  # noqa: E402


DISCOVERED_PATH = Path("project/discovered_theorems.json")
OUT_PATH = Path("project/evolve/routing/ns17_theorem_sets.json")

EXISTING_SETS = (
    "nat_defs_medium", "nat_defs_large_v5", "demo_v1",
    "nat_defs_subset", "nat_more",
    "set_small", "finset_small", "mixed_easy_v2",
    "ns14_nat_extra", "ns14_set_finset_extra",
    "ns14_mixed_easy", "ns14_mixed_medium",
    "ns16_nat_iff_extra", "ns16_nat_div_mod_extra",
    "ns16_nat_order_extra", "ns16_nat_mixed_extra",
)


def main() -> None:
    catalog = json.loads(DISCOVERED_PATH.read_text(encoding="utf-8"))
    used: set[str] = set()
    for s in EXISTING_SETS:
        for cfg in THEOREM_SETS.get(s, []):
            used.add(cfg.full_name)

    by_ns: dict[str, list[dict]] = {}
    for t in catalog["theorems"]:
        name = t.get("full_name", "")
        if name in used or not t.get("has_tactic_proof"):
            continue
        if "." not in name:
            continue
        ns = name.split(".", 1)[0]
        # Exclude hard for Set/Finset/List/Multiset (wrapper won't reach),
        # but keep hard Nat — that's the only remaining Nat surface.
        if t.get("difficulty") == "hard" and ns != "Nat":
            continue
        by_ns.setdefault(ns, []).append({
            "file_path": t["file_path"],
            "full_name": name,
            "difficulty": t.get("difficulty", "?"),
        })

    nat = by_ns.get("Nat", [])[:40]
    set_extras = by_ns.get("Set", [])[:30]
    finset_extras = by_ns.get("Finset", [])[:30]
    list_extras = by_ns.get("List", [])
    multiset_extras = by_ns.get("Multiset", [])

    out: dict[str, list[dict]] = {
        "ns17_nat_remaining": nat,
        "ns17_set_extra": set_extras,
        "ns17_finset_extra": finset_extras,
        "ns17_list_multiset": list_extras + multiset_extras,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")

    total = 0
    for name, items in out.items():
        difficulties: dict[str, int] = {}
        nss: dict[str, int] = {}
        for t in items:
            d = t.get("difficulty", "?")
            difficulties[d] = difficulties.get(d, 0) + 1
            ns = t["full_name"].split(".", 1)[0]
            nss[ns] = nss.get(ns, 0) + 1
        diff_s = ", ".join(f"{k}={v}" for k, v in sorted(difficulties.items()))
        ns_s = ", ".join(f"{k}={v}" for k, v in sorted(nss.items()))
        print(f"  {name}: {len(items)} ({ns_s} | {diff_s})")
        total += len(items)
    print(f"  TOTAL: {total}")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
