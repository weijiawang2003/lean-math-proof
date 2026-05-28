"""NS20 Stage 1 — final Finset/aesop mining surface.

Mines every Finset theorem in project/discovered_theorems.json that
is NOT already covered by:
  - demo_v1
  - ns14_set_finset_extra
  - ns17_finset_extra
  - ns19_finset_aesop_surface
  - finset_small, mixed_easy_v2, ns14_mixed_easy, ns14_mixed_medium,
    ns17_list_multiset (occasional Finset thms)

Unlike NS19, NS20 does NOT filter by aesop-friendly token list — the
remainder is small (~74 thms) so spending the eval budget on every
remaining theorem is cheaper than rejecting candidates the token
filter might miss. Three sets are emitted by difficulty:

  ns20_finset_aesop_extra_easy
  ns20_finset_aesop_extra_medium
  ns20_finset_aesop_extra_hard

Output: project/evolve/routing/ns20_theorem_sets.json.
tasks.py is patched to load these at import via _load_ns20_sets().
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tasks import THEOREM_SETS  # noqa: E402


DISCOVERED_PATH = Path("project/discovered_theorems.json")
OUT_PATH = Path("project/evolve/routing/ns20_theorem_sets.json")

EXCLUDE_SETS = (
    "demo_v1",
    "ns14_set_finset_extra",
    "ns17_finset_extra",
    "ns19_finset_aesop_surface",
    "finset_small",
    "mixed_easy_v2",
    "ns14_mixed_easy",
    "ns14_mixed_medium",
    "ns17_list_multiset",
)


def main() -> None:
    catalog = json.loads(DISCOVERED_PATH.read_text(encoding="utf-8"))
    used: set[str] = set()
    for s in EXCLUDE_SETS:
        for cfg in THEOREM_SETS.get(s, []):
            used.add(cfg.full_name)

    pools: dict[str, list[dict]] = {
        "ns20_finset_aesop_extra_easy": [],
        "ns20_finset_aesop_extra_medium": [],
        "ns20_finset_aesop_extra_hard": [],
    }
    for t in catalog["theorems"]:
        name = t.get("full_name", "")
        if not name.startswith("Finset.") or name in used:
            continue
        if not t.get("has_tactic_proof"):
            continue
        d = t.get("difficulty", "?")
        bucket = f"ns20_finset_aesop_extra_{d}"
        if bucket not in pools:
            continue
        pools[bucket].append({
            "file_path": t["file_path"],
            "full_name": name,
            "difficulty": d,
        })

    # Sort within each bucket by full_name for stable evals.
    for k in pools:
        pools[k].sort(key=lambda t: t["full_name"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(pools, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    for k, v in pools.items():
        print(f"  {k}: {len(v)}")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
