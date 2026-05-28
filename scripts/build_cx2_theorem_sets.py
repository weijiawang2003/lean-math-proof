"""CX2 Stage 2 — build Int theorem sets for iff_omega mining.

Excludes:
  - the 43 Int theorems already probed in cx1_bool_option_int (those
    were measured against NS15-routed-raw + NS9 wrap and produced the
    2 known wrapper-only wins; re-probing the same surface under the
    same baselines yields no new signal)
  - the 3 known wrapper-only-vs-NS9 Int wins
    (Int.le_add_one_iff, Int.le_iff_lt_or_eq, Int.emod_two_eq_zero_or_one)

Builds:
  - cx2_int_iff_omega_easy   — fresh iff_omega candidates from the audit
  - cx2_int_iff_omega_medium — cast+iff candidates (norm_cast lead-in)
  - cx2_int_order_arith      — fresh omega-only le/lt/add/sub candidates
  - cx2_int_mixed            — fresh mixed Int arithmetic surface

Output:
  - project/evolve/routing/cx2_theorem_sets.json
  - project/data/cx2_theorem_sets_meta.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


CX1_PROBED_PATH = Path("project/evolve/routing/cx1_theorem_sets.json")
AUDIT_META_PATH = Path("project/data/cx2_int_catalog_audit_meta.json")
KNOWN_WINS = {
    "Int.le_add_one_iff",
    "Int.le_iff_lt_or_eq",
    "Int.emod_two_eq_zero_or_one",
}


def load_cx1_int_probed() -> set[str]:
    ts = json.load(open(CX1_PROBED_PATH))
    out: set[str] = set()
    # cx1_theorem_sets.json contains a top-level dict {set_name -> [theorems]}.
    for set_name, thms in ts.items():
        if not isinstance(thms, list):
            continue
        for t in thms:
            full = (t.get("full_name") if isinstance(t, dict) else t) or ""
            if full.startswith("Int."):
                out.add(full)
    return out


def main() -> None:
    audit = json.load(open(AUDIT_META_PATH))
    cx1_probed_int = load_cx1_int_probed()

    # Resolve full theorem candidates from the audit by full_name.
    # We need file_path/short_name from audit; the audit dumped enough.
    iff_omega_cands = audit["iff_omega_candidates"]
    omega_only_cands = audit["omega_only_candidates"]
    cast_cands = audit["cast_candidates_top20"]

    # Exclusion set.
    excluded = cx1_probed_int | KNOWN_WINS

    def fresh(cands: list[dict]) -> list[dict]:
        return [c for c in cands if c["full_name"] not in excluded]

    fresh_iff = fresh(iff_omega_cands)
    fresh_omega_only = fresh(omega_only_cands)
    fresh_cast = fresh(cast_cands)

    # The audit's iff_omega_candidates already exclude some non-iff
    # cast theorems. For "cx2_int_iff_omega_medium" we want CAST+IFF
    # theorems — find them by intersecting cast tags with iff_candidate
    # tags by re-reading from audit if needed. For simplicity: use
    # natCast_*_iff naming as the cast+iff overlap.
    cast_iff = [c for c in fresh_cast
                if "iff" in c["full_name"].lower()]

    def to_set_entry(c: dict, difficulty: str) -> dict:
        return {
            "file_path": c["file"],
            "full_name": c["full_name"],
            "difficulty": difficulty,
            "namespace": "Int",
            "family_tags": [t for t in c.get("tags", [])
                            if t != "other"],
        }

    cx2_iff_omega_easy = [to_set_entry(c, "easy") for c in fresh_iff]
    cx2_iff_omega_medium = [to_set_entry(c, "medium") for c in cast_iff]
    cx2_order_arith = [to_set_entry(c, "easy")
                       for c in fresh_omega_only[:50]]
    # Mixed: the rest of fresh_omega_only + any unused fresh_cast.
    used_names = {e["full_name"]
                  for e in (cx2_iff_omega_easy + cx2_iff_omega_medium
                            + cx2_order_arith)}
    cx2_mixed_pool = [
        to_set_entry(c, "medium")
        for c in (fresh_omega_only + fresh_cast)
        if c["full_name"] not in used_names
    ][:40]

    sets = {
        "cx2_int_iff_omega_easy": cx2_iff_omega_easy,
        "cx2_int_iff_omega_medium": cx2_iff_omega_medium,
        "cx2_int_order_arith": cx2_order_arith,
        "cx2_int_mixed": cx2_mixed_pool,
    }

    Path("project/evolve/routing/cx2_theorem_sets.json").write_text(
        json.dumps(sets, indent=2), encoding="utf-8"
    )
    by_file: dict = defaultdict(int)
    for s_name, thms in sets.items():
        for t in thms:
            by_file[(s_name, t["file_path"])] += 1
    meta = {
        "cx1_int_probed_excluded": sorted(cx1_probed_int),
        "n_cx1_probed_excluded": len(cx1_probed_int),
        "known_wins_excluded": sorted(KNOWN_WINS),
        "set_sizes": {k: len(v) for k, v in sets.items()},
        "total_size": sum(len(v) for v in sets.values()),
        "per_set_by_file": {
            f"{s}|{f}": n for (s, f), n in by_file.items()
        },
    }
    Path("project/data/cx2_theorem_sets_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print(f"wrote project/evolve/routing/cx2_theorem_sets.json")
    print(f"wrote project/data/cx2_theorem_sets_meta.json")
    print(f"\nset sizes:")
    for k, v in sets.items():
        print(f"  {k}: {len(v)}")
    print(f"\ntotal: {sum(len(v) for v in sets.values())} theorems")
    print(f"cx1-already-probed Int excluded: {len(cx1_probed_int)}")


if __name__ == "__main__":
    main()
