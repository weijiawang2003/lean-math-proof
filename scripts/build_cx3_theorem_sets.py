"""CX3 Stage 3 — build Bool/Option theorem sets from the catalog audit.

Reads project/data/cx3_bool_option_catalog_audit_meta.json (Stage 2) and
deterministically partitions the *fresh* (unused) Bool/Option candidates
into five disjoint theorem sets, written as
project/evolve/routing/cx3_theorem_sets.json (loaded by tasks._load_cx3_sets).

Partition (disjoint, in priority order):
  1. cx3_bool_decide_easy   — Bool props from Bool/Basic (decide targets)
  2. cx3_bool_simp_medium   — Bool/Set facts (simp/ext targets)
  3. cx3_option_cases_medium— Option map/bind/pbind/pmap needing case split
  4. cx3_option_simp_easy   — verified-available Option simp surface
  5. cx3_bool_option_mixed  — remaining fresh Option (incl. needs-probe)

NOTE: the fresh Bool surface is tiny (Bool/Basic was exhausted by
cx1_bool_option_int in CX1); the bulk of CX3's fresh surface is Option.
"""
from __future__ import annotations

import json
from pathlib import Path

AUDIT = "project/data/cx3_bool_option_catalog_audit_meta.json"
OUT = "project/evolve/routing/cx3_theorem_sets.json"

SIMP_KW = ("isSome", "isNone", "getD", "orElse", "none", "some", "elim",
           "mem", "toList", "guard", "iget", "coe", "get_map", "map_comm",
           "pmap_eq", "isSome_map", "bnot")
CASES_KW = ("bind_congr", "bind_eq_bind", "none_bind", "some_bind")


def main() -> None:
    m = json.load(open(AUDIT))
    ver = [{**c, "availability": "verified"}
           for c in m["fresh_verified_candidates"]]
    needs = [{**c} for c in m["fresh_needs_probe_candidates"]]
    allc = ver + needs

    used: set[str] = set()
    buckets: dict[str, list[dict]] = {
        "cx3_bool_decide_easy": [],
        "cx3_bool_simp_medium": [],
        "cx3_option_cases_medium": [],
        "cx3_option_simp_easy": [],
        "cx3_bool_option_mixed": [],
    }

    def take(c: dict, key: str) -> None:
        if c["full_name"] in used:
            return
        used.add(c["full_name"])
        buckets[key].append(c)

    # 1. Bool decide (Bool/Basic props)
    for c in allc:
        if c["ns"] == "Bool" and "Basic" in c["file"]:
            take(c, "cx3_bool_decide_easy")
    # 2. Bool simp (Bool/Set)
    for c in allc:
        if c["ns"] == "Bool" and "Set" in c["file"]:
            take(c, "cx3_bool_simp_medium")
    # 3. Option cases (map/bind/pbind/pmap that need a none/some split)
    for c in allc:
        if c["full_name"] in used:
            continue
        if c["ns"] == "Option" and c["bucket"] == "likely_cases_simp":
            take(c, "cx3_option_cases_medium")
    for c in allc:
        if c["full_name"] in used:
            continue
        if c["ns"] == "Option" and any(k in c["full_name"] for k in CASES_KW):
            take(c, "cx3_option_cases_medium")
    # 4. Option simp easy (verified-available simp surface)
    for c in allc:
        if c["full_name"] in used:
            continue
        if (c["ns"] == "Option" and c["availability"] == "verified"
                and any(k in c["full_name"] for k in SIMP_KW)):
            take(c, "cx3_option_simp_easy")
    # 5. Mixed (remaining fresh Option, incl. needs-probe)
    for c in allc:
        if c["full_name"] in used:
            continue
        take(c, "cx3_bool_option_mixed")

    diff_map = {c["full_name"]: c.get("difficulty", "?") for c in ver}
    out: dict[str, list[dict]] = {}
    for key, items in buckets.items():
        out[key] = [
            {
                "file_path": c["file"],
                "full_name": c["full_name"],
                "namespace": c["ns"],
                "difficulty": diff_map.get(c["full_name"], "unknown"),
                "family_tags": c["tags"],
                "availability": c["availability"],
                "expected_bucket": c["bucket"],
            }
            for c in sorted(items, key=lambda x: x["full_name"])
        ]

    Path(OUT).write_text(json.dumps(out, indent=2), encoding="utf-8")
    total = sum(len(v) for v in out.values())
    print(f"wrote {OUT}")
    for k, v in out.items():
        print(f"  {k}: {len(v)}")
    print(f"  TOTAL: {total} (fresh universe: {len(allc)})")


if __name__ == "__main__":
    main()
