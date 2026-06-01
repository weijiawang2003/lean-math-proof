#!/usr/bin/env python3
"""RC4R Part 8 — full canonical floor verification.

Reads the RC2 and RC4 benchmark results and verifies, per canonical floor, that RC4 >= RC2 and
RC4 >= the RC2 release floor reference. Hard fail if any floor regresses (an RC2-solved theorem
RC4 fails, or RC4 count < RC2 count).
"""
from __future__ import annotations

import argparse
import json
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLOORS = ("canonical_demo_v1", "canonical_nat_defs_medium", "canonical_nat_defs_large_v5")
# RC2 release floor references (from the RC2 release freeze)
RELEASE_FLOOR = {"canonical_demo_v1": 12, "canonical_nat_defs_medium": 37,
                 "canonical_nat_defs_large_v5": 49}


def _p(*a):
    return os.path.join(_REPO, *a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rc2", required=True)
    ap.add_argument("--rc4", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    rc2 = {r["full_name"]: r for r in json.load(open(_p(args.rc2)))["results"]}
    rc4 = {r["full_name"]: r for r in json.load(open(_p(args.rc4)))["results"]}

    rows, all_pass = [], True
    for floor in FLOORS:
        names = [fn for fn, r in rc2.items() if floor in (r.get("sets") or [])]
        rc2_solved = sorted(fn for fn in names if rc2[fn]["status"] == "solved")
        rc4_solved = sorted(fn for fn in names if rc4.get(fn, {}).get("status") == "solved")
        regressed = sorted(set(rc2_solved) - set(rc4_solved))
        gained = sorted(set(rc4_solved) - set(rc2_solved))
        ref = RELEASE_FLOOR.get(floor)
        floor_pass = (len(rc4_solved) >= len(rc2_solved) and not regressed
                      and (ref is None or len(rc4_solved) >= ref))
        all_pass = all_pass and floor_pass
        rows.append({"floor": floor, "n": len(names), "rc2_solved": len(rc2_solved),
                     "rc4_solved": len(rc4_solved), "release_floor_ref": ref,
                     "delta": len(rc4_solved) - len(rc2_solved),
                     "regressed_theorems": regressed, "gained_theorems": gained,
                     "floor_pass": floor_pass})

    out = {"generated_by": "scripts/rc4r_full_floor_verification.py",
           "floors": rows, "all_floors_pass": all_pass,
           "total_regressions": sum(len(r["regressed_theorems"]) for r in rows),
           "hard_fail": not all_pass}
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# RC4 full canonical floor verification", "",
          f"- **all floors pass (RC4 ≥ RC2 ≥ release floor, no regression): {all_pass}**",
          f"- total regressions: {out['total_regressions']}", "",
          "| floor | n | RC2 | RC4 | release ref | delta | regressed | pass |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['floor']} | {r['n']} | {r['rc2_solved']} | {r['rc4_solved']} | "
                  f"{r['release_floor_ref']} | {r['delta']} | {len(r['regressed_theorems'])} | "
                  f"{r['floor_pass']} |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc4r-floor] all_pass={all_pass} regressions={out['total_regressions']} "
          f"rows={[(r['floor'], r['rc2_solved'], r['rc4_solved']) for r in rows]}")


if __name__ == "__main__":
    main()
