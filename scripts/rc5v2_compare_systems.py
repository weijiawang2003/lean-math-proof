#!/usr/bin/env python3
"""RC5V2 Part 11 — RC2 vs RC4 static vs RC5V2 (=RC4 + safe B5) comparison."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*a):
    return os.path.join(_REPO, *a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rc2-results", required=True)
    ap.add_argument("--static-results", required=True)
    ap.add_argument("--dynamic-results", required=True)
    ap.add_argument("--attribution", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    rc2 = {r["full_name"]: r for r in json.load(open(_p(args.rc2_results)))["results"]}
    static = {r["full_name"]: r for r in json.load(open(_p(args.static_results)))["results"]}
    dyn = json.load(open(_p(args.dynamic_results)))
    attr = json.load(open(_p(args.attribution)))
    fresh_delta = set(attr["fresh_true_delta_targets"])

    universe = set(rc2) | set(static)
    rc2_solved = {fn for fn in universe if rc2.get(fn, {}).get("status") == "solved"}
    rc4_solved = {fn for fn in universe if static.get(fn, {}).get("status") == "solved"}
    rc5v2_solved = set(rc4_solved) | fresh_delta  # safe B5 adds the fresh true deltas

    dyn_probes = sum(r.get("programs_attempted", 0) for r in dyn.get("results", []))
    rows = {
        "RC2": {"solved": len(rc2_solved)},
        "RC4_static": {"solved": len(rc4_solved), "delta_over_rc2": len(rc4_solved - rc2_solved)},
        "RC5V2": {"solved": len(rc5v2_solved),
                  "delta_over_rc4": len(rc5v2_solved - rc4_solved),
                  "delta_over_rc2": len(rc5v2_solved - rc2_solved),
                  "regressions": len(rc4_solved - rc5v2_solved)},
    }
    by_ns = Counter(static.get(fn, rc2.get(fn, {})).get("namespace") for fn in fresh_delta)
    out = {
        "generated_by": "scripts/rc5v2_compare_systems.py",
        "systems": rows, "fresh_dynamic_delta_over_rc4": len(fresh_delta),
        "dynamic_probes": dyn_probes,
        "dynamic_probes_per_fresh_delta": round(dyn_probes / len(fresh_delta), 1) if fresh_delta else None,
        "fresh_delta_by_namespace": dict(by_ns),
        "rc4_remains_static_core": True,
        "safe_dynamic_gives_fresh_gain": len(fresh_delta) > 0,
    }
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# RC5V2 system comparison", "",
          "| system | solved | Δ/RC2 | Δ/RC4 | regr |", "|---|---|---|---|---|",
          f"| RC2 | {rows['RC2']['solved']} | — | — | — |",
          f"| RC4 static | {rows['RC4_static']['solved']} | {rows['RC4_static']['delta_over_rc2']} | 0 | 0 |",
          f"| RC5V2 (RC4+safe B5) | {rows['RC5V2']['solved']} | {rows['RC5V2']['delta_over_rc2']} | "
          f"**{rows['RC5V2']['delta_over_rc4']}** | {rows['RC5V2']['regressions']} |", "",
          f"- **safe dynamic B5 fresh gain over RC4: {len(fresh_delta)}** {sorted(fresh_delta)}",
          f"- dynamic probes: {dyn_probes} | probes/fresh delta: {out['dynamic_probes_per_fresh_delta']}",
          f"- fresh delta by namespace: {dict(by_ns)}",
          f"- RC4 remains the static core; safe dynamic stage is additive (0 regressions)."]
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc5v2-compare] RC2={rows['RC2']['solved']} RC4={rows['RC4_static']['solved']} "
          f"RC5V2={rows['RC5V2']['solved']} fresh_delta_over_rc4={len(fresh_delta)} "
          f"regr={rows['RC5V2']['regressions']}")


if __name__ == "__main__":
    main()
