"""MX2 Stage 3 — build Set-aesop theorem sets from the MX1 frontier + audit.

Five disjoint sets to test whether a Set-gated `aesop` fallback captures the
MX1 misses and similar Set lemmas without overfiring or regressing:

  mx2_set_aesop_known        the 2 known MX1 aesop-misses.
  mx2_set_finite_frontier    Set.Finite./Set.toFinset candidates (aesop-amenable).
  mx2_set_aesop_frontier     broader Set (image/preimage/insert) candidates.
  mx2_set_negative_control   relation/function-shaped Set lemmas (EqOn/InjOn/
                             MapsTo/Infinite) where aesop should NOT help —
                             measures overfiring cost / false fires.
  mx2_mixed_preservation_control  small cross-namespace (Set/Finset/List/Multiset)
                             control to confirm Set-gating leaves others alone.

No Lean here; availability confirmed at eval time. Registered via
tasks._load_mx2_sets(). The known REPL-hanger is excluded.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAND = ROOT / "project/data/mx2_set_aesop_candidate_meta.json"
AUDIT = ROOT / "project/data/mx1_symbolic_frontier_audit_meta.json"
OUT = ROOT / "project/evolve/routing/mx2_theorem_sets.json"

KNOWN_BAD = {"Multiset.eq_of_mem_map_const"}
# relation/function-shaped Set prefixes — aesop should NOT close these (control)
NEG_PREFIXES = ("Set.EqOn.", "Set.InjOn.", "Set.MapsTo.", "Set.Infinite.",
                "Set.LeftInvOn.", "Set.SurjOn.", "Set.BijOn.")
CAP_FINITE = 10
CAP_FRONTIER = 10
CAP_NEG = 10
CAP_MIXED = 8


def main() -> None:
    cm = json.loads(CAND.read_text())
    audit = {c["full_name"]: c
             for c in json.loads(AUDIT.read_text())["candidates"]}

    def fp(fn):
        return audit.get(fn, {}).get("file_path", "")

    def ent(fn):
        return {"file_path": fp(fn), "full_name": fn}

    used = set()

    def take(cands, cap):
        out = []
        for fn in cands:
            if fn in used or fn in KNOWN_BAD or not fp(fn):
                continue
            used.add(fn)
            out.append(ent(fn))
            if len(out) >= cap:
                break
        return out

    known = [k for k in cm["known_aesop_wins"]]
    for k in known:
        used.add(k)
    known_set = [ent(k) for k in known if fp(k)]

    # Set.Finite. + Set.toFinset bucket (aesop-amenable)
    finite_pool = ([x["full_name"] for x in cm["buckets"]
                    .get("Set.Finite.toFinset", {}).get("theorems", [])]
                   + [x["full_name"] for x in cm["buckets"]
                      .get("Set.Finite.", {}).get("theorems", [])])
    set_finite = take(finite_pool, CAP_FINITE)

    # broader Set (image / preimage / insert)
    broad_pool = []
    for b in ("Set.image", "Set.preimage", "Set.insert", "Set.offDiag"):
        broad_pool += [x["full_name"]
                       for x in cm["buckets"].get(b, {}).get("theorems", [])]
    set_frontier = take(broad_pool, CAP_FRONTIER)

    # negative control: relation/function-shaped Set lemmas
    neg_pool = sorted(fn for fn, c in audit.items()
                      if c["namespace"] == "Set"
                      and any(fn.startswith(p) for p in NEG_PREFIXES))
    neg = take(neg_pool, CAP_NEG)

    # mixed preservation control: a few fresh from each other namespace
    mixed = []
    for ns in ("Finset", "List", "Multiset"):
        pool = sorted(fn for fn, c in audit.items()
                      if c["namespace"] == ns and fn not in KNOWN_BAD)
        mixed += take(pool, 4)
    # plus a couple of Set so the control exercises the gate boundary
    mixed += take(sorted(fn for fn, c in audit.items()
                         if c["namespace"] == "Set"), 4)
    mixed = mixed[:CAP_MIXED]

    sets = {
        "mx2_set_aesop_known": known_set,
        "mx2_set_finite_frontier": set_finite,
        "mx2_set_aesop_frontier": set_frontier,
        "mx2_set_negative_control": neg,
        "mx2_mixed_preservation_control": mixed,
    }
    OUT.write_text(json.dumps(sets, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    total = sum(len(v) for v in sets.values())
    print(f"wrote {OUT.relative_to(ROOT)}  (total {total})")
    for k, v in sets.items():
        print(f"  {k:34s} {len(v)}")


if __name__ == "__main__":
    main()
