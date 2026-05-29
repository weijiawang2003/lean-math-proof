"""WX3 Stage 2 — build fresh Multiset theorem sets from the audit.

Reads project/data/wx3_multiset_catalog_audit_meta.json and writes five
disjoint sets to project/evolve/routing/wx3_theorem_sets.json (loaded by
tasks._load_wx3_sets). All candidates are confirmed-available, fresh
(not in any prior theorem set), and namespaced `Multiset.`.

Sets:
  wx3_multiset_simp_easy       — simp-only-leaning facts (easy/medium)
  wx3_multiset_induction_easy  — recursive-structure lemmas → induction_on
  wx3_multiset_ext_medium      — extensionality / count reasoning
  wx3_multiset_quotient_medium — Quot/induction_on territory (+hard induction)
  wx3_multiset_mixed           — balanced disjoint sample across shapes

Disjoint by construction: each theorem appears in at most one set.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / "project/data/wx3_multiset_catalog_audit_meta.json"
OUT = ROOT / "project/evolve/routing/wx3_theorem_sets.json"

SIMP_CAP = 40
IND_CAP = 40
EXT_CAP = 35
QUOT_CAP = 20
MIXED_CAP = 30


def emit(items):
    out, seen = [], set()
    for c in items:
        if c["full_name"] in seen:
            continue
        seen.add(c["full_name"])
        out.append({
            "file_path": c["file"],
            "full_name": c["full_name"],
            "namespace": "Multiset",
            "difficulty": c.get("difficulty", "?"),
            "shape": c["shape"],
        })
    return out


def main() -> None:
    m = json.loads(AUDIT.read_text(encoding="utf-8"))
    cands = m["fresh_candidates"]
    by_shape: dict[str, list] = {}
    for c in cands:
        by_shape.setdefault(c["shape"], []).append(c)

    def srt(items):
        # easy/medium first, then by tactic-count (simpler first), then name
        order = {"easy": 0, "medium": 1, "hard": 2, "?": 3}
        return sorted(items, key=lambda c: (
            order.get(c["difficulty"], 3),
            c.get("num_tactics_approx") or 99,
            c["full_name"]))

    used: set[str] = set()

    def take(items, cap):
        picked = []
        for c in items:
            if c["full_name"] in used:
                continue
            picked.append(c)
            used.add(c["full_name"])
            if len(picked) >= cap:
                break
        return picked

    simp = take(srt([c for c in by_shape.get("simp", [])
                     if c["difficulty"] in ("easy", "medium")]), SIMP_CAP)
    ind = take(srt([c for c in by_shape.get("induction", [])
                    if c["difficulty"] in ("easy", "medium")]), IND_CAP)
    ext = take(srt(by_shape.get("ext", [])), EXT_CAP)
    # quotient is tiny (6); pad with the hardest induction lemmas, which are
    # the ones most likely to genuinely need Multiset.induction_on.
    quot_pool = srt(by_shape.get("quotient", [])) + sorted(
        [c for c in by_shape.get("induction", [])
         if c["difficulty"] == "hard"],
        key=lambda c: c["full_name"])
    quot = take(quot_pool, QUOT_CAP)

    # mixed = balanced sample of whatever remains, across all shapes
    mixed = []
    rotation = ["simp", "induction", "ext", "quotient", "hard"]
    pools = {s: srt(by_shape.get(s, [])) for s in rotation}
    idx = {s: 0 for s in rotation}
    while len(mixed) < MIXED_CAP:
        progressed = False
        for s in rotation:
            pool = pools[s]
            while idx[s] < len(pool):
                c = pool[idx[s]]
                idx[s] += 1
                if c["full_name"] not in used:
                    mixed.append(c)
                    used.add(c["full_name"])
                    progressed = True
                    break
            if len(mixed) >= MIXED_CAP:
                break
        if not progressed:
            break

    # tiny hand-spanning smoke set (2 per shape) drawn from the chosen sets,
    # for Stage 5 syntax smoke testing only.
    chosen = {"simp": simp, "induction": ind, "ext": ext, "quotient": quot}
    smoke = []
    for s in ("simp", "induction", "ext", "quotient"):
        smoke.extend(chosen[s][:2])

    sets = {
        "wx3_multiset_simp_easy": emit(simp),
        "wx3_multiset_induction_easy": emit(ind),
        "wx3_multiset_ext_medium": emit(ext),
        "wx3_multiset_quotient_medium": emit(quot),
        "wx3_multiset_mixed": emit(mixed),
        "wx3_multiset_smoke": emit(smoke),
    }
    OUT.write_text(json.dumps(sets, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    total = sum(len(v) for v in sets.values())
    print(f"wrote {OUT.relative_to(ROOT)}")
    for k, v in sets.items():
        print(f"  {k}: {len(v)}")
    print(f"  TOTAL (disjoint): {total}")


if __name__ == "__main__":
    main()
