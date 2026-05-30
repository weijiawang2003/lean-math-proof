"""AX3 Stage 1 — audit the held-out fresh Multiset induction surface.

WX3 evaluated 165 of the 251 fresh-available Multiset theorems and found the
`induction {var} using Multiset.induction_on <;> simp_all` action closes a
broad set (not only induction-*named* lemmas — many simp/ext-shaped ones too).
AX3 mines the remaining ~86 held-out fresh candidates to grow the clean
symbolic-label pool, reserving disjoint held-out eval sets and a negative
control.

Reads project/data/wx3_multiset_catalog_audit_meta.json (the 251 fresh
candidates, each tagged with shape/difficulty) and tasks.THEOREM_SETS (to
exclude everything already used by WX3 / prior arcs).

Splits the held-out surface into four disjoint sets:
  ax3_multiset_induction_mine     — induction/simp/ext candidates likely
                                     closable by induction_on (label mining)
  ax3_multiset_induction_heldout  — reserved induction/simp (eval, not mined)
  ax3_multiset_mixed_heldout      — reserved mixed/ext (robustness eval)
  ax3_multiset_negative_control   — hard/ext where induction_on+simp is NOT
                                     expected (false-positive eval)

Outputs:
  project/data/ax3_multiset_heldout_audit_meta.json
  project/evolve/reports/ax3_multiset_heldout_audit.md
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WX3_AUDIT = ROOT / "project/data/wx3_multiset_catalog_audit_meta.json"
OUT_META = ROOT / "project/data/ax3_multiset_heldout_audit_meta.json"
OUT_REPORT = ROOT / "project/evolve/reports/ax3_multiset_heldout_audit.md"

# induction-shaped keywords (spec): names that tend to carry a Multiset
# variable amenable to Multiset.induction_on.
IND_KW = ("cons", "induction", "map", "bind", "fold", "count", "erase",
          "attach", "replicate", "nodup", "mem", "sum", "join", "card",
          "add", "sub", "filter", "pmap", "scanl")

# allocation caps (deterministic, disjoint)
MINE_IND, MINE_SIMP, MINE_EXT = 14, 24, 10
HELD_IND, HELD_SIMP = 5, 7
MIXED_EXT = 14


def kw(name: str) -> bool:
    s = name.split(".")[-1].lower()
    return any(k in s for k in IND_KW)


def main() -> None:
    sys.path.insert(0, str(ROOT))
    import tasks

    a = json.loads(WX3_AUDIT.read_text(encoding="utf-8"))
    fresh = a["fresh_candidates"]

    # everything already used anywhere (WX3 + prior arcs)
    used: set[str] = set()
    for thms in tasks.THEOREM_SETS.values():
        for t in thms:
            used.add(t.full_name)

    heldout = [c for c in fresh if c["full_name"] not in used]
    for c in heldout:
        c["induction_keyword"] = kw(c["full_name"])

    by_shape: dict[str, list] = {}
    for c in heldout:
        by_shape.setdefault(c["shape"], []).append(c)

    diff_order = {"easy": 0, "medium": 1, "hard": 2, "?": 3}

    def srt(items, prefer_kw=True):
        return sorted(items, key=lambda c: (
            0 if (prefer_kw and c["induction_keyword"]) else 1,
            diff_order.get(c["difficulty"], 3),
            c.get("num_tactics_approx") or 99,
            c["full_name"]))

    ind = srt(by_shape.get("induction", []))
    simp = srt(by_shape.get("simp", []))
    ext = srt(by_shape.get("ext", []), prefer_kw=False)
    hard = srt(by_shape.get("hard", []), prefer_kw=False)

    taken: set[str] = set()

    def take(pool, n):
        out = []
        for c in pool:
            if c["full_name"] in taken:
                continue
            out.append(c)
            taken.add(c["full_name"])
            if len(out) >= n:
                break
        return out

    mine = take(ind, MINE_IND) + take(simp, MINE_SIMP) + take(ext, MINE_EXT)
    induction_heldout = take(ind, HELD_IND) + take(simp, HELD_SIMP)
    mixed_heldout = take(ext, MIXED_EXT)
    # negative control = remaining ext + all hard (induction_on+simp unlikely)
    negative_control = [c for c in ext if c["full_name"] not in taken] + \
        take(hard, len(hard))
    for c in negative_control:
        taken.add(c["full_name"])

    def emit(items):
        return [{"file_path": c["file"], "full_name": c["full_name"],
                 "namespace": "Multiset", "difficulty": c["difficulty"],
                 "shape": c["shape"],
                 "induction_keyword": c["induction_keyword"]}
                for c in items]

    splits = {
        "ax3_multiset_induction_mine": emit(mine),
        "ax3_multiset_induction_heldout": emit(induction_heldout),
        "ax3_multiset_mixed_heldout": emit(mixed_heldout),
        "ax3_multiset_negative_control": emit(negative_control),
    }
    # strict disjointness assertion
    allnames = [t["full_name"] for v in splits.values() for t in v]
    assert len(allnames) == len(set(allnames)), "splits not disjoint!"

    meta = {
        "source": str(WX3_AUDIT.relative_to(ROOT)),
        "fresh_available_total": a["fresh_available_count"],
        "already_used_count": len(used & {c["full_name"] for c in fresh}),
        "heldout_total": len(heldout),
        "heldout_by_shape": {k: len(v) for k, v in by_shape.items()},
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "split_total": len(allnames),
        "splits": splits,
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    lines = ["# AX3 Stage 1 — held-out Multiset induction surface audit\n"]
    lines.append(f"- fresh available (WX3 catalog): "
                 f"{a['fresh_available_count']}")
    lines.append(f"- used by WX3 / prior arcs (excluded): {len(used & {c['full_name'] for c in fresh})}")
    lines.append(f"- **held-out fresh candidates: {len(heldout)}**")
    lines.append(f"- held-out by shape: "
                 f"{meta['heldout_by_shape']}\n")
    lines.append("## Disjoint AX3 splits\n")
    lines.append("| set | n | role |")
    lines.append("|---|---:|---|")
    roles = {
        "ax3_multiset_induction_mine": "label mining (induction/simp/ext)",
        "ax3_multiset_induction_heldout": "reserved eval (induction/simp)",
        "ax3_multiset_mixed_heldout": "reserved eval (ext/robustness)",
        "ax3_multiset_negative_control": "expected-NULL (ext/hard)",
    }
    for k, v in splits.items():
        lines.append(f"| `{k}` | {len(v)} | {roles[k]} |")
    lines.append(f"\nTotal (disjoint): **{len(allnames)}**")
    lines.append("\n_Generated by scripts/ax3_multiset_heldout_audit.py._")
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {OUT_META.relative_to(ROOT)}")
    print(f"wrote {OUT_REPORT.relative_to(ROOT)}")
    print(f"held-out={len(heldout)} by_shape={meta['heldout_by_shape']}")
    for k, v in splits.items():
        kwn = sum(1 for t in v if t["induction_keyword"])
        print(f"  {k}: {len(v)} ({kwn} kw-match)")
    print(f"  TOTAL disjoint: {len(allnames)}")


if __name__ == "__main__":
    main()
