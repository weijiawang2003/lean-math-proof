"""NS4 mini skeleton-level mutator (proof-of-concept).

Reads the ns3-combined genome, applies one skeleton-level edit per
variant, and writes a fresh genome.json + a label. Each variant is
intended to be fed to `python -m evolve.run_large_v5 --best-genome
<path>` against nat_defs_medium.

The four variant types match `ns4_skeleton_bag_design_note.md` Stage 7:

  reorder_iff:       move iff-slot specifics to AFTER iff-slot generics.
                     Predicts a regression (NS1 said specifics first).
  disable_one:       drop one iff-slot generic skeleton (the catch-all
                     `constructor <;> intro h_split <;> simp_all` — the
                     observed top winner). Predicts a regression.
  duplicate_to_lt:   copy one any-slot specific (the one with hyp_pos)
                     into the lt-slot. Predicts neutral or +0 (likely
                     already shadowed by base path).

Run all three by writing genomes to /tmp/ns4_variants/<label>/genome.json
and looping `run_large_v5` over them.

Skeleton-bag is enabled (`use_skeleton_bag=True`) on every variant so
the edits flow through the new code path.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


SRC = "project/evolve/autonomous_runs/v5-ns3-20260522-222000-9beeab/eval/ns3-combined/genome.json"
DST_ROOT = Path("/tmp/ns4_variants")


def _classify(t: str) -> int:
    from evolve.strategy_wrapper import classify_template_specificity
    return classify_template_specificity(t)[0]


def variant_reorder_iff(g: dict) -> dict:
    """Move all iff-slot specifics to AFTER all iff-slot generics.

    Because the priority block stable-sorts each slot by specificity at
    emission time, just reordering the *raw* list wouldn't change
    output. To actually move them, we have to bypass the NS1 sort —
    which is the legacy invariant we're testing. Here, we tag specifics
    as 'generic' by wrapping them in a generic-classified prefix that
    keeps the same tactic. That demonstrates the mutator can break
    NS1 (and predicts a regression).

    Simpler approach: swap the raw lists themselves AND set
    use_skeleton_bag=True. The bag still NS1-sorts internally, so this
    variant is mostly a no-op against the legacy. Instead we DROP the
    specifics from the iff slot and KEEP only the generics. This is
    closer to "what if you move all specifics out of the iff slot."
    """
    g = copy.deepcopy(g)
    pt = g["priority_templates"]
    iff = pt.get("iff", [])
    specifics = [t for t in iff if _classify(t) == 0]
    generics = [t for t in iff if _classify(t) == 1]
    # Skeleton-edit: remove specifics from iff (regression predicted).
    pt["iff"] = generics
    g["use_skeleton_bag"] = True
    return g


def variant_disable_one(g: dict) -> dict:
    """Disable the iff catch-all generic that observed-wins 17 proofs.

    Tests the cost of removing the highest-firing single skeleton.
    """
    g = copy.deepcopy(g)
    pt = g["priority_templates"]
    iff = pt.get("iff", [])
    target = "constructor <;> intro h_split <;> simp_all"
    pt["iff"] = [t for t in iff if t != target]
    g["use_skeleton_bag"] = True
    return g


def variant_duplicate_to_lt(g: dict) -> dict:
    """Duplicate one iff-slot specific (one that does not need hyp_pos)
    into the lt-slot. Tests cross-shape skeleton-transfer.
    """
    g = copy.deepcopy(g)
    pt = g["priority_templates"]
    iff = pt.get("iff", [])
    # Pick a specific that already names a Nat lemma and doesn't need a
    # hypothesis the lt-state may lack. The classification heuristic
    # treats Nat.* mentions as specific.
    candidates = [t for t in iff if _classify(t) == 0 and "{hyp_" not in t]
    chosen = candidates[0] if candidates else None
    if chosen is not None:
        lt = list(pt.get("lt", []))
        if chosen not in lt:
            lt.append(chosen)
        pt["lt"] = lt
    g["use_skeleton_bag"] = True
    return g


def main() -> int:
    src = json.loads(Path(SRC).read_text())
    DST_ROOT.mkdir(parents=True, exist_ok=True)
    variants = [
        ("reorder_iff", variant_reorder_iff),
        ("disable_one", variant_disable_one),
        ("duplicate_to_lt", variant_duplicate_to_lt),
    ]
    for label, fn in variants:
        out_dir = DST_ROOT / label
        out_dir.mkdir(parents=True, exist_ok=True)
        g = fn(src)
        (out_dir / "genome.json").write_text(json.dumps(g, indent=2, ensure_ascii=False))
        print(f"[{label}]")
        print(f"  genome -> {out_dir / 'genome.json'}")
        # Tiny summary diff
        src_iff = src["priority_templates"].get("iff", [])
        new_iff = g["priority_templates"].get("iff", [])
        if len(src_iff) != len(new_iff):
            print(f"  iff slot: {len(src_iff)} -> {len(new_iff)} entries")
        for shape in ("lt", "le", "eq", "any"):
            sl = len(src["priority_templates"].get(shape, []))
            nl = len(g["priority_templates"].get(shape, []))
            if sl != nl:
                print(f"  {shape} slot: {sl} -> {nl} entries")
    print(f"\nTo eval: for d in {DST_ROOT}/*/; do echo $d; python -m evolve.run_large_v5 --best-genome $d/genome.json --theorem-set nat_defs_medium --ckpt-dir project/models/gen_v5 --out-dir /tmp/ns4_variants_run/$(basename $d) --timeout-seconds 1200; done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
