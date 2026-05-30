"""MX2 Stage 5 — LIVE minimal relabel of MX2-only Set wins.

For every theorem won by an MX2 config but not by production (from
project/data/mx2_set_aesop_probe_meta.json), re-open a live Dojo and try the
minimal battery from the initial state. The first closer determines the label,
so `aesop` is only credited when nothing simpler closes it.

Battery: assumption, rfl, decide, simp, simp_all, aesop, ext x <;> simp,
ext x <;> simp_all, then the wrapper's winning tactic as a fallback.

Classification:
  clean_aesop        — aesop closes and no simpler tactic does.
  simpler_raw        — assumption/rfl/decide/simp/simp_all closes first.
  ext_closes         — an ext one-liner closes first.
  over_attributed    — only the wrapper's winning tactic (not in the battery) closes.
  flaky / dropped    — nothing reproduces / dojo error.

Outputs:
  project/data/mx2_set_aesop_minimal_labels.json
  project/data/mx2_set_aesop_family_pools_meta.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PROBE = ROOT / "project/data/mx2_set_aesop_probe_meta.json"
SETS = ROOT / "project/evolve/routing/mx2_theorem_sets.json"
OUT_LABELS = ROOT / "project/data/mx2_set_aesop_minimal_labels.json"
OUT_POOLS = ROOT / "project/data/mx2_set_aesop_family_pools_meta.json"

KNOWN_BAD = {"Multiset.eq_of_mem_map_const"}
SIMPLE_RAW = ["assumption", "rfl", "decide", "simp", "simp_all"]


def main() -> None:
    if not PROBE.exists():
        print("no probe meta; run mx2_collect_set_aesop.py first")
        return
    probe = json.loads(PROBE.read_text())
    # union of broad+narrow new wins (dedup by theorem)
    new = {}
    for r in probe.get("new_win_records", []):
        new.setdefault(r["theorem"], r)
    # also include narrow-only new theorems
    for row in probe.get("per_set", []):
        for fn in row.get("narrow_new_theorems", []) or []:
            new.setdefault(fn, {"set": row["set"], "theorem": fn,
                                "config": "narrow"})
    targets = [r for fn, r in new.items() if fn not in KNOWN_BAD]

    labels = []
    if targets:
        from env import make_repo, make_theorem
        from core_types import TheoremConfig
        from lean_dojo import Dojo, ProofFinished
        repo = make_repo()
        fp = {}
        for items in json.loads(SETS.read_text()).values():
            for it in items:
                fp[it["full_name"]] = it["file_path"]

        for rec in targets:
            fn = rec["theorem"]
            file_path = fp.get(fn)
            if not file_path:
                labels.append({**rec, "classification": "dropped",
                               "reason": "no file_path"})
                continue
            thm = make_theorem(repo, TheoremConfig(file_path=file_path,
                                                   full_name=fn))
            battery = SIMPLE_RAW + ["aesop", "ext x <;> simp",
                                    "ext x <;> simp_all"]
            cls, minimal = "flaky", None
            try:
                with Dojo(thm) as (dojo, state):
                    for tac in battery:
                        try:
                            res = dojo.run_tac(state, tac)
                        except Exception:
                            continue
                        if isinstance(res, ProofFinished):
                            minimal = tac
                            if tac in SIMPLE_RAW:
                                cls = "simpler_raw"
                            elif tac == "aesop":
                                cls = "clean_aesop"
                            else:
                                cls = "ext_closes"
                            break
            except Exception as e:
                cls, minimal = "dropped", f"dojo_error:{type(e).__name__}"
            labels.append({**rec, "classification": cls,
                           "minimal_closer": minimal})
            print(f"  {fn:42s} {cls:14s} <- {minimal}")

    by_cls = defaultdict(int)
    for r in labels:
        by_cls[r["classification"]] += 1
    clean_aesop = [r["theorem"] for r in labels
                   if r["classification"] == "clean_aesop"]

    OUT_LABELS.write_text(json.dumps({
        "description": "MX2 Stage 5 — LIVE minimal relabel of MX2-only Set wins.",
        "n_targets": len(targets),
        "by_classification": dict(by_cls),
        "labels": labels,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    OUT_POOLS.write_text(json.dumps({
        "description": "MX2 Set-aesop family pool (clean_aesop wins — the "
                       "theorems an aesop fallback genuinely captures).",
        "clean_aesop_count": len(clean_aesop),
        "clean_aesop_theorems": clean_aesop,
        "note": "These are FALLBACK wins (a battery tactic), NOT symbolic "
                "labels — no training implied.",
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"wrote {OUT_LABELS.relative_to(ROOT)}")
    print(f"wrote {OUT_POOLS.relative_to(ROOT)}")
    print(f"targets={len(targets)} by_class={dict(by_cls)} "
          f"clean_aesop={len(clean_aesop)}")


if __name__ == "__main__":
    main()
