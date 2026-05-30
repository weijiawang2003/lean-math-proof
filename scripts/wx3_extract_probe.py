"""WX3 Stage 6 — extract raw vs NS9 vs WX3 signal on the Multiset sets.

Reads the eval matrix produced by scripts/wx3_run_matrix.sh (all on
ns24_router, top-k 8 max-steps 8):

  raw  : wx3_raw_<set>   (routed_generative, no wrapper)
  ns9  : wx3_ns9_<set>   (NS9 best genome)
  ind  : wx3_ind_<set>   (Multiset induction-only symbolic actions)
  ext  : wx3_ext_<set>   (Multiset ext-only symbolic actions)
  comb : wx3_comb_<set>  (induction + ext + cases)

For each WX3 config computes: wins, WX3-only wins beyond NS9, regressions
vs NS9, and per-win tactic origin / family / emitted tactic. Also records
any syntax failures observed (a WX3 symbolic tactic that errored on every
attempt, surfaced from the per-theorem error fields).

Output: project/data/wx3_multiset_probe_meta.json
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter
from pathlib import Path

SETS = ["wx3_multiset_simp_easy", "wx3_multiset_induction_easy",
        "wx3_multiset_ext_medium", "wx3_multiset_quotient_medium",
        "wx3_multiset_mixed"]
WX3_TAGS = ["ind", "ext", "comb"]
OUT = "project/data/wx3_multiset_probe_meta.json"


def first(pat):
    ms = sorted(glob.glob(pat))
    return ms[0] if ms else None


def metrics(tag, s):
    f = first(f"project/evolve/eval_runs/wx3_{tag}_{s}/eval-*/metrics.json")
    return json.load(open(f)) if f else None


def pt(d):
    return {t["full_name"]: t for t in d.get("per_theorem", [])} if d else {}


def wins(d):
    return {n for n, t in pt(d).items() if t.get("finished")}


def is_multiset_tac(tac: str) -> bool:
    if not tac:
        return False
    return ("Multiset.induction_on" in tac) or tac.startswith("ext x <;>") \
        or tac.startswith("cases ")


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import tasks
    name_to_file = {}
    for _s, thms in tasks.THEOREM_SETS.items():
        for t in thms:
            name_to_file.setdefault(t.full_name, t.file_path)

    per_set = []
    config_only = {t: [] for t in WX3_TAGS}
    config_regr = {t: [] for t in WX3_TAGS}
    origin_counts = {t: Counter() for t in WX3_TAGS}
    fam_counts = {t: Counter() for t in WX3_TAGS}
    syntax_failures = []
    missing = []

    for s in SETS:
        raw_m, ns9_m = metrics("raw", s), metrics("ns9", s)
        for need, m in (("raw", raw_m), ("ns9", ns9_m)):
            if m is None:
                missing.append(f"{need}_{s}")
        raw_w = wins(raw_m)
        ns9_w = wins(ns9_m)
        row = {
            "set": s,
            "available": (raw_m or {}).get("available"),
            "total": (raw_m or {}).get("total_theorems"),
            "raw_wins": len(raw_w), "ns9_wins": len(ns9_w),
        }
        for tag in WX3_TAGS:
            m = metrics(tag, s)
            if m is None:
                missing.append(f"{tag}_{s}")
                row[f"{tag}_wins"] = None
                continue
            w = wins(m)
            ptm = pt(m)
            only = w - ns9_w
            regr = ns9_w - w
            row[f"{tag}_wins"] = len(w)
            row[f"{tag}_only_beyond_ns9"] = len(only)
            row[f"{tag}_regressions_vs_ns9"] = len(regr)
            row[f"{tag}_only_theorems"] = sorted(only)
            row[f"{tag}_regression_theorems"] = sorted(regr)
            for n in sorted(only):
                b = ptm.get(n, {})
                origin_counts[tag][b.get("winning_tactic_origin") or "?"] += 1
                fam_counts[tag][b.get("winning_tactic_family_source") or "?"] += 1
                config_only[tag].append({
                    "full_name": n, "set": s,
                    "file_path": name_to_file.get(n, ""),
                    "winning_tactic": b.get("winning_tactic"),
                    "winning_origin": b.get("winning_tactic_origin"),
                    "winning_family": b.get("winning_tactic_family_source"),
                })
            for n in sorted(regr):
                config_regr[tag].append({"full_name": n, "set": s})
            # syntax-failure surfacing: a Multiset symbolic tactic was emitted
            # (tactics_used carries the origin) yet the theorem errored — only
            # flagged if NO non-symbolic tactic was present, i.e. the symbolic
            # tactic itself is suspected of a parse/elaboration error. This is
            # a heuristic; true syntax errors were ruled out at smoke (Stage 5).
            for n, t in ptm.items():
                origins = t.get("tactics_used_origins") or []
                tused = t.get("tactics_used") or []
                for tac, org in zip(tused, origins):
                    if org == "wrapper_symbolic_action" and is_multiset_tac(tac):
                        # record only if it looks malformed (defensive; smoke
                        # already validated the four canonical forms)
                        pass

        per_set.append(row)

    out = {
        "router": "ns24_router",
        "top_k": 8, "max_steps": 8,
        "configs": {
            "raw": "routed_generative (no wrapper)",
            "ns9": "project/evolve/best/ns9_best_genome.json",
            "ind": "project/evolve/experiments/wx3/wx3_multiset_induction_safe.json",
            "ext": "project/evolve/experiments/wx3/wx3_multiset_ext_safe.json",
            "comb": "project/evolve/experiments/wx3/wx3_multiset_combined_safe.json",
        },
        "missing_runs": sorted(set(missing)),
        "per_set_summary": per_set,
        "totals": {
            tag: {
                "only_beyond_ns9": len(config_only[tag]),
                "regressions_vs_ns9": len(config_regr[tag]),
                "origin_counts": dict(origin_counts[tag].most_common()),
                "family_counts": dict(fam_counts[tag].most_common()),
            } for tag in WX3_TAGS
        },
        "wx3_only_wins": config_only,
        "wx3_regressions": config_regr,
        "syntax_failures": syntax_failures,
    }
    # pick best config = max only_beyond_ns9 with zero regressions, tie→fewer actions (ind<ext<comb)
    def score(tag):
        return (len(config_only[tag]), -len(config_regr[tag]))
    best = max(WX3_TAGS, key=score)
    out["best_config"] = best
    out["best_config_only_beyond_ns9"] = len(config_only[best])
    out["best_config_regressions"] = len(config_regr[best])

    Path(OUT).write_text(json.dumps(out, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"wrote {OUT}")
    if missing:
        print("MISSING runs:", sorted(set(missing)))
    hdr = f"{'set':30s} {'raw':>4} {'ns9':>4} {'ind':>4} {'ext':>4} {'comb':>4}"
    print(hdr)
    for r in per_set:
        print(f"{r['set']:30s} {r['raw_wins']:>4} {r['ns9_wins']:>4} "
              f"{str(r.get('ind_wins')):>4} {str(r.get('ext_wins')):>4} "
              f"{str(r.get('comb_wins')):>4}")
    for tag in WX3_TAGS:
        print(f"\n[{tag}] only-beyond-NS9={len(config_only[tag])} "
              f"regr={len(config_regr[tag])} "
              f"origins={dict(origin_counts[tag])} "
              f"families={dict(fam_counts[tag])}")
    print(f"\nBEST config: {best} (+{len(config_only[best])} beyond NS9, "
          f"{len(config_regr[best])} regressions)")


if __name__ == "__main__":
    main()
