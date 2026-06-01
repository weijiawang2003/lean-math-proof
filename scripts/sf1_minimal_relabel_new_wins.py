#!/usr/bin/env python3
"""SF1 stage (f): minimal-sufficient relabeling QUEUE.

Builds a real, artifact-producing queue that downstream NS23-style minimal-tactic
relabeling will consume. Full tactical replay (live Lean) is still partial, so
this stage produces the *queue + schema + family-specific ladders*, not confirmed
wins. A theorem is only ever a CANDIDATE here; a new family win is claimed only
after minimal-sufficient relabeling confirms it (NS23 discipline).

Inputs:
  classified_frontier.jsonl   (Stage C)
  eval_matrix_results.json     (Stage E; may be dry-run with no solved/failed)
  sf1_candidate_families.json  (family registry)

Outputs:
  relabel_queue.jsonl
  relabel_summary.json

Queue criteria (any one):
  - RC1 failed but experimental solved        (from eval results, if available)
  - raw failed but RC1 solved                 (")
  - NS9 failed but RC1 solved                 (")
  - RC1 failed AND classification suggests a strong candidate family
  - high-confidence missing-lemma candidate (future_failure_driven score high)
  - high Set.Finite/toFinset or Multiset family score
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import sf1_common as C
    _read = C.read_json_or_jsonl
    _write = C.write_jsonl
    _ensure = C.ensure_parent_dir
except Exception:  # pragma: no cover
    def _read(path):
        rows = []
        if not os.path.isfile(path):
            return rows
        with open(path, encoding="utf-8", errors="replace") as fh:
            txt = fh.read()
        try:
            obj = json.loads(txt)
            return obj if isinstance(obj, list) else [obj]
        except json.JSONDecodeError:
            for line in txt.splitlines():
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _ensure(path):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        return path

    def _write(rows, path):
        _ensure(path)
        n = 0
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
        return n

LADDERS = {
    "wx3_multiset_induction": [
        "simp", "simp_all", "aesop",
        "induction {var} using Multiset.induction_on <;> simp_all",
    ],
    "mx2_set_finite_tofinset_aesop": [
        "simp", "simp_all", "aesop", "classical; aesop", "try exact?",
    ],
    "arith": ["simp", "omega", "simp_all <;> omega"],
    "extensionality": ["simp", "ext", "aesop"],
    "default": ["simp", "simp_all", "aesop"],
}
FAMILY_PROBE = {
    "wx3_multiset_induction": "induction {var} using Multiset.induction_on <;> simp_all",
    "mx2_set_finite_tofinset_aesop": "Set.Finite/toFinset bridge: simp [Set.Finite.toFinset]; aesop",
}
REJECTED = {"broad_set_aesop_rejected", "mx1_set_finset_symbolic_rejected"}


def _status_map(eval_results):
    """Return {decl_name: {policy: solved_bool}} from real eval results.

    Reads the live per-theorem records written by sf1_eval_matrix.py for runs
    whose status == 'ran'. Dry-run / blocked / failed entries contribute nothing
    (we never infer per-theorem status from an aggregate string).
    """
    out = {}
    for res in (eval_results or {}).get("results", []):
        if res.get("status") != "ran":
            continue
        policy = res.get("policy")
        for t in (res.get("per_theorem") or []):
            n = t.get("full_name") or t.get("theorem") or t.get("decl_name")
            if not n:
                continue
            out.setdefault(n, {})[policy] = bool(t.get("solved"))
    return out


def _ladder_for(top_family, tags):
    if top_family == "wx3_multiset_induction" or "likely_multiset_induction" in tags:
        return "wx3_multiset_induction", LADDERS["wx3_multiset_induction"]
    if top_family == "mx2_set_finite_tofinset_aesop" or "likely_set_finite_bridge" in tags:
        return "mx2_set_finite_tofinset_aesop", LADDERS["mx2_set_finite_tofinset_aesop"]
    if "likely_omega" in tags:
        return "arith", LADDERS["arith"]
    if "likely_extensionality" in tags:
        return "extensionality", LADDERS["extensionality"]
    return "default", LADDERS["default"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SF1 (f): minimal-relabel queue.")
    p.add_argument("--classified-frontier",
                   default="project/evolve/experiments/sf1/out/real/classified_frontier.jsonl")
    p.add_argument("--eval-results",
                   default="project/evolve/experiments/sf1/out/real/eval_matrix_results.json")
    p.add_argument("--family-registry",
                   default="project/evolve/experiments/sf1/sf1_candidate_families.json")
    p.add_argument("--out",
                   default="project/evolve/experiments/sf1/out/real/relabel_queue.jsonl")
    p.add_argument("--summary-out",
                   default="project/evolve/experiments/sf1/out/real/relabel_summary.json")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not os.path.isfile(args.classified_frontier):
        print(f"[sf1:relabel] ERROR: classified frontier missing: "
              f"{args.classified_frontier}", file=sys.stderr)
        return 2
    classified = _read(args.classified_frontier)
    eval_results = None
    if os.path.isfile(args.eval_results):
        try:
            eval_results = json.load(open(args.eval_results))
        except Exception:
            eval_results = None
    per_thm_status = _status_map(eval_results)
    eval_has_perthm = bool(per_thm_status)

    queue = []
    for r in classified:
        name = r.get("decl_name")
        if not name:
            continue
        tags = r.get("tags", [])
        scores = r.get("candidate_family_scores", {})
        top = r.get("top_candidate_family")
        conf = r.get("classification_confidence", 1.0)

        wx3 = scores.get("wx3_multiset_induction", 0.0)
        mx2 = scores.get("mx2_set_finite_tofinset_aesop", 0.0)
        fdl = scores.get("future_failure_driven_lemma_candidate", 0.0)
        rc1 = scores.get("rc1_production_stack", 0.0)

        reasons = []
        # eval-driven reasons (only when a real run gave per-theorem status)
        st = per_thm_status.get(name, {})
        if st:
            if st.get("experimental") and not st.get("rc1"):
                reasons.append("RC1 failed but experimental solved")
            if st.get("rc1") and not st.get("raw"):
                reasons.append("raw failed but RC1 solved")
            if st.get("rc1") and not st.get("ns9"):
                reasons.append("NS9 failed but RC1 solved")
            if not st.get("rc1") and max(wx3, mx2) >= 0.6:
                reasons.append("RC1 failed and strong candidate family")
        # classification-driven reasons (always available)
        if wx3 >= 0.6:
            reasons.append(f"high wx3 multiset-induction score ({wx3})")
        if mx2 >= 0.6:
            reasons.append(f"high mx2 Set.Finite/toFinset score ({mx2})")
        if fdl >= 0.5:
            reasons.append(f"high-uncertainty missing-lemma candidate ({fdl})")

        if not reasons:
            continue

        fam_key, ladder = _ladder_for(top, tags)
        # candidate family attribution: prefer a productive family, else the
        # discovery placeholder.
        candidate_family = top
        cand_score = scores.get(top, 0.0)
        if top in REJECTED:
            candidate_family = "future_failure_driven_lemma_candidate"
            cand_score = fdl

        if max(wx3, mx2) >= 0.75 or (st and st.get("experimental") and not st.get("rc1")):
            priority = "high"
        elif max(wx3, mx2, fdl) >= 0.55:
            priority = "medium"
        else:
            priority = "low"

        notes = []
        if not eval_has_perthm:
            notes.append("queued from classification only (no live per-theorem eval yet); "
                         "not a confirmed win — needs minimal-sufficient relabel")
        if top in REJECTED:
            notes.append(f"classifier top family '{top}' is rejected/off; routed to "
                         f"failure-driven lemma discovery instead")

        var = "s" if "has_multiset" in tags else "x"
        queue.append({
            "decl_name": name,
            "namespace": r.get("namespace"),
            "reason": "; ".join(reasons),
            "baseline_status": st or {"note": "no live per-theorem eval available"},
            "candidate_family": candidate_family,
            "candidate_family_score": round(cand_score, 3),
            "suggested_relabel_ladder": [t.replace("{var}", var) for t in ladder],
            "family_specific_probe": FAMILY_PROBE.get(candidate_family,
                                                      "").replace("{var}", var) or None,
            "priority": priority,
            "notes": notes,
        })

    # deterministic order: priority then score then name
    prio_rank = {"high": 0, "medium": 1, "low": 2}
    queue.sort(key=lambda q: (prio_rank.get(q["priority"], 3),
                              -q["candidate_family_score"], q["decl_name"]))
    _write(queue, args.out)

    summary = {
        "queue_size": len(queue),
        "eval_per_theorem_available": eval_has_perthm,
        "by_priority": dict(Counter(q["priority"] for q in queue)),
        "by_candidate_family": dict(Counter(q["candidate_family"] for q in queue)),
        "by_namespace": dict(Counter(q["namespace"] for q in queue)),
        "high_priority_decls": [q["decl_name"] for q in queue if q["priority"] == "high"],
        "note": ("Queue built from classification (Stage C) + eval status (Stage E). "
                 "No new family win is claimed; NS23 minimal-sufficient relabel must "
                 "confirm each candidate before any promotion."),
    }
    _ensure(args.summary_out)
    json.dump(summary, open(args.summary_out, "w"), ensure_ascii=False, indent=2)

    print(f"[sf1:relabel] queue={len(queue)} per_theorem_eval={eval_has_perthm} "
          f"by_priority={summary['by_priority']} "
          f"by_family={summary['by_candidate_family']}")
    print(f"[sf1:relabel] -> {args.out}\n              -> {args.summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
