#!/usr/bin/env python3
"""RC4C Part 2 — extract d2_simp_aesop evidence from TR3 / TR5 / TR6.

Pulls every credited win whose winning program is a depth-2 `simp [L] <;> aesop` over a
single retrieved lemma L, records (namespace, winning tactic, lemma L), deduplicates by
theorem, and classifies overlap:

  overlaps_rc4b = L in {Set.disjoint_left, Multiset.disjoint_left}   (RC4B already
                  validated `simp [<NS>.disjoint_left] <;> aesop`)
  overlaps_rc4a = theorem already credited by RC4A def_unfold_simp

A. pure RC4C wins (no RC4A/RC4B overlap), B. RC4B-overlap wins, C. questionable.

Classification per source: TR3 -> TRUE_DELTA, TR5 -> TRUE_RC4C_EVIDENCE (live rank-1
reproduction), TR6 -> FRESH_TRUE_DELTA. A credited win whose program is not actually a
`simp [L] <;> aesop` depth-2 form is marked NEEDS_REVIEW and excluded.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RC4B_LEMMAS = ("Set.disjoint_left", "Multiset.disjoint_left")
_RC4A_CRED = "project/evolve/experiments/rc4_candidates/def_unfold_simp/out/minimal_attribution.json"


def _p(*a):
    return os.path.join(_REPO, *a)


def _load(path):
    return json.load(open(_p(path)))


def _prog(r):
    """Return (tactic_str, used_lemmas_list) for a credited attribution record."""
    wp = r.get("winning_program")
    if isinstance(wp, dict):
        return wp.get("tactic"), (wp.get("used_lemmas") or [])
    tac = wp if isinstance(wp, str) else None
    return tac, (r.get("winning_lemmas") or [])


def _is_d2_simp_aesop(tac):
    return bool(tac) and "<;> aesop" in tac and tac.strip().startswith("simp [")


def _lemma_of(tac, lemmas):
    if lemmas:
        return lemmas[0]
    m = re.search(r"simp \[([^\]]+)\]", tac or "")
    if m:
        return m.group(1).split(",")[0].strip()
    return None


def _fp_map(tr3_results, tr6_plan, extra):
    m = {}
    for r in tr3_results.get("results", []):
        if r.get("file_path"):
            m.setdefault(r["full_name"], r["file_path"])
    for t in tr6_plan.get("theorems", []):
        if t.get("file_path"):
            m.setdefault(t["full_name"], t["file_path"])
    for fn, fp in extra.items():
        m.setdefault(fn, fp)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tr3-attribution", required=True)
    ap.add_argument("--tr3-results", required=True)
    ap.add_argument("--tr3-family", required=False)
    ap.add_argument("--tr3-lemma", required=False)
    ap.add_argument("--tr5-attribution", required=True)
    ap.add_argument("--tr5-evidence", required=True)
    ap.add_argument("--tr6-attribution", required=True)
    ap.add_argument("--tr6-evidence", required=True)
    ap.add_argument("--tr6-plan", required=True)
    ap.add_argument("--rc4b-attribution", required=True)
    ap.add_argument("--rc4b-results", required=True)
    ap.add_argument("--out-known", required=True)
    ap.add_argument("--out-summary-json", required=True)
    ap.add_argument("--out-summary-md", required=True)
    args = ap.parse_args()

    tr3_attr = _load(args.tr3_attribution)
    tr3_results = _load(args.tr3_results)
    tr5_attr = _load(args.tr5_attribution)
    tr6_attr = _load(args.tr6_attribution)
    tr6_plan = _load(args.tr6_plan)
    rc4b_attr = _load(args.rc4b_attribution)

    # known file paths from RC4B known_wins (Set/Multiset disjoint share paths)
    rc4b_fp = {}
    rc4b_known_path = "project/evolve/experiments/rc4_candidates/disjoint_left_bridge/theorem_sets/known_wins.json"
    if os.path.exists(_p(rc4b_known_path)):
        for w in _load(rc4b_known_path):
            if w.get("file_path"):
                rc4b_fp[w["full_name"]] = w["file_path"]
    fpm = _fp_map(tr3_results, tr6_plan, rc4b_fp)

    # RC4A credited targets (for overlaps_rc4a)
    rc4a_cred = set()
    if os.path.exists(_p(_RC4A_CRED)):
        ra = _load(_RC4A_CRED)
        rc4a_cred = set(ra.get("true_def_unfold_win_targets", [])
                        or [r["full_name"] for r in ra.get("records", []) if r.get("credited")])

    raw, needs_review = [], []

    def _add(records, source, fresh, cls):
        for r in records:
            if not r.get("credited"):
                continue
            tac, lemmas = _prog(r)
            if not _is_d2_simp_aesop(tac):
                continue
            L = _lemma_of(tac, lemmas)
            if not L:
                needs_review.append({"full_name": r["full_name"], "source": source,
                                     "reason": "d2 program but no lemma parsed", "tactic": tac})
                continue
            raw.append({"full_name": r["full_name"], "namespace": r.get("namespace"),
                        "winning_tactic": tac, "lemma": L, "source": source,
                        "fresh": fresh, "classification": cls})

    _add(tr3_attr.get("records", []), "TR3", False, "TRUE_DELTA")
    _add(tr5_attr.get("records", []), "TR5", False, "TRUE_RC4C_EVIDENCE")
    _add(tr6_attr.get("records", []), "TR6", True, "FRESH_TRUE_DELTA")

    # dedup by theorem (TR6 fresh dominates)
    by_thm = {}
    for row in raw:
        fn = row["full_name"]
        cur = by_thm.get(fn)
        if cur is None:
            by_thm[fn] = dict(row, sources=[row["source"]])
        else:
            cur["sources"].append(row["source"])
            if row["fresh"] and not cur["fresh"]:
                cur.update({"winning_tactic": row["winning_tactic"], "lemma": row["lemma"],
                            "fresh": True, "classification": "FRESH_TRUE_DELTA"})

    wins = []
    for fn, e in by_thm.items():
        ns = e.get("namespace") or fn.split(".")[0]
        L = e["lemma"]
        ov_b = L in _RC4B_LEMMAS
        ov_a = fn in rc4a_cred
        bucket = "B_overlap_rc4b" if ov_b else ("C_overlap_rc4a" if ov_a else "A_pure_rc4c")
        wins.append({
            "full_name": fn, "file_path": fpm.get(fn), "namespace": ns,
            "winning_tactic": e["winning_tactic"], "lemma": L,
            "source": "+".join(sorted(set(e["sources"]), key=["TR3", "TR5", "TR6"].index)),
            "sources": sorted(set(e["sources"])), "fresh": e["fresh"],
            "classification": e["classification"],
            "overlaps_rc4b": ov_b, "overlaps_rc4a": ov_a, "bucket": bucket,
            "rc2_status": "failed",
        })
    wins.sort(key=lambda w: (w["bucket"], w["namespace"], w["full_name"]))
    json.dump(wins, open(_p(args.out_known), "w"), ensure_ascii=False, indent=2)

    by_ns = Counter(w["namespace"] for w in wins)
    by_lemma = Counter(w["lemma"] for w in wins)
    pure = [w["full_name"] for w in wins if w["bucket"] == "A_pure_rc4c"]
    ov_b = [w["full_name"] for w in wins if w["overlaps_rc4b"]]
    ov_a = [w["full_name"] for w in wins if w["overlaps_rc4a"]]
    fresh = [w["full_name"] for w in wins if w["fresh"]]
    repro = [w["full_name"] for w in wins if not w["fresh"]]
    overlap_dominates = len(ov_b) > len(pure)

    summary = {
        "generated_by": "scripts/rc4c_extract_d2_simp_aesop_evidence.py",
        "num_known_wins": len(wins),
        "by_namespace": dict(by_ns), "by_lemma": dict(by_lemma),
        "pure_rc4c_nonoverlap": {"count": len(pure), "targets": pure},
        "overlap_rc4b": {"count": len(ov_b), "targets": ov_b},
        "overlap_rc4a": {"count": len(ov_a), "targets": ov_a},
        "fresh_wins": {"count": len(fresh), "targets": fresh},
        "reproduction_wins": {"count": len(repro), "targets": repro},
        "overlap_dominates": overlap_dominates,
        "overlap_note": ("MOST RC4C evidence overlaps RC4B (the disjoint_left depth-2 form)."
                         if overlap_dominates else
                         "RC4C has substantial pure (non-overlap) evidence beyond RC4B."),
        "needs_review": needs_review,
        "mechanism": "simp [L] <;> aesop over a small allowlist of retrieved lemmas L (per-lemma narrow gate)",
        "allowlist": sorted(by_lemma.keys()),
    }
    json.dump(summary, open(_p(args.out_summary_json), "w"), ensure_ascii=False, indent=2)

    md = ["# RC4C — d2_simp_aesop evidence", "",
          f"- known wins (deduped): **{len(wins)}**",
          f"- by namespace: {dict(by_ns)}",
          f"- by lemma: {dict(by_lemma)}",
          f"- **pure RC4C (non-overlap): {len(pure)}** {pure}",
          f"- overlap with RC4B: {len(ov_b)} {ov_b}",
          f"- overlap with RC4A: {len(ov_a)} {ov_a}",
          f"- fresh: {len(fresh)} | reproduction: {len(repro)}",
          f"- overlap_dominates: **{overlap_dominates}**",
          f"- needs_review (excluded): {len(needs_review)}", "",
          "| theorem | ns | lemma | tactic | source | fresh | overlap | bucket |",
          "|---|---|---|---|---|---|---|---|"]
    for w in wins:
        ov = "RC4B" if w["overlaps_rc4b"] else ("RC4A" if w["overlaps_rc4a"] else "none")
        md.append(f"| `{w['full_name']}` | {w['namespace']} | `{w['lemma']}` | "
                  f"`{w['winning_tactic']}` | {w['source']} | {w['fresh']} | {ov} | {w['bucket']} |")
    open(_p(args.out_summary_md), "w").write("\n".join(md) + "\n")

    print(f"[rc4c-extract] {len(wins)} known wins | by_ns={dict(by_ns)} | pure={len(pure)} "
          f"overlap_rc4b={len(ov_b)} overlap_rc4a={len(ov_a)} | fresh={len(fresh)} repro={len(repro)} | "
          f"overlap_dominates={overlap_dominates} | needs_review={len(needs_review)}")


if __name__ == "__main__":
    main()
