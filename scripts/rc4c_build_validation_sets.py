#!/usr/bin/env python3
"""RC4C Part 4 — build validation theorem sets for the d2_simp_aesop candidate.

Sets:
  known_wins_all              all 12 attributed RC4C evidence theorems.
  known_wins_nonoverlap       the evidence theorems whose lemma is NOT Set/Multiset.disjoint_left
                              (the genuinely-new-to-RC4 material; RC4C_nonoverlap gate fires).
  fresh_holdout_all           fresh pool theorems where the RC4C_all gate fires, not known.
  fresh_holdout_nonoverlap    fresh pool theorems where the RC4C_nonoverlap gate fires, not known.
  negative_controls           disjoint/pair/biUnion/forall-shaped goals in OTHER namespaces
                              (or with forbidding context) where the gate must NOT fire.
  namespace_negative_controls Nat/Order/List/Finset/Multiset/Set failures outside every gate.
  canonical_smoke             demo_v1 + nat_defs_medium + nat_defs_large_v5 floor samples.

Plus validation_manifest with sizes, overlap with RC4A/RC4B, known-vs-fresh split,
namespace distribution and expected gate emissions per set & per mode.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))
import rc4c_gate as G  # noqa: E402

POLICY = "project/evolve/experiments/rc4_candidates/d2_simp_aesop/d2_simp_aesop_policy.json"
CONF_SOURCES = [
    "project/evolve/experiments/tr6/out/tr6_rc2_confirmation.json",
    "project/evolve/experiments/tr3/out/tr3_rc2_confirmation.json",
    "project/evolve/experiments/tr5/out/tr5_rc2_confirmation.json",
    "project/evolve/experiments/sf4/out/rc2_failure_confirmation.json",
    "project/evolve/experiments/tr2/out/tr2_rc2_confirmation.json",
]
_GATE_NS = ("Set", "Multiset", "Finset", "List")


def _p(*a):
    return os.path.join(_REPO, *a)


def _stmt(r):
    return r.get("statement_text") or r.get("goal_text")


def _entry(policy, full_name, file_path, namespace, goal_text, expected, rc2_hint=None):
    ns = G.namespace_of(namespace, full_name)
    fa, ma_a, ta, na, la = G.gate_fires(policy, namespace, goal_text, full_name, mode="all")
    fn_, ma_n, tn, nn, ln = G.gate_fires(policy, namespace, goal_text, full_name, mode="nonoverlap")
    return {"full_name": full_name, "file_path": file_path, "namespace": ns,
            "goal_text": goal_text,
            "gate_should_fire_all": fa, "gate_should_fire_nonoverlap": fn_,
            "candidate_actions_all": na, "candidate_actions_nonoverlap": nn,
            "candidate_tactics_all": ta, "candidate_tactics_nonoverlap": tn,
            "overlap_family": G.overlap_family_of(ma_a) if fa else "none",
            "rc2_status_hint": rc2_hint, "expected_behavior": expected}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--known-wins", required=True)
    ap.add_argument("--tr6-frontier", required=True)
    ap.add_argument("--tr6-batch", required=True)
    ap.add_argument("--tr6-confirmation", required=True)
    ap.add_argument("--tr6-plan", required=False)
    ap.add_argument("--discovered", default="project/discovered_theorems.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-fresh-all", type=int, default=30)
    ap.add_argument("--max-fresh-nonoverlap", type=int, default=20)
    ap.add_argument("--max-neg", type=int, default=18)
    ap.add_argument("--max-ns-neg", type=int, default=20)
    ap.add_argument("--smoke-medium", type=int, default=15)
    ap.add_argument("--smoke-large", type=int, default=15)
    args = ap.parse_args()

    policy = G.load_policy(POLICY)
    out_dir = _p(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    known = json.load(open(_p(args.known_wins)))
    known_names = {w["full_name"] for w in known}
    nonoverlap_known = {w["full_name"] for w in known if not w["overlaps_rc4b"]}

    # ---- confirmations (rc2 status + statements) ----
    conf = {}
    for src in CONF_SOURCES:
        if not os.path.exists(_p(src)):
            continue
        for r in json.load(open(_p(src))).get("results", []):
            conf.setdefault(r["full_name"], r)

    def rc2_hint(fn):
        r = conf.get(fn)
        if not r:
            return None
        if r.get("rc2_finished"):
            return "solved"
        c = r.get("classification")
        return {"CONFIRMED_RC2_FAILURE": "failed", "OPEN_FLAKE": "open_flake"}.get(c,
               "failed" if c else None)

    # ---- pooled candidate rows ----
    batch = json.load(open(_p(args.tr6_batch))).get("theorems", [])
    frontier = [json.loads(l) for l in open(_p(args.tr6_frontier))]
    discovered = (json.load(open(_p(args.discovered))) or {}).get("theorems", [])
    plan = json.load(open(_p(args.tr6_plan))) if args.tr6_plan and os.path.exists(_p(args.tr6_plan)) else {}
    fp_extra = {t["full_name"]: t.get("file_path") for t in plan.get("theorems", []) if t.get("file_path")}

    pool = {}
    for r in batch + frontier:
        fn = r.get("full_name")
        if not fn:
            continue
        pool.setdefault(fn, {"file_path": r.get("file_path") or fp_extra.get(fn),
                             "namespace": r.get("namespace") or fn.split(".")[0],
                             "stmt": _stmt(r)})
    for r in discovered:
        fn = r.get("full_name")
        if not fn or fn in pool:
            continue
        fp = r.get("file_path") or fp_extra.get(fn)
        pool[fn] = {"file_path": fp, "namespace": fn.split(".")[0],
                    "stmt": G.statement_from_source(fp, fn)}

    # ---- 1/2) known wins ----
    kw_all, kw_non = [], []
    for w in known:
        stmt = _stmt(conf.get(w["full_name"], {})) or w.get("goal_text") \
            or G.statement_from_source(w.get("file_path"), w["full_name"])
        e = _entry(policy, w["full_name"], w["file_path"], w["namespace"], stmt,
                   f"gate fires (L={w['lemma']}); expected reproduction/NEW WIN; "
                   f"overlap={'RC4B' if w['overlaps_rc4b'] else 'none'}",
                   rc2_hint(w["full_name"]) or "failed")
        e["evidence_lemma"] = w["lemma"]
        e["overlaps_rc4b"] = w["overlaps_rc4b"]
        kw_all.append(e)
        if w["full_name"] in nonoverlap_known:
            kw_non.append(e)

    # ---- 3/4) fresh holdouts (gate fires, not a known win) ----
    ordered = list(pool.items())
    ordered.sort(key=lambda kv: (0 if kv[0] in conf else 1, kv[0]))
    fresh_all, fresh_non, chosen = [], [], set(known_names)
    fresh_all_names = set()
    for fn, info in ordered:
        if fn in chosen:
            continue
        ns = G.namespace_of(info["namespace"], fn)
        if ns not in _GATE_NS:
            continue
        fa, *_ = G.gate_fires(policy, ns, info["stmt"], fn, mode="all")
        if not fa:
            continue
        e = _entry(policy, fn, info["file_path"], ns, info["stmt"],
                   "fresh holdout; gate fires; win NOT guaranteed", rc2_hint(fn))
        if len(fresh_all) < args.max_fresh_all:
            fresh_all.append(e); chosen.add(fn); fresh_all_names.add(fn)
            if e["gate_should_fire_nonoverlap"] and len(fresh_non) < args.max_fresh_nonoverlap:
                fresh_non.append(e)
    # top up nonoverlap holdouts if the all-cap filled before enough nonoverlap ones
    if len(fresh_non) < args.max_fresh_nonoverlap:
        for fn, info in ordered:
            if fn in chosen or fn in fresh_all_names:
                continue
            ns = G.namespace_of(info["namespace"], fn)
            if ns not in _GATE_NS:
                continue
            fn_, *_ = G.gate_fires(policy, ns, info["stmt"], fn, mode="nonoverlap")
            if not fn_:
                continue
            e = _entry(policy, fn, info["file_path"], ns, info["stmt"],
                       "fresh holdout (nonoverlap); gate fires; win NOT guaranteed", rc2_hint(fn))
            fresh_non.append(e); chosen.add(fn)
            if len(fresh_non) >= args.max_fresh_nonoverlap:
                break

    # ---- 5) negative_controls: gate-shaped tokens but OTHER namespace / should not fire ----
    neg = []
    SHAPE_TOK = ("disjoint", "pair", "biunion", "forall")
    for fn, info in ordered:
        if fn in chosen or len(neg) >= args.max_neg:
            continue
        ns = G.namespace_of(info["namespace"], fn)
        blob = ((fn or "") + " " + (info["stmt"] or "")).lower()
        if not any(t in blob for t in SHAPE_TOK):
            continue
        fa, *_ = G.gate_fires(policy, ns, info["stmt"], fn, mode="all")
        if fa:
            continue  # would fire -> belongs in a fresh set, not a negative control
        e = _entry(policy, fn, info["file_path"], ns, info["stmt"],
                   "gate-shaped token but out of scope; gate must NOT fire (off-gate check)",
                   rc2_hint(fn))
        neg.append(e); chosen.add(fn)

    # ---- 6) namespace_negative_controls ----
    ns_neg = []
    pref = ("Nat", "Order", "List", "Finset", "Multiset", "Set")
    cand = []
    for fn, info in ordered:
        if fn in chosen:
            continue
        ns = G.namespace_of(info["namespace"], fn)
        if ns not in pref:
            continue
        fa, *_ = G.gate_fires(policy, ns, info["stmt"], fn, mode="all")
        if fa:
            continue
        cand.append((fn, info, ns))
    cand.sort(key=lambda x: (pref.index(x[2]), x[0]))
    for fn, info, ns in cand[: args.max_ns_neg]:
        e = _entry(policy, fn, info["file_path"], ns, info["stmt"],
                   "out-of-gate namespace failure; gate must NOT fire (off-gate check)",
                   rc2_hint(fn))
        ns_neg.append(e); chosen.add(fn)

    # ---- 7) canonical_smoke ----
    smoke = []
    try:
        from tasks import get_theorems
    except Exception:
        get_theorems = None

    def _add_smoke(setname, n):
        if get_theorems is None:
            return
        try:
            ts = get_theorems(setname)
        except Exception:
            return
        sel = ts if (n is None or len(ts) <= n) else ts[:n]
        for t in sel:
            stmt = G.statement_from_source(t.file_path, t.full_name)
            smoke.append(_entry(policy, t.full_name, t.file_path,
                                t.full_name.split(".")[0], stmt,
                                f"canonical floor ({setname}); gate generally must not fire"))
    _add_smoke("demo_v1", None)
    _add_smoke("nat_defs_medium", args.smoke_medium)
    _add_smoke("nat_defs_large_v5", args.smoke_large)

    sets = {
        "known_wins_all": kw_all,
        "known_wins_nonoverlap": kw_non,
        "fresh_holdout_all": fresh_all,
        "fresh_holdout_nonoverlap": fresh_non,
        "negative_controls": neg,
        "namespace_negative_controls": ns_neg,
        "canonical_smoke": smoke,
    }
    for name, data in sets.items():
        json.dump(data, open(os.path.join(out_dir, f"{name}.json"), "w"),
                  ensure_ascii=False, indent=2)

    def _ns_dist(rows):
        from collections import Counter
        return dict(Counter(r["namespace"] for r in rows))

    manifest = {
        "generated_by": "scripts/rc4c_build_validation_sets.py",
        "policy": POLICY,
        "allowlist_lemmas": policy["allowlist_lemmas"],
        "sizes": {k: len(v) for k, v in sets.items()},
        "total": sum(len(v) for v in sets.values()),
        "unique_total": len({e["full_name"] for v in sets.values() for e in v}),
        "gate_fire_counts_all": {k: sum(1 for e in v if e["gate_should_fire_all"])
                                 for k, v in sets.items()},
        "gate_fire_counts_nonoverlap": {k: sum(1 for e in v if e["gate_should_fire_nonoverlap"])
                                        for k, v in sets.items()},
        "namespace_distribution": {k: _ns_dist(v) for k, v in sets.items()},
        "overlap_with_evidence": {k: sum(1 for e in v if e["full_name"] in known_names)
                                  for k, v in sets.items()},
        "known_vs_fresh": {
            "known_wins_all": len(kw_all), "known_wins_nonoverlap": len(kw_non),
            "fresh_holdout_all": len(fresh_all), "fresh_holdout_nonoverlap": len(fresh_non),
        },
        "expected_gate_emissions": {
            "known_wins_all": "RC4C_all fires on all (reproduce)",
            "known_wins_nonoverlap": "RC4C_nonoverlap fires on all 4 pure theorems",
            "fresh_holdout_all": "RC4C_all fires on all; wins NOT guaranteed",
            "fresh_holdout_nonoverlap": "RC4C_nonoverlap fires on all; wins NOT guaranteed",
            "negative_controls": "0 emissions (out-of-scope namespace / forbidden context)",
            "namespace_negative_controls": "0 emissions",
            "canonical_smoke": "≈0 emissions; floors preserved",
        },
        "set_files": {k: os.path.relpath(os.path.join(out_dir, f"{k}.json"), _REPO)
                      for k in sets},
    }
    json.dump(manifest, open(os.path.join(out_dir, "validation_manifest.json"), "w"),
              ensure_ascii=False, indent=2)

    print(f"[rc4c-sets] sizes={manifest['sizes']} total={manifest['total']} "
          f"unique={manifest['unique_total']}")
    print(f"           gate_fire_all={manifest['gate_fire_counts_all']}")
    print(f"           gate_fire_nonoverlap={manifest['gate_fire_counts_nonoverlap']}")
    print(f"           overlap_with_evidence={manifest['overlap_with_evidence']}")


if __name__ == "__main__":
    main()
