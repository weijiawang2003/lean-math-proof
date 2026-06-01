#!/usr/bin/env python3
"""RC4B Part 4 — build validation theorem sets for the disjoint_left bridge candidate.

Sets:
  known_wins                  the 11 extracted RC4B evidence theorems (gate fires; wins).
  fresh_holdout_set           Set disjoint_* theorems NOT among the known wins (gate
                              fires; win NOT guaranteed — out-of-sample generalization).
  fresh_holdout_multiset      Multiset disjoint_* theorems NOT among the known wins.
  disjoint_negative_controls  disjoint_* theorems in OTHER namespaces (Finset/Order/List)
                              — gate must NOT fire (namespace gate suppresses them).
  namespace_negative_controls Nat/List/Finset/Order non-disjoint failures — gate must NOT fire.
  canonical_smoke             demo_v1 + nat_defs_medium + nat_defs_large_v5 floor samples.

Plus a validation_manifest with sizes, evidence overlap, fresh-vs-known split and the
expected gate behaviour per set.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))
import rc4b_gate as G  # noqa: E402

POLICY = "project/evolve/experiments/rc4_candidates/disjoint_left_bridge/disjoint_left_bridge_policy.json"
CONF_SOURCES = [
    "project/evolve/experiments/tr6/out/tr6_rc2_confirmation.json",
    "project/evolve/experiments/tr3/out/tr3_rc2_confirmation.json",
    "project/evolve/experiments/tr5/out/tr5_rc2_confirmation.json",
    "project/evolve/experiments/sf4/out/rc2_failure_confirmation.json",
    "project/evolve/experiments/tr2/out/tr2_rc2_confirmation.json",
]


def _p(*a):
    return os.path.join(_REPO, *a)


def _stmt(r):
    return r.get("statement_text") or r.get("goal_text")


def _entry(policy, full_name, file_path, namespace, goal_text, expected, rc2_hint=None):
    fires, bns, tactics, anames, lemma = G.gate_fires(policy, namespace, goal_text, full_name)
    ns = G.namespace_of(namespace, full_name)
    return {"full_name": full_name, "file_path": file_path, "namespace": ns,
            "goal_text": goal_text, "gate_should_fire": fires,
            "bridge_namespace": bns, "bridge_lemma": lemma,
            "candidate_tactics": tactics, "candidate_actions": anames,
            "rc2_status_hint": rc2_hint, "expected_behavior": expected}


def _is_disjoint(full_name, stmt):
    blob = ((full_name or "") + " " + (stmt or "")).lower()
    return "disjoint" in blob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--known-wins", required=True)
    ap.add_argument("--tr6-frontier", required=True)
    ap.add_argument("--tr6-batch", required=True)
    ap.add_argument("--tr6-confirmation", required=True)
    ap.add_argument("--discovered", default="project/discovered_theorems.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-fresh-set", type=int, default=14)
    ap.add_argument("--max-fresh-multiset", type=int, default=20)
    ap.add_argument("--max-disjoint-neg", type=int, default=15)
    ap.add_argument("--max-ns-neg", type=int, default=20)
    ap.add_argument("--smoke-medium", type=int, default=15)
    ap.add_argument("--smoke-large", type=int, default=15)
    args = ap.parse_args()

    policy = G.load_policy(POLICY)
    out_dir = _p(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    known = json.load(open(_p(args.known_wins)))
    known_names = {w["full_name"] for w in known}

    # ---- pooled candidate rows: eval batch + frontier pool (+ discovered) ----
    batch = json.load(open(_p(args.tr6_batch))).get("theorems", [])
    frontier = [json.loads(l) for l in open(_p(args.tr6_frontier))]
    discovered = (json.load(open(_p(args.discovered))) or {}).get("theorems", [])
    # rc2 status + statement maps from confirmations (reuse-first authority)
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
        if c == "CONFIRMED_RC2_FAILURE":
            return "failed"
        if c == "OPEN_FLAKE":
            return "open_flake"
        return "failed" if c else None

    pool = {}   # full_name -> {file_path, namespace, stmt}
    for r in batch + frontier:
        fn = r.get("full_name")
        if not fn:
            continue
        pool.setdefault(fn, {"file_path": r.get("file_path"),
                             "namespace": r.get("namespace") or fn.split(".")[0],
                             "stmt": _stmt(r)})
    for r in discovered:
        fn = r.get("full_name")
        if not fn or fn in pool:
            continue
        fp = r.get("file_path")
        pool[fn] = {"file_path": fp, "namespace": fn.split(".")[0],
                    "stmt": G.statement_from_source(fp, fn)}

    # ---- 1) known_wins (enriched) ----
    kw = []
    for w in known:
        stmt = _stmt(conf.get(w["full_name"], {})) or w.get("goal_text") \
            or G.statement_from_source(w.get("file_path"), w["full_name"])
        kw.append(_entry(policy, w["full_name"], w["file_path"], w["namespace"], stmt,
                         f"gate fires ({w['bridge_lemma']}); expected reproduction/NEW WIN",
                         rc2_hint(w["full_name"]) or "failed"))

    # ---- 2/3) fresh holdouts (Set / Multiset disjoint, gate fires, not a known win) ----
    fresh_set, fresh_ms = [], []
    chosen = set(known_names)
    # eval-batch first (already RC2-confirmed -> literal RC2 reused), then frontier/discovered
    ordered = list(pool.items())
    ordered.sort(key=lambda kv: (0 if kv[0] in conf else 1, kv[0]))
    for fn, info in ordered:
        if fn in chosen:
            continue
        ns = G.namespace_of(info["namespace"], fn)
        if ns not in ("Set", "Multiset"):
            continue
        if not _is_disjoint(fn, info["stmt"]):
            continue
        fires, *_ = G.gate_fires(policy, ns, info["stmt"], fn)
        if not fires:
            continue
        e = _entry(policy, fn, info["file_path"], ns, info["stmt"],
                   "fresh holdout; gate fires; win NOT guaranteed", rc2_hint(fn))
        if ns == "Set" and len(fresh_set) < args.max_fresh_set:
            fresh_set.append(e); chosen.add(fn)
        elif ns == "Multiset" and len(fresh_ms) < args.max_fresh_multiset:
            fresh_ms.append(e); chosen.add(fn)

    # ---- 4) disjoint_negative_controls: disjoint theorems in OTHER namespaces ----
    disj_neg = []
    for fn, info in ordered:
        if fn in chosen or len(disj_neg) >= args.max_disjoint_neg:
            continue
        ns = G.namespace_of(info["namespace"], fn)
        if ns in ("Set", "Multiset"):
            continue
        if not _is_disjoint(fn, info["stmt"]):
            continue
        e = _entry(policy, fn, info["file_path"], ns, info["stmt"],
                   "disjoint but OTHER namespace; gate must NOT fire (off-gate check)",
                   rc2_hint(fn))
        disj_neg.append(e); chosen.add(fn)

    # ---- 5) namespace_negative_controls: Nat/List/Finset/Order non-disjoint failures ----
    ns_neg = []
    pref = ("Nat", "List", "Finset", "Order")
    cand = []
    for fn, info in ordered:
        if fn in chosen:
            continue
        ns = G.namespace_of(info["namespace"], fn)
        if ns not in pref:
            continue
        if _is_disjoint(fn, info["stmt"]):
            continue
        cand.append((fn, info, ns))
    cand.sort(key=lambda x: (pref.index(x[2]), x[0]))
    for fn, info, ns in cand[: args.max_ns_neg]:
        e = _entry(policy, fn, info["file_path"], ns, info["stmt"],
                   "non-disjoint other namespace; gate must NOT fire (off-gate check)",
                   rc2_hint(fn))
        ns_neg.append(e); chosen.add(fn)

    # ---- 6) canonical_smoke ----
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
        "known_wins": kw,
        "fresh_holdout_set": fresh_set,
        "fresh_holdout_multiset": fresh_ms,
        "disjoint_negative_controls": disj_neg,
        "namespace_negative_controls": ns_neg,
        "canonical_smoke": smoke,
    }
    for name, data in sets.items():
        json.dump(data, open(os.path.join(out_dir, f"{name}.json"), "w"),
                  ensure_ascii=False, indent=2)

    manifest = {
        "generated_by": "scripts/rc4b_build_validation_sets.py",
        "policy": POLICY,
        "bridge_lemmas": policy["bridge_lemmas"],
        "sizes": {k: len(v) for k, v in sets.items()},
        "total": sum(len(v) for v in sets.values()),
        "gate_fire_counts": {k: sum(1 for e in v if e["gate_should_fire"])
                             for k, v in sets.items()},
        "fresh_vs_known": {
            "known_wins": {"known": len(kw), "fresh_evidence": sum(1 for e in known if e.get("fresh"))},
            "fresh_holdout_set": {"fresh_out_of_sample": len(fresh_set)},
            "fresh_holdout_multiset": {"fresh_out_of_sample": len(fresh_ms)},
        },
        "overlap_with_evidence": {k: sum(1 for e in v if e["full_name"] in known_names)
                                  for k, v in sets.items()},
        "expected_gate_emissions": {
            "known_wins": "fires on all (Set/Multiset disjoint) — reproduce wins",
            "fresh_holdout_set": "fires on all; wins NOT guaranteed",
            "fresh_holdout_multiset": "fires on all; wins NOT guaranteed",
            "disjoint_negative_controls": "0 emissions (namespace gate suppresses Finset/Order/List disjoint)",
            "namespace_negative_controls": "0 emissions",
            "canonical_smoke": "≈0 emissions; floors preserved",
        },
        "leakage_notes": [
            "known_wins are the 3 TR3/TR5 Set reproductions + 8 TR6 fresh wins (in-sample by construction).",
            "fresh_holdout_set / fresh_holdout_multiset are disjoint_* theorems that gate-fire "
            "but are NOT credited evidence wins — out-of-sample generalization tests.",
            "disjoint_negative_controls hold the namespace gate honest: Finset/Order/List "
            "disjoint theorems (incl. Finset.disjoint_left, deliberately out of scope) must "
            "show 0 emissions.",
            "namespace_negative_controls + canonical_smoke must show 0 gate emissions / 0 regressions.",
        ],
        "set_files": {k: os.path.relpath(os.path.join(out_dir, f"{k}.json"), _REPO)
                      for k in sets},
    }
    json.dump(manifest, open(os.path.join(out_dir, "validation_manifest.json"), "w"),
              ensure_ascii=False, indent=2)

    print(f"[rc4b-sets] sizes={manifest['sizes']} total={manifest['total']}")
    print(f"           gate_fire_counts={manifest['gate_fire_counts']}")
    print(f"           overlap_with_evidence={manifest['overlap_with_evidence']}")


if __name__ == "__main__":
    main()
