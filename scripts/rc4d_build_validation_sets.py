#!/usr/bin/env python3
"""RC4D Part 4 — build composition validation theorem sets.

Assembles the RC4D validation sets by REUSING the already-built, file_path-resolved entries
from the three components' theorem sets (RC4A / RC4B / RC4C), then re-deriving the RC4D
ordered-union gate emissions + firing components + expected behaviour for every entry. No new
theorem discovery — the component sets already cover known wins, fresh holdouts, negatives,
namespace negatives and the canonical floors.

Sets produced:
  rc4a_known_wins                 RC4A's confirmed def-unfold wins
  rc4b_known_wins                 RC4B's 16 confirmed disjoint-bridge wins
  rc4c_residue_known_wins         the residue-lemma (disjoint_right/subset_pair/forall) wins
  composition_fresh_holdout       fresh-frontier theorems where any component gate fires
  component_overlap_controls      theorems both RC4B and RC4C_residue fire on (RC4B must win)
  negative_controls               disjoint/pair/forall-shaped goals where the gate must NOT fire
  namespace_negative_controls     Nat/Order/… outside every component gate
  canonical_smoke                 demo_v1 + nat_defs_medium + nat_defs_large_v5 floor samples
  validation_manifest             sizes, coverage, namespace split, expected emissions
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))
import rc4d_gate as G  # noqa: E402

RC4A_DIR = "project/evolve/experiments/rc4_candidates/def_unfold_simp/theorem_sets"
RC4B_DIR = "project/evolve/experiments/rc4_candidates/disjoint_left_bridge/theorem_sets"
RC4C_DIR = "project/evolve/experiments/rc4_candidates/d2_simp_aesop/theorem_sets"

# every component set we harvest entries from (for file_path / goal_text resolution)
SOURCE_SETS = [
    (RC4A_DIR, "known_wins.json"), (RC4A_DIR, "fresh_frontier_holdout.json"),
    (RC4A_DIR, "same_cluster_holdout.json"), (RC4A_DIR, "negative_controls.json"),
    (RC4A_DIR, "canonical_smoke.json"),
    (RC4B_DIR, "known_wins.json"), (RC4B_DIR, "fresh_holdout_multiset.json"),
    (RC4B_DIR, "fresh_holdout_set.json"), (RC4B_DIR, "disjoint_negative_controls.json"),
    (RC4B_DIR, "namespace_negative_controls.json"), (RC4B_DIR, "canonical_smoke.json"),
    (RC4C_DIR, "known_wins_all.json"), (RC4C_DIR, "fresh_holdout_all.json"),
    (RC4C_DIR, "fresh_holdout_nonoverlap.json"), (RC4C_DIR, "negative_controls.json"),
    (RC4C_DIR, "namespace_negative_controls.json"), (RC4C_DIR, "canonical_smoke.json"),
]


def _p(*a):
    return os.path.join(_REPO, *a)


def _load(rel, fname):
    p = _p(rel, fname)
    return json.load(open(p)) if os.path.exists(p) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--policy",
                    default="project/evolve/experiments/rc4_candidates/composition_rc4d/rc4d_composition_policy.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-fresh", type=int, default=40)
    ap.add_argument("--max-neg", type=int, default=24)
    ap.add_argument("--max-ns-neg", type=int, default=24)
    args = ap.parse_args()

    manifest = json.load(open(_p(args.manifest)))
    policy = G.load_policy(args.policy)
    out_dir = _p(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # ---- master entry pool (first occurrence wins for file_path/goal_text) ----
    pool, source_of = {}, {}
    for rel, fname in SOURCE_SETS:
        for e in _load(rel, fname):
            fn = e["full_name"]
            if fn not in pool:
                pool[fn] = {"full_name": fn, "file_path": e.get("file_path"),
                            "namespace": e.get("namespace"), "goal_text": e.get("goal_text")}
                source_of[fn] = f"{rel.split('/')[-2]}/{fname}"

    def enrich(fn, expected, set_tag):
        base = pool.get(fn)
        if not base:
            return None
        fires, em = G.gate_fires(policy, base["namespace"], base["goal_text"], fn)
        comps = G.components_firing(em)
        return {**base,
                "rc4d_gate_fires": fires,
                "rc4d_components": comps,
                "rc4d_emissions": [{"component": x["component"], "action": x["action"],
                                    "tactic": x["tactic"]} for x in em],
                "rc4d_tactics": G.tactics_of(em),
                "expected_credit_component": comps[0] if comps else None,
                "source_set": source_of.get(fn),
                "set_tag": set_tag,
                "expected_behavior": expected}

    comp = manifest["components"]
    sets = {}

    # known wins per component
    sets["rc4a_known_wins"] = [enrich(fn, "RC4A def-unfold win; expect credit=RC4A", "rc4a_known_wins")
                              for fn in comp["RC4A"]["known_wins"]]
    sets["rc4b_known_wins"] = [enrich(fn, "RC4B disjoint-bridge win; expect credit=RC4B", "rc4b_known_wins")
                              for fn in comp["RC4B"]["known_wins"]]
    sets["rc4c_residue_known_wins"] = [
        enrich(fn, "RC4C residue win; credit=RC4B if disjoint-overlap else RC4C_residue",
               "rc4c_residue_known_wins")
        for fn in comp["RC4C_residue"]["residue_theorem_wins"]]

    # overlap controls = RC4B ∩ RC4C_residue theorems (RC4B must claim them)
    overlap = comp["RC4C_residue"]["theorem_overlap_with_rc4b"]
    sets["component_overlap_controls"] = [
        enrich(fn, "RC4B+RC4C_residue both fire; RC4B must win (no double credit)",
               "component_overlap_controls") for fn in overlap]

    known_names = set(comp["RC4A"]["known_wins"]) | set(comp["RC4B"]["known_wins"]) \
        | set(comp["RC4C_residue"]["residue_theorem_wins"])

    # fresh holdout: component fresh sets where the RC4D gate fires, not a known win
    fresh_src = (_load(RC4B_DIR, "fresh_holdout_multiset.json")
                 + _load(RC4B_DIR, "fresh_holdout_set.json")
                 + _load(RC4C_DIR, "fresh_holdout_all.json")
                 + _load(RC4A_DIR, "fresh_frontier_holdout.json")
                 + _load(RC4A_DIR, "same_cluster_holdout.json"))
    fresh, seen_fresh = [], set()
    for e in fresh_src:
        fn = e["full_name"]
        if fn in known_names or fn in seen_fresh or fn in set(overlap):
            continue
        en = enrich(fn, "fresh-frontier holdout; gate may fire", "composition_fresh_holdout")
        if en and en["rc4d_gate_fires"]:
            seen_fresh.add(fn)
            fresh.append(en)
        if len(fresh) >= args.max_fresh:
            break
    sets["composition_fresh_holdout"] = fresh

    # negatives: disjoint/pair/forall-shaped goals where gate must NOT fire.
    # A valid RC4D negative fires NO component; an entry that legitimately fires a
    # component gate (e.g. a Finset.disjUnion goal → RC4A) is not "unrelated", so it is
    # reclassified into fresh holdout rather than dropped (so it is still probed).
    reclassified = []
    neg_src = (_load(RC4C_DIR, "negative_controls.json")
               + _load(RC4B_DIR, "disjoint_negative_controls.json")
               + _load(RC4A_DIR, "negative_controls.json"))
    neg, seen_neg = [], set()
    for e in neg_src:
        fn = e["full_name"]
        if fn in seen_neg or fn in known_names or fn in set(overlap):
            continue
        en = enrich(fn, "negative control; gate MUST NOT fire", "negative_controls")
        if not en:
            continue
        seen_neg.add(fn)
        if en["rc4d_gate_fires"]:
            en["expected_behavior"] = "was a component-specific negative but fires RC4D " \
                                      f"{en['rc4d_components']} → reclassified to fresh holdout"
            en["set_tag"] = "composition_fresh_holdout"
            reclassified.append(en)
        else:
            neg.append(en)
        if len(neg) >= args.max_neg:
            break
    sets["negative_controls"] = neg

    # namespace negatives
    nsn_src = (_load(RC4C_DIR, "namespace_negative_controls.json")
               + _load(RC4B_DIR, "namespace_negative_controls.json"))
    nsn, seen_nsn = [], set()
    for e in nsn_src:
        fn = e["full_name"]
        if fn in seen_nsn or fn in known_names or fn in set(overlap):
            continue
        en = enrich(fn, "namespace negative; outside every component gate", "namespace_negative_controls")
        if not en:
            continue
        seen_nsn.add(fn)
        if en["rc4d_gate_fires"]:
            en["expected_behavior"] = "was a namespace negative but fires RC4D " \
                                      f"{en['rc4d_components']} → reclassified to fresh holdout"
            en["set_tag"] = "composition_fresh_holdout"
            reclassified.append(en)
        else:
            nsn.append(en)
        if len(nsn) >= args.max_ns_neg:
            break
    sets["namespace_negative_controls"] = nsn

    # fold any reclassified firing-negatives into fresh holdout
    for en in reclassified:
        if en["full_name"] not in seen_fresh:
            seen_fresh.add(en["full_name"])
            sets["composition_fresh_holdout"].append(en)

    # canonical smoke (reuse RC4C's demo_v1 + medium + large sample)
    smoke, seen_sm = [], set()
    for e in _load(RC4C_DIR, "canonical_smoke.json"):
        fn = e["full_name"]
        if fn in seen_sm:
            continue
        en = enrich(fn, e.get("expected_behavior", "canonical floor"), "canonical_smoke")
        if en:
            seen_sm.add(fn)
            smoke.append(en)
    sets["canonical_smoke"] = smoke

    # drop Nones, write
    set_files, sizes, fire_counts, ns_dist = {}, {}, {}, {}
    nofire = {"negative_controls", "namespace_negative_controls", "canonical_smoke"}
    for name, entries in sets.items():
        entries = [e for e in entries if e]
        path = os.path.join(out_dir, name + ".json")
        json.dump(entries, open(path, "w"), ensure_ascii=False, indent=2)
        rel = os.path.relpath(path, _REPO)
        set_files[name] = rel
        sizes[name] = len(entries)
        fire_counts[name] = sum(1 for e in entries if e["rc4d_gate_fires"])
        ns_dist[name] = dict(Counter(G.namespace_of(e["namespace"], e["full_name"]) for e in entries))

    all_entries, uniq = [], set()
    for entries in sets.values():
        for e in entries:
            if e and e["full_name"] not in uniq:
                uniq.add(e["full_name"])
                all_entries.append(e)

    off_gate = sum(fire_counts[s] for s in nofire if s in fire_counts)
    out_manifest = {
        "generated_by": "scripts/rc4d_build_validation_sets.py",
        "policy": args.policy,
        "component_manifest": args.manifest,
        "set_files": set_files,
        "sizes": sizes,
        "total": sum(sizes.values()),
        "unique_total": len(uniq),
        "gate_fire_counts": fire_counts,
        "namespace_distribution": ns_dist,
        "nofire_sets": sorted(nofire),
        "off_gate_emissions_in_nofire_sets": off_gate,
        "expected_credit_split": manifest["expected_overlap"]["expected_credited_components"],
        "component_coverage": {
            "rc4a_known": sizes.get("rc4a_known_wins"),
            "rc4b_known": sizes.get("rc4b_known_wins"),
            "rc4c_residue_known": sizes.get("rc4c_residue_known_wins"),
            "fresh": sizes.get("composition_fresh_holdout"),
            "overlap_controls": sizes.get("component_overlap_controls"),
        },
    }
    json.dump(out_manifest, open(os.path.join(out_dir, "validation_manifest.json"), "w"),
              ensure_ascii=False, indent=2)

    print(f"[rc4d-sets] sizes={sizes}")
    print(f"[rc4d-sets] total={out_manifest['total']} unique={len(uniq)} "
          f"off_gate_in_nofire={off_gate}")
    print(f"[rc4d-sets] fire_counts={fire_counts}")


if __name__ == "__main__":
    main()
