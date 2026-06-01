#!/usr/bin/env python3
"""RC4R Part 3 — build the RC2-vs-RC4 benchmark theorem sets.

Sets:
  canonical_demo_v1 / canonical_nat_defs_medium / canonical_nat_defs_large_v5
      materialized from the registered theorem sets (full floors).
  rc4_known_wins             the 23 RC4D minimal-attribution credited wins.
  fresh_out_of_sample_frontier
      TR6 fresh-pool theorems NOT used in RC4D validation, balanced across namespaces;
      includes the RC4-gate-firing fresh (where a fresh delta can occur) plus balanced
      non-firing fresh (safety/coverage). Target 100-200.
  negative_controls          RC4D negative + namespace-negative controls (gate must not fire).
  offgate_controls           broad-gate traps: Finset.disjoint, Nat/order, non-disjoint Set,
                             non-matching List/Multiset (all must NOT fire any RC4 component).
  benchmark_manifest         set_files + sizes + overlap + fresh status + ns dist + expected emit.

The RC4 ordered-union gate (rc4d_gate) is re-derived per entry so each carries its firing
components and the expected credit component.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))
import rc4d_gate as G  # noqa: E402

RC4D = "project/evolve/experiments/rc4_candidates/composition_rc4d"
POLICY = RC4D + "/rc4d_composition_policy.json"
FRESH_POOL = "project/evolve/experiments/tr6/cases/tr6_fresh_frontier_pool.jsonl"
FLOOR_SETS = {"canonical_demo_v1": "demo_v1",
              "canonical_nat_defs_medium": "nat_defs_medium",
              "canonical_nat_defs_large_v5": "nat_defs_large_v5"}
# offgate trap shapes
OFFGATE_NS = ("Nat", "Int", "Order", "Bool", "Option")


def _p(*a):
    return os.path.join(_REPO, *a)


def _materialize_floor(setname):
    """Return [{full_name, file_path, namespace}] for a registered theorem set."""
    import dataclasses
    from core_types import TheoremConfig  # noqa
    import eval_rollout_all as E
    tcs = E.get_theorems(setname)
    out = []
    for t in tcs:
        fn = getattr(t, "full_name", None)
        fp = getattr(t, "file_path", None)
        if fn:
            out.append({"full_name": fn, "file_path": fp, "namespace": fn.split(".")[0]})
    return out


def _enrich(policy, e, set_tag, expected):
    fn = e["full_name"]
    goal = e.get("goal_text") or e.get("statement_text")
    fires, em = G.gate_fires(policy, e.get("namespace"), goal, fn)
    comps = G.components_firing(em)
    return {"full_name": fn, "file_path": e.get("file_path"),
            "namespace": G.namespace_of(e.get("namespace"), fn),
            "goal_text": goal,
            "rc4_gate_fires": fires, "rc4_components": comps,
            "rc4_tactics": G.tactics_of(em),
            "expected_credit_component": comps[0] if comps else None,
            "set_tag": set_tag, "expected_behavior": expected}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-fresh-firing", type=int, default=80)
    ap.add_argument("--max-fresh-nonfiring", type=int, default=45)
    ap.add_argument("--max-per-mono-ns", type=int, default=7)
    ap.add_argument("--max-offgate", type=int, default=28)
    args = ap.parse_args()

    policy = G.load_policy(POLICY)
    out_dir = _p(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # ---- RC4D-used names (out-of-sample exclusion) ----
    man = json.load(open(_p(RC4D, "theorem_sets/validation_manifest.json")))
    used = set()
    for s, rel in man["set_files"].items():
        for e in json.load(open(_p(rel))):
            used.add(e["full_name"])

    sets = {}

    # ---- canonical floors ----
    for tag, regname in FLOOR_SETS.items():
        entries = _materialize_floor(regname)
        sets[tag] = [_enrich(policy, e, tag, f"canonical floor {regname}") for e in entries]

    # ---- rc4_known_wins (RC4D credited) ----
    attr = json.load(open(_p(RC4D, "out/minimal_attribution.json")))
    credited = attr["credited_targets"]
    # file_paths from RC4D sets
    fp_map = {}
    for s, rel in man["set_files"].items():
        for e in json.load(open(_p(rel))):
            fp_map.setdefault(e["full_name"], e)
    comp_of = {r["full_name"]: r["winning_component"] for r in attr["records"] if r["credited"]}
    kw = []
    for fn in credited:
        e = fp_map.get(fn, {"full_name": fn})
        en = _enrich(policy, e, "rc4_known_wins",
                     f"RC4D credited win (component {comp_of.get(fn)})")
        en["rc4d_credited_component"] = comp_of.get(fn)
        kw.append(en)
    sets["rc4_known_wins"] = kw

    # ---- fresh out-of-sample frontier ----
    pool = [json.loads(l) for l in open(_p(FRESH_POOL))]
    mono_ns = {"Monotone", "MonotoneOn", "Antitone", "AntitoneOn",
               "StrictMono", "StrictMonoOn", "StrictAnti", "StrictAntiOn"}
    firing, nonfiring = [], []
    mono_count = Counter()
    nonfire_ns = Counter()
    for r in pool:
        fn = r["full_name"]
        if fn in used:
            continue
        en = _enrich(policy, r, "fresh_out_of_sample_frontier", "fresh out-of-sample")
        if not en["file_path"]:
            continue
        if en["rc4_gate_fires"]:
            ns = en["namespace"]
            # cap redundant mono def-unfold theorems for balance
            if ns in mono_ns:
                if mono_count[ns] >= args.max_per_mono_ns:
                    continue
                mono_count[ns] += 1
            firing.append(en)
        else:
            nonfiring.append((en, r))
    firing = firing[:args.max_fresh_firing]
    # balanced non-firing across namespaces
    nf_by_ns = defaultdict(list)
    for en, r in nonfiring:
        nf_by_ns[en["namespace"]].append(en)
    chosen_nf, idx = [], 0
    order = sorted(nf_by_ns, key=lambda k: -len(nf_by_ns[k]))
    while len(chosen_nf) < args.max_fresh_nonfiring and any(nf_by_ns.values()):
        progressed = False
        for ns in order:
            if nf_by_ns[ns]:
                chosen_nf.append(nf_by_ns[ns].pop(0)); progressed = True
                if len(chosen_nf) >= args.max_fresh_nonfiring:
                    break
        if not progressed:
            break
    sets["fresh_out_of_sample_frontier"] = firing + chosen_nf

    # ---- negative controls (reuse RC4D) ----
    neg_used = set()
    neg = []
    for s in ("negative_controls", "namespace_negative_controls"):
        rel = man["set_files"].get(s)
        if not rel:
            continue
        for e in json.load(open(_p(rel))):
            if e["full_name"] in neg_used:
                continue
            neg_used.add(e["full_name"])
            neg.append(_enrich(policy, e, "negative_controls", "gate must NOT fire"))
    sets["negative_controls"] = neg

    # ---- offgate controls (broad-gate traps) ----
    # RC4D negatives that are Finset.disjoint / Nat / order / non-disjoint Set / List/Multiset,
    # plus fresh non-firing of those shapes.
    traps, trap_used = [], set()
    for en in neg:
        if en["full_name"] in trap_used:
            continue
        trap_used.add(en["full_name"]); traps.append({**en, "set_tag": "offgate_controls",
                     "expected_behavior": "broad-gate trap; must not fire"})
    # add fresh non-firing Finset.disjoint + Nat/Int/Order shapes
    for r in pool:
        fn = r["full_name"]
        if fn in used or fn in trap_used:
            continue
        ns = (r.get("namespace") or fn.split(".")[0])
        blob = (fn + " " + (r.get("statement_text") or "")).lower()
        is_trap = ((ns == "Finset" and "disjoint" in blob)
                   or ns in OFFGATE_NS
                   or (ns == "Set" and "disjoint" not in blob and "subset_pair" not in blob
                       and "monotone" not in blob and "antitone" not in blob)
                   or (ns in ("List", "Multiset") and "disjoint" not in blob and "forall" not in blob))
        if not is_trap or not r.get("file_path"):
            continue
        en = _enrich(policy, r, "offgate_controls", "broad-gate trap (fresh); must not fire")
        if en["rc4_gate_fires"]:
            continue  # only keep genuine non-firing traps
        trap_used.add(fn); traps.append(en)
        if len(traps) >= args.max_offgate:
            break
    sets["offgate_controls"] = traps

    # ---- write + manifest ----
    set_files, sizes, fire_counts, ns_dist, overlap_rc4d, fresh_counts = {}, {}, {}, {}, {}, {}
    nofire = {"negative_controls", "offgate_controls"}
    for name, entries in sets.items():
        path = os.path.join(out_dir, name + ".json")
        json.dump(entries, open(path, "w"), ensure_ascii=False, indent=2)
        set_files[name] = os.path.relpath(path, _REPO)
        sizes[name] = len(entries)
        fire_counts[name] = sum(1 for e in entries if e["rc4_gate_fires"])
        ns_dist[name] = dict(Counter(e["namespace"] for e in entries))
        overlap_rc4d[name] = sum(1 for e in entries if e["full_name"] in used)
        fresh_counts[name] = sum(1 for e in entries if e["full_name"] not in used)

    uniq = set()
    for entries in sets.values():
        for e in entries:
            uniq.add(e["full_name"])
    off_in_nofire = sum(fire_counts[s] for s in nofire if s in fire_counts)

    out_manifest = {
        "generated_by": "scripts/rc4r_build_benchmark_sets.py",
        "policy": POLICY, "rc4d_used_excluded": len(used),
        "set_files": set_files, "sizes": sizes, "total": sum(sizes.values()),
        "unique_total": len(uniq), "gate_fire_counts": fire_counts,
        "namespace_distribution": ns_dist, "overlap_with_rc4d": overlap_rc4d,
        "fresh_counts": fresh_counts, "nofire_sets": sorted(nofire),
        "off_gate_emissions_in_nofire_sets": off_in_nofire,
        "fresh_frontier_component_split": dict(Counter(
            c for e in sets["fresh_out_of_sample_frontier"] for c in e["rc4_components"])),
    }
    json.dump(out_manifest, open(os.path.join(out_dir, "benchmark_manifest.json"), "w"),
              ensure_ascii=False, indent=2)
    print(f"[rc4r-sets] sizes={sizes}")
    print(f"[rc4r-sets] total={out_manifest['total']} unique={len(uniq)} "
          f"off_gate_in_nofire={off_in_nofire}")
    print(f"[rc4r-sets] fresh frontier: {sizes['fresh_out_of_sample_frontier']} "
          f"(firing {fire_counts['fresh_out_of_sample_frontier']}), "
          f"ns={ns_dist['fresh_out_of_sample_frontier']}")
    print(f"[rc4r-sets] fresh frontier component split={out_manifest['fresh_frontier_component_split']}")


if __name__ == "__main__":
    main()
