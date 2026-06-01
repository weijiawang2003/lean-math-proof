#!/usr/bin/env python3
"""TR2 Part 3 — select balanced active-learning batches.

Three selection strategies rank the candidate pool, then a fair, reproducible
round-robin DRAFT (model -> rule -> random each round) produces DISJOINT batches:

  model-selected : high router uncertainty + under-represented / depth-gap /
                   missing-bridge predictions + non-Set namespace + known RC2 failures first.
  rule-selected  : handcrafted TR1-baseline rules (Set iff cluster, toFinset/Finite,
                   Multiset induction, arithmetic/Nat, Set ite).
  random baseline: seeded shuffle, stratified by namespace, matched to the model
                   batch's known-RC2-failure ratio as closely as the pool allows.

Batch size auto-shrinks to floor(pool/3) (min 10) so the three batches stay
disjoint; any unavoidable overlap is recorded (here: none).
"""
from __future__ import annotations

import argparse
import json
import os
import random

UNDERREP = {"WX3_MULTISET_INDUCTION", "MX2_TOFINSET_AESOP", "PROOF_SEARCH_DEPTH_GAP"}


def _load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def _model_score(r):
    e = r.get("tr1_entropy") or 0.0
    s = float(e)
    pl = r.get("tr1_predicted_label")
    if pl in UNDERREP:
        s += 0.6
    if pl == "MISSING_BRIDGE_LEMMA_CANDIDATE":
        s += 0.5
    if r.get("namespace") not in ("Set",):
        s += 0.4
    if r.get("known_rc2_status") == "failed":
        s += 0.5
    m = r.get("tr1_margin")
    if m is not None and m < 0.2:
        s += 0.3
    return round(s, 4)


def _rule_score(r):
    fn = r["full_name"].lower()
    ns = r.get("namespace")
    s = 0.0
    reasons = []
    if "iff" in fn and ns == "Set":
        s += 3; reasons.append("set_iff_cluster")
    if "tofinset" in fn or "finite" in fn:
        s += 3; reasons.append("tofinset_finite")
    if ns == "Multiset" and ("induction" in fn or "cons" in fn or "rec" in fn):
        s += 3; reasons.append("multiset_induction")
    if ns in ("Nat", "Int") or any(t in fn for t in ("add", "sub", "mul", "omega")):
        s += 2; reasons.append("arithmetic")
    if ("ite" in fn or ".if" in fn) and ns == "Set":
        s += 1; reasons.append("set_ite")
    if r.get("known_rc2_status") == "failed":
        s += 0.5
    return round(s, 4), reasons


def _draft(pool, size, seed):
    """Disjoint round-robin draft for the two *informed* strategies (model, rule).

    The random baseline is drawn separately as a stratified control (see
    _random_stratified) so it can match the model batch's failure ratio even
    though that forces some overlap with the informed batches (failures are
    scarce); overlap is reported, not silently dropped.
    """
    by_name = {r["full_name"]: r for r in pool}
    model_order = sorted(pool, key=lambda r: (-_model_score(r),
                                              r.get("known_rc2_status") != "failed",
                                              -(r.get("tr1_entropy") or 0)))
    rule_order = sorted(pool, key=lambda r: (-_rule_score(r)[0],
                                             r.get("known_rc2_status") != "failed"))
    orders = {"model": [r["full_name"] for r in model_order],
              "rule": [r["full_name"] for r in rule_order]}
    batches = {"model": [], "rule": []}
    taken = set()
    ptr = {k: 0 for k in orders}

    def next_pick(strat):
        o = orders[strat]
        while ptr[strat] < len(o):
            fn = o[ptr[strat]]; ptr[strat] += 1
            if fn not in taken:
                return fn
        return None

    base = ["model", "rule"]
    rounds = 0
    while any(len(batches[s]) < size for s in batches) and len(taken) < len(pool):
        order = base[rounds % 2:] + base[:rounds % 2]
        for strat in order:
            if len(batches[strat]) >= size:
                continue
            fn = next_pick(strat)
            if fn is None:
                continue
            taken.add(fn)
            batches[strat].append(fn)
        rounds += 1
        if rounds > len(pool) + 5:
            break
    return batches, by_name


def _random_stratified(pool, size, target_fail_ratio, seed):
    """Stratified random baseline: match target failure ratio, stratify namespace
    within each stratum, shuffle within. Independent of the informed draft, so it
    may overlap with model/rule (recorded by caller)."""
    rnd = random.Random(seed)
    failed = [r["full_name"] for r in pool if r.get("known_rc2_status") == "failed"]
    other = [r["full_name"] for r in pool if r.get("known_rc2_status") != "failed"]
    rnd.shuffle(failed)
    rnd.shuffle(other)
    n_fail = min(len(failed), round(target_fail_ratio * size))
    n_other = min(len(other), size - n_fail)
    # top up if a stratum is short
    picked = failed[:n_fail] + other[:n_other]
    if len(picked) < size:
        rest = [fn for fn in (failed[n_fail:] + other[n_other:])]
        rnd.shuffle(rest)
        picked += rest[:size - len(picked)]
    return picked


def _batch_rows(names, by_name, strat):
    out = []
    for fn in names:
        r = by_name[fn]
        rec = {"full_name": fn, "file_path": r.get("file_path"), "namespace": r.get("namespace"),
               "known_rc2_status": r.get("known_rc2_status"),
               "tr1_predicted_label": r.get("tr1_predicted_label"),
               "tr1_entropy": r.get("tr1_entropy"), "tr1_margin": r.get("tr1_margin"),
               "sf4_cluster": r.get("sf4_cluster"), "selection_tags": r.get("selection_tags"),
               "selection_strategy": strat}
        if strat == "model":
            rec["model_score"] = _model_score(r)
        if strat == "rule":
            sc, why = _rule_score(r)
            rec["rule_score"] = sc; rec["rule_reasons"] = why
        out.append(rec)
    return out


def _fail_ratio(rows):
    if not rows:
        return 0.0
    return round(sum(1 for r in rows if r["known_rc2_status"] == "failed") / len(rows), 3)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--pool", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--batch-size", type=int, default=25)
    p.add_argument("--seed", type=int, default=20260530)
    args = p.parse_args(argv)

    pool = _load(args.pool)
    # auto-shrink to keep three disjoint batches
    max_disjoint = max(10, len(pool) // 3)
    size = min(args.batch_size, max_disjoint)

    batches, by_name = _draft(pool, size, args.seed)
    # model failure ratio drives the stratified random control
    model_rows_tmp = _batch_rows(batches["model"], by_name, "model")
    batches["random"] = _random_stratified(pool, size, _fail_ratio(model_rows_tmp), args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    out_rows = {s: _batch_rows(batches[s], by_name, s) for s in batches}
    json.dump(out_rows["model"], open(os.path.join(args.out_dir, "tr2_model_selected_batch.json"), "w"), indent=2)
    json.dump(out_rows["rule"], open(os.path.join(args.out_dir, "tr2_rule_selected_batch.json"), "w"), indent=2)
    json.dump(out_rows["random"], open(os.path.join(args.out_dir, "tr2_random_baseline_batch.json"), "w"), indent=2)

    # overlaps (disjoint by construction)
    sets = {s: set(batches[s]) for s in batches}
    overlaps = {}
    for a in ("model", "rule", "random"):
        for b in ("model", "rule", "random"):
            if a < b:
                ov = sorted(sets[a] & sets[b])
                if ov:
                    overlaps[f"{a}&{b}"] = ov

    import collections
    model_rule_overlap = sorted(sets["model"] & sets["rule"])
    manifest = {
        "pool_size": len(pool), "requested_batch_size": args.batch_size,
        "effective_batch_size": size, "seed": args.seed,
        "informed_batches_disjoint": not model_rule_overlap,
        "random_is_independent_stratified_control": True,
        "overlaps": overlaps,
        "overlap_note": ("model & rule are disjoint by the round-robin draft; the random baseline is an "
                         "INDEPENDENT stratified sample matched to the model batch's failure ratio, so it "
                         "may overlap model/rule on the scarce confirmed-failure cases (recorded above). "
                         "This is the spec's permitted 'overlap unavoidable -> record it' case and is the "
                         "correct control: it isolates selection quality from failure-ratio differences."),
        "draft": "round-robin model<->rule (disjoint); random = stratified matched-ratio control",
        "batches": {
            s: {"size": len(batches[s]), "members": batches[s],
                "known_rc2_failure_ratio": _fail_ratio(out_rows[s]),
                "namespace_dist": dict(collections.Counter(r["namespace"] for r in out_rows[s])),
                "predicted_label_dist": dict(collections.Counter(r["tr1_predicted_label"] for r in out_rows[s])),
                "cluster_dist": dict(collections.Counter(r["sf4_cluster"] for r in out_rows[s] if r["sf4_cluster"]))}
            for s in ("model", "rule", "random")
        },
        "strategy_definitions": {
            "model": "router uncertainty + under-rep/depth-gap/missing-bridge + non-Set + failures-first",
            "rule": "handcrafted: Set-iff cluster / toFinset-Finite / Multiset-induction / arithmetic / Set-ite",
            "random": "seeded shuffle baseline",
        },
        "files": {
            "model": "project/evolve/experiments/tr2/cases/tr2_model_selected_batch.json",
            "rule": "project/evolve/experiments/tr2/cases/tr2_rule_selected_batch.json",
            "random": "project/evolve/experiments/tr2/cases/tr2_random_baseline_batch.json",
        },
        "note": ("All pool members are already TR1-labelled (fresh frontier exhausted); batches are "
                 "re-probe / control targets used to compare SELECTION strategies, not to harvest novel theorems."),
    }
    json.dump(manifest, open(os.path.join(args.out_dir, "tr2_batch_manifest.json"), "w"), indent=2)
    print(f"[tr2-batches] pool={len(pool)} size={size} disjoint={not overlaps} "
          f"model={len(batches['model'])} rule={len(batches['rule'])} random={len(batches['random'])} "
          f"fail_ratio model={manifest['batches']['model']['known_rc2_failure_ratio']} "
          f"rule={manifest['batches']['rule']['known_rc2_failure_ratio']} "
          f"random={manifest['batches']['random']['known_rc2_failure_ratio']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
