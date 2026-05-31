#!/usr/bin/env python3
"""TR3 Part 2 — build the retrieval-aware depth-search case pool.

Case types:
  A  SF5 proof-depth targets
  B  SF5 existing-lemma / retrieval-routing targets
  C  confirmed SF4 RC2 failures not covered by SF5
  D  fresh SF1 frontier cases (with file_path)
  E  multi-namespace expansion sampled from discovered_theorems.json
     (Set / Finset / Multiset / Nat / List / Order / Algebra if present)

Deduped by full_name. Cases without a file_path go to the unresolved sidecar.
Goal text + features are backfilled from the local traced Mathlib source where
available (same approach as SF5). known_rc2_status comes from the SF4/TR2
identical-config confirmations (failed/solved) else unknown.
"""
from __future__ import annotations

import argparse
import json
import os
import re

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRACED_ROOT = os.path.expanduser(
    "~/.cache/lean_dojo/leanprover-community-mathlib4-"
    "29dcec074de168ac2bf835a77ef68bbe069194c5/mathlib4")

EXPANSION_NAMESPACES = ("Set", "Finset", "Multiset", "Nat", "List", "Order", "Algebra")


def _p(*a):
    return os.path.join(_REPO, *a)


def _load_json(path):
    fp = _p(path)
    return json.load(open(fp)) if os.path.exists(fp) else None


def _load_jsonl(path):
    fp = _p(path)
    if not os.path.exists(fp):
        return []
    return [json.loads(l) for l in open(fp) if l.strip()]


def _features(text):
    t = text or ""
    low = t.lower()
    return {
        "has_iff": ("↔" in t) or ("iff" in low),
        "has_subset": ("⊆" in t) or ("subset" in low),
        "has_ssubset": ("⊂" in t) or ("ssubset" in low),
        "has_eq": (" = " in t),
        "has_monotone": "monotone" in low,
        "has_strictmono": ("strictmono" in low) or ("strict_mono" in low),
        "has_set": ("set" in low) or ("∈" in t) or ("∪" in t) or ("∩" in t) or ("⊆" in t),
        "has_singleton": ("singleton" in low) or ("{" in t),
        "has_insert": "insert" in low,
        "has_compl": ("compl" in low) or ("ᶜ" in t),
        "has_pair": "pair" in low,
        "has_ite": ("ite" in low) or (" if " in low),
        "has_union": ("union" in low) or ("∪" in t),
        "has_empty": ("empty" in low) or ("∅" in t),
        "has_nat": ("nat" in low) or ("ℕ" in t),
        "has_tofinset": "tofinset" in low,
        "has_arith": any(s in t for s in ("+", "*", "≤", "<", "-")) and ("ℕ" in t or "nat" in low),
    }


def _statement_from_source(file_path, full_name, root=_TRACED_ROOT):
    if not file_path or not root or not os.path.isdir(root):
        return None
    fp = os.path.join(root, file_path)
    if not os.path.exists(fp):
        return None
    short = full_name.split(".")[-1]
    pat = re.compile(r"^\s*(?:protected\s+|@\[[^\]]*\]\s*)*(?:theorem|lemma|def)\s+"
                     + re.escape(short) + r"\b")
    try:
        lines = open(fp, encoding="utf-8", errors="replace").read().splitlines()
    except OSError:
        return None
    for i, ln in enumerate(lines):
        if pat.match(ln):
            buf = []
            for j in range(i, min(i + 14, len(lines))):
                buf.append(lines[j])
                if ":=" in lines[j]:
                    break
            text = " ".join(s.strip() for s in buf)
            idx = text.find(":=")
            if idx != -1:
                text = text[:idx]
            return re.sub(r"\s+", " ", text).strip()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-pool", required=True)
    ap.add_argument("--out-summary-json", required=True)
    ap.add_argument("--out-summary-md", required=True)
    ap.add_argument("--out-unresolved",
                    default="project/evolve/experiments/tr3/cases/tr3_case_pool_unresolved.jsonl")
    ap.add_argument("--max-cases", type=int, default=150)
    args = ap.parse_args()

    # ---- load sources ----
    sf5_targets = _load_json("project/evolve/experiments/sf5/cases/sf5_missing_bridge_targets.json") or []
    sf5_attr = _load_json("project/evolve/experiments/sf5/out/sf5_retrieval_attribution.json") or {}
    sf5_label = {r["full_name"]: r["classification"] for r in sf5_attr.get("records", [])}
    sf4_conf = _load_json("project/evolve/experiments/sf4/out/rc2_failure_confirmation.json") or {}
    sf4_results = {r["full_name"]: r for r in sf4_conf.get("results", [])}
    tr2_conf = _load_json("project/evolve/experiments/tr2/out/tr2_rc2_confirmation.json") or {}
    tr2_results = {r["full_name"]: r for r in tr2_conf.get("results", [])}
    frontier = _load_jsonl("project/evolve/experiments/sf1/out/real/frontier_with_paths.jsonl")
    discovered = (_load_json("project/discovered_theorems.json") or {}).get("theorems", [])

    # rc2 status map from identical-config confirmations
    def rc2_status(fn):
        for src in (sf4_results, tr2_results):
            r = src.get(fn)
            if r:
                c = r.get("classification")
                if c == "CONFIRMED_RC2_FAILURE":
                    return "failed"
                if c in ("RC2_SOLVED", "NOW_SOLVED_BY_RC2"):
                    return "solved"
        return "unknown"

    pool = {}      # full_name -> case
    unresolved = []

    def add(fn, file_path, namespace, source, case_type, priority,
            cluster_id=None, sf5lab=None):
        if fn in pool:
            # keep highest priority / earliest type
            if priority > pool[fn]["priority"]:
                pool[fn]["priority"] = priority
            pool[fn].setdefault("sources", [])
            if source not in pool[fn]["sources"]:
                pool[fn]["sources"].append(source)
            return
        if not file_path:
            unresolved.append({"full_name": fn, "namespace": namespace,
                               "source": source, "reason": "no file_path"})
            return
        stmt = _statement_from_source(file_path, fn)
        case = {
            "full_name": fn,
            "file_path": file_path,
            "namespace": namespace or (fn.split(".")[0] if "." in fn else ""),
            "source": source,
            "sources": [source],
            "case_type": case_type,
            "known_rc2_status": rc2_status(fn),
            "cluster_id": cluster_id,
            "sf5_label": sf5lab,
            "goal_text": stmt,
            "last_goal": stmt,
            "features": _features(stmt or fn),
            "priority": priority,
        }
        pool[fn] = case

    # A + B: SF5 targets
    for t in sf5_targets:
        fn = t["full_name"]
        lab = sf5_label.get(fn)
        ctype = "B" if lab in ("EXISTING_LEMMA_GAP", "RETRIEVAL_ROUTING_GAP") else "A"
        add(fn, t.get("file_path"), t.get("namespace"), "sf5", ctype,
            priority=3, cluster_id=t.get("cluster_id"), sf5lab=lab)

    # C: confirmed SF4 RC2 failures not in SF5
    for fn, r in sf4_results.items():
        if r.get("classification") == "CONFIRMED_RC2_FAILURE" and fn not in pool:
            add(fn, r.get("file_path"), r.get("namespace"), "sf4_confirmed_failure",
                "C", priority=3)

    # D: fresh SF1 frontier
    for row in frontier:
        fn = row.get("name") or row.get("full_name") or row.get("decl_name")
        if not fn:
            continue
        add(fn, row.get("file_path"), row.get("namespace"), "sf1_frontier", "D",
            priority=2)

    # E: multi-namespace expansion from discovered_theorems
    # prioritise namespaces under-represented in the confirmed-failure pool; take a
    # deterministic slice per namespace up to the budget.
    by_ns = {}
    for th in discovered:
        ns0 = th["full_name"].split(".")[0]
        if ns0 in EXPANSION_NAMESPACES:
            by_ns.setdefault(ns0, []).append(th)
    # round-robin across namespaces for breadth, hardest-first within a namespace
    diff_rank = {"hard": 0, "medium": 1, "easy": 2}
    for ns in by_ns:
        by_ns[ns].sort(key=lambda t: (diff_rank.get(t.get("difficulty"), 3), t["full_name"]))
    order = sorted(EXPANSION_NAMESPACES, key=lambda n: -len(by_ns.get(n, [])))
    idx = {n: 0 for n in order}
    budget = max(0, args.max_cases - len(pool))
    added_e = 0
    while added_e < budget and any(idx[n] < len(by_ns.get(n, [])) for n in order):
        for ns in order:
            lst = by_ns.get(ns, [])
            if idx[ns] >= len(lst):
                continue
            th = lst[idx[ns]]
            idx[ns] += 1
            before = len(pool)
            add(th["full_name"], th.get("file_path"), ns, "discovered_expansion", "E",
                priority=1)
            if len(pool) > before:
                added_e += 1
            if added_e >= budget:
                break

    cases = sorted(pool.values(), key=lambda c: (-c["priority"], c["case_type"], c["full_name"]))

    os.makedirs(os.path.dirname(_p(args.out_pool)), exist_ok=True)
    with open(_p(args.out_pool), "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(_p(args.out_unresolved), "w", encoding="utf-8") as f:
        for u in unresolved:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")

    from collections import Counter
    type_hist = Counter(c["case_type"] for c in cases)
    ns_hist = Counter(c["namespace"] for c in cases)
    rc2_hist = Counter(c["known_rc2_status"] for c in cases)
    summary = {
        "generated_by": "scripts/tr3_build_case_pool.py",
        "num_cases": len(cases),
        "num_unresolved": len(unresolved),
        "case_type_histogram": dict(type_hist),
        "namespace_histogram": dict(ns_hist.most_common()),
        "known_rc2_status_histogram": dict(rc2_hist),
        "target_size": args.max_cases,
        "frontier_exhaustion_note": (
            "Confirmed-failure case types A/B/C are bounded by the SF4/SF5 pool; "
            "fresh signal comes only from D (SF1 frontier) and E (discovered expansion). "
            "Most E cases are 'easy' and expected to be RC2_SOLVED — confirmed failures "
            "remain the scarce, valuable subset."),
        "sources": {
            "sf5_targets": len(sf5_targets),
            "sf4_confirmed_failures": sum(1 for r in sf4_results.values()
                                          if r.get("classification") == "CONFIRMED_RC2_FAILURE"),
            "sf1_frontier": len(frontier),
            "discovered_theorems": len(discovered),
            "expansion_added": added_e,
        },
    }
    json.dump(summary, open(_p(args.out_summary_json), "w"), ensure_ascii=False, indent=2)

    md = ["# TR3 case pool", "",
          f"- cases: **{len(cases)}** (target ≤{args.max_cases})",
          f"- unresolved (no file_path): {len(unresolved)}",
          f"- case types: {dict(type_hist)}",
          f"- known RC2 status: {dict(rc2_hist)}", "",
          "## Namespaces", ""]
    for ns, c in ns_hist.most_common():
        md.append(f"- {ns}: {c}")
    md += ["", "## Note", "", summary["frontier_exhaustion_note"]]
    open(_p(args.out_summary_md), "w").write("\n".join(md) + "\n")

    print(f"[tr3-pool] {len(cases)} cases, types={dict(type_hist)}, rc2={dict(rc2_hist)}, "
          f"unresolved={len(unresolved)}")


if __name__ == "__main__":
    main()
