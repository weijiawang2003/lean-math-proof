#!/usr/bin/env python3
"""TR5 Part 2 — build the live-search target pool.

Aggregates targets from TR4 (active queue, ranking eval, budget sim, program rows),
TR3 (case pool, rc2 confirmation, retrieval, depth plan, attribution), and RC4A
(candidate results, minimal attribution), tagging each with category A–G, known RC2 /
TR3 / RC4A outcomes, the TR4 ranker score, and candidate-family tags. Deduped by
full_name. No live Lean here.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, OrderedDict

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*a):
    return os.path.join(_REPO, *a)


def _load(path):
    fp = _p(path)
    return json.load(open(fp)) if os.path.exists(fp) else None


def _rows(path):
    fp = _p(path)
    return [json.loads(l) for l in open(fp) if l.strip()] if os.path.exists(fp) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-pool", required=True)
    ap.add_argument("--out-summary-json", required=True)
    ap.add_argument("--out-summary-md", required=True)
    args = ap.parse_args()

    # ---------- load sources ----------
    tr3_attr = {r["full_name"]: r for r in
                (_load("project/evolve/experiments/tr3/out/tr3_attribution.json") or {}).get("records", [])}
    tr3_res = {r["full_name"]: r for r in
               (_load("project/evolve/experiments/tr3/out/tr3_depth_program_results.json") or {}).get("results", [])}
    tr3_plan = {t["full_name"]: t for t in
                (_load("project/evolve/experiments/tr3/out/tr3_depth_program_plan.json") or {}).get("theorems", [])}
    tr3_conf = {r["full_name"]: r for r in
                (_load("project/evolve/experiments/tr3/out/tr3_rc2_confirmation.json") or {}).get("results", [])}
    rc4_res = {r["full_name"]: r for r in
               (_load("project/evolve/experiments/rc4_candidates/def_unfold_simp/out/candidate_results.json") or {}).get("results", [])}
    rc4_attr = {r["full_name"]: r for r in
                (_load("project/evolve/experiments/rc4_candidates/def_unfold_simp/out/minimal_attribution.json") or {}).get("records", [])}
    queue = {q["full_name"]: q for q in
             (_load("project/evolve/experiments/tr4/out/tr4_active_probe_queue.json") or {}).get("queue", [])}
    tr4_rows = _rows("project/evolve/experiments/tr4/data/tr4_program_examples.jsonl")
    # best (max) ranker score per theorem from the active queue (leakage-free OOF HGB)
    tr4_score = {fn: q.get("expected_value") for fn, q in queue.items()}
    # theorem -> namespace / file_path (prefer plan, then tr3 rows, then tr4 rows)
    meta = {}
    for fn, t in tr3_plan.items():
        meta[fn] = {"namespace": t.get("namespace"), "file_path": t.get("file_path"),
                    "goal_text": t.get("goal_text"), "cluster_id": t.get("cluster_id")}
    for r in tr4_rows:
        meta.setdefault(r["full_name"], {})
        m = meta[r["full_name"]]
        m.setdefault("namespace", r.get("namespace"))
        m.setdefault("file_path", r.get("file_path"))
        m.setdefault("goal_text", r.get("goal_text"))
        m.setdefault("cluster_id", r.get("cluster_id"))

    # TR3 live wins / true-deltas
    tr3_true_delta = set((_load("project/evolve/experiments/tr3/out/tr3_attribution.json") or {}).get("true_delta_targets", []))
    tr3_winners = {fn: r["best_win"] for fn, r in tr3_res.items() if r.get("best_win")}
    rc4_winners = set((_load("project/evolve/experiments/rc4_candidates/def_unfold_simp/out/minimal_attribution.json") or {}).get("true_def_unfold_win_targets", []))

    def disjoint_left_win(fn):
        bw = tr3_winners.get(fn)
        return bool(bw and any("disjoint_left" in (L or "") for L in bw.get("lemmas", [])))

    def d2_aesop_win(fn):
        bw = tr3_winners.get(fn)
        return bool(bw and bw.get("family") == "d2_simp_aesop")

    # ---------- assemble candidate set ----------
    pool = OrderedDict()  # full_name -> row

    def cat_tags(fn):
        tags = []
        if disjoint_left_win(fn):
            tags.append("rc4b_set_disjoint_left")
        if d2_aesop_win(fn):
            tags.append("rc4c_d2_simp_aesop")
        if fn in rc4_winners:
            tags.append("rc4a_def_unfold")
        # from queue recommended top programs
        q = queue.get(fn)
        if q:
            for rp in q.get("recommended_programs", [])[:3]:
                if rp.get("used_lemma") and "disjoint_left" in rp["used_lemma"]:
                    tags.append("rc4b_set_disjoint_left")
                if rp.get("family") == "d2_simp_aesop":
                    tags.append("rc4c_d2_simp_aesop")
        return sorted(set(tags))

    def add(fn, category, priority):
        m = meta.get(fn, {})
        ns = m.get("namespace") or (fn.split(".")[0] if "." in fn else "")
        # known rc2 status
        if fn in tr3_attr:
            rc2 = "failed"  # all TR3 program theorems are confirmed RC2 failures
        elif fn in rc4_res:
            rc2 = "failed" if not rc4_res[fn].get("rc2_finished") else "solved"
        else:
            rc2 = "unknown"
        tr3_out = tr3_attr.get(fn, {}).get("classification")
        if fn in tr3_winners:
            tr3_out = f"WIN:{tr3_winners[fn]['family']}"
        rc4_out = None
        if fn in rc4_attr:
            rc4_out = rc4_attr[fn].get("classification")
        if fn in pool:
            # keep earliest (highest-priority) category but merge tags
            row = pool[fn]
            row["candidate_family_tags"] = sorted(set(row["candidate_family_tags"]) | set(cat_tags(fn)))
            row["priority"] = max(row["priority"], priority)
            return
        pool[fn] = {
            "full_name": fn, "file_path": m.get("file_path"), "namespace": ns,
            "target_category": category,
            "known_rc2_status": rc2,
            "known_tr3_outcome": tr3_out,
            "known_rc4a_outcome": rc4_out,
            "tr4_ranker_score": tr4_score.get(fn),
            "candidate_family_tags": cat_tags(fn),
            "priority": priority,
        }

    # A. known TR3 winners (live-rerun sanity) — highest priority
    for fn in tr3_winners:
        add(fn, "A_tr3_winner", 100)
    # B. RC4A known wins
    for fn in rc4_winners:
        add(fn, "B_rc4a_win", 95)
    # C. RC4B support: Set.disjoint_left targets (wins + similar unresolved)
    for fn in tr3_plan:
        ns = meta.get(fn, {}).get("namespace") or ""
        if disjoint_left_win(fn):
            add(fn, "C_rc4b_disjoint", 90)
    for fn, q in queue.items():
        if any("disjoint_left" in (rp.get("used_lemma") or "") for rp in q.get("recommended_programs", [])[:5]):
            add(fn, "C_rc4b_disjoint", 80)
    # D. RC4C support: d2_simp_aesop targets (wins + high-score candidates)
    for fn in tr3_plan:
        if d2_aesop_win(fn):
            add(fn, "D_rc4c_d2aesop", 90)
    for fn, q in queue.items():
        if any(rp.get("family") == "d2_simp_aesop" for rp in q.get("recommended_programs", [])[:3]):
            add(fn, "D_rc4c_d2aesop", 78)
    # E. high-confidence TR4 predictions not yet live under ranker ordering (open theorems, high EV)
    for fn, q in queue.items():
        if (not q.get("had_win")) and (q.get("expected_value") or 0) >= 0.3:
            add(fn, "E_high_confidence", 70)
    # F. high-uncertainty / active-learning
    for fn, q in queue.items():
        if q.get("selection_reason") == "high_uncertainty":
            add(fn, "F_high_uncertainty", 50)
    # G. underrepresented namespaces (Finset/List/Nat/Multiset/Order)
    UNDER = {"Finset", "List", "Nat", "Multiset", "Order"}
    for fn, q in queue.items():
        ns = q.get("namespace")
        if ns in UNDER:
            add(fn, "G_underrep_namespace", 45)
    # backfill: any remaining TR3-plan / queue theorem as E/G to reach size
    for fn in tr3_plan:
        ns = meta.get(fn, {}).get("namespace") or ""
        cat = "G_underrep_namespace" if ns in UNDER else "E_high_confidence"
        add(fn, cat, 40)

    rows = list(pool.values())
    rows.sort(key=lambda r: (-r["priority"], r["full_name"]))

    os.makedirs(os.path.dirname(_p(args.out_pool)), exist_ok=True)
    with open(_p(args.out_pool), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_cat = Counter(r["target_category"] for r in rows)
    by_ns = Counter(r["namespace"] for r in rows)
    by_rc2 = Counter(r["known_rc2_status"] for r in rows)
    n_known_win = sum(1 for r in rows if (r["known_tr3_outcome"] or "").startswith("WIN") or r["target_category"] == "B_rc4a_win")
    rc4b = [r["full_name"] for r in rows if "rc4b_set_disjoint_left" in r["candidate_family_tags"]]
    rc4c = [r["full_name"] for r in rows if "rc4c_d2_simp_aesop" in r["candidate_family_tags"]]
    summary = {
        "generated_by": "scripts/tr5_build_target_pool.py",
        "num_targets": len(rows),
        "by_category": dict(by_cat), "by_namespace": dict(by_ns),
        "by_known_rc2_status": dict(by_rc2),
        "num_known_winners": n_known_win,
        "rc4b_disjoint_left_targets": rc4b,
        "rc4c_d2_simp_aesop_targets": rc4c,
        "num_with_ranker_score": sum(1 for r in rows if r["tr4_ranker_score"] is not None),
    }
    json.dump(summary, open(_p(args.out_summary_json), "w"), ensure_ascii=False, indent=2)

    md = ["# TR5 target pool", "",
          f"- **{len(rows)} targets** (deduped by full_name)",
          f"- by category: {dict(by_cat)}",
          f"- by namespace: {dict(by_ns)}",
          f"- by known RC2 status: {dict(by_rc2)}",
          f"- known winners (TR3/RC4A): {n_known_win}",
          f"- RC4B (Set.disjoint_left) targets: {len(rc4b)} → {rc4b}",
          f"- RC4C (d2_simp_aesop) targets: {len(rc4c)} → {rc4c}", "",
          "## Top 20 by priority", "",
          "| full_name | ns | category | rc2 | tr3 | ranker | tags |",
          "|---|---|---|---|---|---|---|"]
    for r in rows[:20]:
        md.append(f"| `{r['full_name']}` | {r['namespace']} | {r['target_category']} | "
                  f"{r['known_rc2_status']} | {r['known_tr3_outcome']} | "
                  f"{r['tr4_ranker_score']} | {','.join(r['candidate_family_tags'])} |")
    open(_p(args.out_summary_md), "w").write("\n".join(md) + "\n")
    print(f"[tr5-pool] {len(rows)} targets; cats={dict(by_cat)}; ns={dict(by_ns)}")
    print(f"  rc4b={len(rc4b)} rc4c={len(rc4c)} known_winners={n_known_win}")


if __name__ == "__main__":
    main()
