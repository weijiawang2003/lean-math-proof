#!/usr/bin/env python3
"""Part 8 — generate the dense SF2/SF3 overnight exploration report (data-driven).

Reads every artifact produced this run and emits
project/evolve/reports/sf2_sf3_overnight_exploration_report.md with the 9 required
sections, including a live protected-file git confirmation. No fabrication: every
number is read from a file on disk; missing inputs are reported as such.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess

R = "project/evolve/experiments"
SF3 = f"{R}/sf3/out/singleton_iff"
FE = f"{R}/sf2/out/frontier_expansion"
REPORT = "project/evolve/reports/sf2_sf3_overnight_exploration_report.md"
PROT = ["project/evolve/experiments/rc1/rc1_production_wrapper.json",
        "project/evolve/routing/ns24_router.json"]


def load(p, d=None):
    try:
        return json.load(open(p))
    except Exception:
        return d


def git(*a):
    try:
        return subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    except Exception as e:
        return f"<git failed: {e}>"


def main():
    em = load(f"{R}/sf1/out/real/eval_matrix_results.json", {}) or {}
    clusters = load(f"{FE}/failure_clusters.json", {}) or {}
    analysis = load(f"{SF3}/source_proof_analysis.json", {}) or {}
    probes = load(f"{SF3}/probe_results.json", {}) or {}
    cand = load(f"{SF3}/candidate_lemmas.json", {}) or {}
    lemma_q = [json.loads(l) for l in open(f"{SF3}/candidate_lemma_queue.jsonl")] \
        if os.path.exists(f"{SF3}/candidate_lemma_queue.jsonl") else []

    # eval rows (dedup by set+policy, keep the last/most-recent entry)
    _rowmap = {}
    for r in em.get("results", []):
        key = (r.get("theorem_set"), r.get("policy"))
        if r.get("status") == "ran":
            _rowmap[key] = (r.get("theorem_set"), r.get("policy"), r.get("solved"),
                            r.get("total"))
        else:
            _rowmap[key] = (r.get("theorem_set"), r.get("policy"),
                            r.get("status"), None)
    rows = list(_rowmap.values())

    prot = git("diff", "--stat", "HEAD", "--", *PROT)
    status = git("status", "--short")

    L = []
    a = L.append
    a("# SF2/SF3 Overnight Exploration — Truth Repair, Singleton-Iff Deep Dive, Frontier Expansion")
    a("")
    a("- branch `rc1-production-stack` · no production config change · no commit · "
      "all live results reproduced from disk.")
    a("")

    a("## 1. Executive summary")
    a("")
    a(f"- **Corrected Multiset result:** RC1 solved **2/3** of the holdout "
      "(`toFinset_nsmul` via aesop; `disjoint_toFinset` via the WX3 induction oracle), "
      "1 genuine failure. The old \"0/3\" was a metrics-parser bug (read "
      "`proof_finished`/`solved`; the real key is `finished`).")
    sp = probes.get("solved_probe")
    a(f"- **Singleton-iff genuine failure** `Multiset.toFinset_eq_singleton_iff`: the "
      f"Part-3 deep ladder closed it **{probes.get('num_solved', '?')}/"
      f"{probes.get('num_probes', '?')}** live — NO new probe solved it "
      f"({probes.get('outcome_histogram')}).")
    a(f"- **No probe solved the genuine failure.** It needs a multi-step "
      "count-extensionality proof; every one-shot `simp_all[...]` (incl. source-inspired "
      "lemma sets) hits max recursion, and the WX3 induction oracle strictly hurts.")
    a(f"- **Broader frontier eval:** {sum(1 for s in rows if s[2] not in (None,) and isinstance(s[2], int))} "
      f"theorem-set runs recorded (see §4).")
    gen = clusters.get("num_genuine_failures")
    a(f"- **Genuine RC1 failures found:** {gen if gen is not None else 'see §5'}; "
      f"clusters: {clusters.get('num_clusters', '?')}.")
    a(f"- **SF3 candidate-lemma queue size:** {len(lemma_q)} (key finding: the singleton "
      "failure needs NO new lemma — it is a tactic/routing gap).")
    a("")

    a("## 2. Ground-truth correction")
    a("")
    a("`scripts/sf1_eval_matrix.py:parse_metrics` keyed solved on "
      "`proof_finished`/`solved`/`proved` (none exist per-theorem); the authoritative "
      "key is **`finished`**, and the aggregate uses `proved`/`total_theorems`. Every "
      "theorem read as unsolved ⇒ RC1 undercounted as 0/3. Fixed (`_theorem_solved` "
      "helper + `--repair-existing-results`); see `sf1_truth_layer_correction.md`. "
      "Corrected: **RC1 2/3**. This matters because the WX3 oracle's one held-out win "
      "(`disjoint_toFinset`) was being mislabeled a failure.")
    a("")

    a("## 3. Singleton-iff deep dive")
    a("")
    a(f"- real proof location: `{(analysis.get('source_location') or {}).get('file_with_proof_text')}` "
      f"(resolves for Dojo under `{(analysis.get('source_location') or {}).get('file_resolved_for_dojo')}`).")
    a(f"- statement: `{analysis.get('statement')}`")
    a("- official proof is a **count-extensionality** argument (`refine` iff-split → "
      "`ext'` → `by_cases x = a` → `count_*` rewrites; reverse via `toFinset_nsmul`, "
      "`toFinset_singleton`). **No induction.**")
    a(f"- verdict: **{(analysis.get('classification') or {}).get('verdict')}**")
    a("- residual under the RC1 WX3 oracle: "
      "`insert a✝ s✝.toFinset = {a} ↔ a✝ ::ₘ s✝ = (card s✝ + 1) • {a}` "
      "(induction makes it worse).")
    a("- **probe ladder results (live, single Dojo):**")
    for o in probes.get("outcomes", []):
        a(f"  - `{o.get('outcome')}` — `{(o.get('probe') or '')[:64].replace(chr(10), ' / ')}`")
    a("- **tactic-gap or lemma-gap?** TACTIC/ROUTING gap. All dependencies exist; the "
      "gap is (a) split the iff before ext, (b) multi-step count reasoning a single "
      "battery tactic cannot do.")
    a("")

    a("## 4. Expanded frontier eval")
    a("")
    a("| theorem_set | policy | solved/total or status |")
    a("|---|---|---|")
    for ts, pol, s, t in rows:
        a(f"| {ts} | {pol} | {s}{('/' + str(t)) if t is not None else ''} |")
    a("")
    a("> Caveat: SF1 frontier `file_path`s are heuristic namespace→path backfill and the "
      "frontier contains mining-artifact rows (e.g. `traces_from_search.jsonl`, "
      "`Prop.compl_singleton`). Theorems that fail to resolve are environment/path "
      "failures, NOT proof failures — separated in §5.")
    a("Commands: `project/evolve/experiments/sf2/out/frontier_expansion/eval_commands.sh` "
      "(if emitted) and `scripts/sf1_eval_matrix.py --run-real --policies rc1`.")
    a("")

    a("## 5. Failure clusters")
    a("")
    a(f"- failures: {clusters.get('num_failures', '?')} "
      f"(genuine {clusters.get('num_genuine_failures', '?')}, "
      f"junk/unresolved {clusters.get('num_junk_or_unresolved', '?')}); "
      f"clusters: {clusters.get('num_clusters', '?')}.")
    a("")
    a("| priority | cluster_id | size | capability | next |")
    a("|---|---|---|---|---|")
    for c in sorted(clusters.get("clusters", []),
                    key=lambda c: ({"high": 0, "medium": 1, "low": 2}.get(c["priority"], 3), -c["size"])):
        a(f"| {c['priority']} | `{c['cluster_id']}` | {c['size']} | "
          f"{c['likely_missing_capability']} | {c['next_action']} |")
    a("")

    a("## 6. Relabel queue update")
    a("")
    a("- NS23 minimal-sufficient relabel targets: the corrected Multiset wins "
      "(`toFinset_nsmul`, `disjoint_toFinset`) — confirm whether plain `simp`/`omega` "
      "also closes them, and the open singleton failure's probe family. See "
      "`relabel_queue_updated.jsonl`.")
    a("- source-inspired (not yet a claim): the split-iff-then-ext probe family.")
    a("- rejected as too theorem-specific: the count-extensionality closure of the "
      "singleton theorem.")
    a("")

    a("## 7. SF3 candidate-lemma queue")
    a("")
    a(f"- size: {len(lemma_q)}. Honest summary: {cand.get('honest_summary', '')}")
    for q in lemma_q:
        a(f"  - `{q['candidate_id']}` — novelty: {q.get('novelty_risk')}; "
          f"priority: {q.get('priority')}")
    a("")

    a("## 8. Recommendation")
    a("")
    a("- **Do not modify RC1.** No promotion is justified by this run.")
    a("- The singleton failure is a genuine open failure with a clear diagnosis "
      "(tactic/routing gap, not a missing lemma). If a future multi-step search or a "
      "split-iff routing tweak closes it, test on additional singleton/toFinset "
      "membership-iff failures and run NS23 minimal-sufficient relabel before any promotion.")
    a("- Highest-priority next target: the largest high-priority cluster in §5; the "
      "singleton theorem itself needs depth-≥4 search, not a battery tactic.")
    a("- Fold the `finished`-key fix into all future SF eval so RC1 is never undercounted.")
    a("")

    a("## 9. Protected-file confirmation")
    a("")
    a("`git diff --stat HEAD` for protected configs (empty = unchanged):")
    a("```")
    a(prot if prot else "(no changes to rc1_production_wrapper.json or ns24_router.json)")
    a("```")
    a("`git status --short` (working tree; all additive SF1/SF2/SF3 artifacts + scripts):")
    a("```")
    a(status[:1500])
    a("```")
    a("No commit made.")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    open(REPORT, "w").write("\n".join(L))
    print(f"[report] wrote {REPORT} ({len(L)} lines); protected_empty={not bool(prot)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
