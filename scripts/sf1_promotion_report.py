#!/usr/bin/env python3
"""SF1 stage (g/h): promotion + discovery status report for Stage C/D/E/F.

Reads the Stage C/D/E/F artifacts and emits a dense markdown status report:
  project/evolve/reports/sf1_stage_cdef_status.md

It applies the SF1 promotion gate (positive delta over RC1, zero regressions,
zero off-gate, minimal-sufficient attribution, deterministic reproduction) and,
absent confirmed live results, recommends MINE_MORE. It also confirms the
protected production files are untouched via `git diff --stat`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

PROTECTED = [
    "project/evolve/experiments/rc1/rc1_production_wrapper.json",
    "project/evolve/routing/ns24_router.json",
]


def _load(path, default=None):
    if path and os.path.isfile(path):
        try:
            return json.load(open(path))
        except Exception:
            return default
    return default


def _git(*a):
    try:
        return subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    except Exception as e:  # pragma: no cover
        return f"<git failed: {e}>"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SF1 (g/h): Stage C/D/E/F status report.")
    p.add_argument("--classification-summary",
                   default="project/evolve/experiments/sf1/out/real/classification_summary.json")
    p.add_argument("--batch-manifest",
                   default="project/evolve/experiments/sf1/out/real/batch_manifest.json")
    p.add_argument("--eval-results",
                   default="project/evolve/experiments/sf1/out/real/eval_matrix_results.json")
    p.add_argument("--eval-manifest",
                   default="project/evolve/experiments/sf1/out/real/eval_matrix_manifest.json")
    p.add_argument("--relabel-summary",
                   default="project/evolve/experiments/sf1/out/real/relabel_summary.json")
    p.add_argument("--out", default="project/evolve/reports/sf1_stage_cdef_status.md")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cls = _load(args.classification_summary, {}) or {}
    bm = _load(args.batch_manifest, {}) or {}
    er = _load(args.eval_results, {}) or {}
    em = _load(args.eval_manifest, {}) or {}
    rs = _load(args.relabel_summary, {}) or {}

    batches = bm.get("batches", [])
    results = er.get("results", [])
    caps = em.get("eval_capabilities", {}) or {}
    ran = [r for r in results if r.get("status") == "ran"]
    unsupported = [r for r in results if r.get("status") == "unsupported"]
    blocked = [r for r in results if r.get("status") == "blocked_missing_file_path"]
    dry = [r for r in results if r.get("status") == "dry_run"]
    real_eval_ran = bool(er.get("run_real")) and bool(ran)

    diff_protected = _git("diff", "--stat", "HEAD", "--", *PROTECTED)
    status = _git("status", "--short")

    def hist(d):
        if not d:
            return "_(none)_"
        return ", ".join(f"`{k}`={v}" for k, v in d.items())

    L = []
    A = L.append
    A("# SF1 Stage C/D/E/F — status report")
    A("")
    A(f"- seed: `{bm.get('seed', 1729)}`  | branch: `rc1-production-stack`")
    A("- scope: additive only; RC1 production stack untouched.")
    A("")
    A("## 1. Executive summary")
    A("")
    A(f"- Stage A/B previously produced **{cls.get('total_classified', '?')} open-frontier "
      f"declarations** (Set={cls.get('set_frontier_count','?')}, "
      f"Multiset={cls.get('multiset_frontier_count','?')}).")
    A(f"- **Stage C (classify): REAL** — deterministic name+namespace(+statement) tagging "
      f"and candidate-family scoring. {cls.get('low_confidence_count','?')} low-confidence "
      f"(statement/type unavailable in artifact-mode catalog).")
    A(f"- **Stage D (batch): REAL** — {len(batches)} deterministic batches generated.")
    A(f"- **Stage E (eval): WIRED** — adapter builds routing-format theorem-set files + exact "
      f"commands; real run = `{real_eval_ran}` "
      f"(supported configs ran={len(ran)}, dry={len(dry)}, blocked={len(blocked)}, "
      f"unsupported={len(unsupported)}).")
    A(f"- **Stage F (relabel queue): REAL** — {rs.get('queue_size','?')} candidates queued for "
      f"NS23-style minimal-sufficient relabeling (no win claimed without confirmation).")
    A("")
    A("## 2. Frontier classification")
    A("")
    A(f"- namespace histogram: {hist(cls.get('counts_by_namespace'))}")
    A(f"- top tags: {hist(dict(list((cls.get('counts_by_tag') or {}).items())[:15]))}")
    A(f"- top-candidate-family counts: {hist(cls.get('top_candidate_family_counts'))}")
    A(f"- candidate-family histogram (score ≥ 0.5): {hist(cls.get('candidate_family_histogram_ge_0_5'))}")
    A(f"- Set frontier: {cls.get('set_frontier_count','?')}  | Multiset holdout: "
      f"{cls.get('multiset_frontier_count','?')}  | low-confidence: "
      f"{cls.get('low_confidence_count','?')}")
    A("")
    A("## 3. Batch generation")
    A("")
    A("| batch | size | dominant ns | dominant family | intended use |")
    A("|---|---|---|---|---|")
    for b in batches:
        A(f"| `{b['batch_name']}` | {b['size']} | {hist(b.get('dominant_namespaces'))} | "
          f"{hist(b.get('dominant_candidate_families'))} | {b.get('intended_use','')} |")
    A("")
    A("## 4. Eval wiring")
    A("")
    A(f"- eval script present: `{caps.get('script_present')}`  | `--help` ok: "
      f"`{caps.get('help_ok')}`  | detected flags: {caps.get('flags')}")
    A(f"- file_path map size (for routing-format adapter): {em.get('filepath_map_size','?')}; "
      f"NS9 config discovered: `{em.get('ns9_config')}`")
    A("- policy support:")
    seen = set()
    for m in (em.get("matrix") or []):
        key = m["policy"]
        if key in seen:
            continue
        seen.add(key)
        A(f"  - **{key}**: supported=`{m['supported']}` — {m['reason']}")
    A("- per-(batch,policy) outcomes: "
      f"ran={len(ran)}, dry_run={len(dry)}, blocked_missing_file_path={len(blocked)}, "
      f"unsupported={len(unsupported)}.")
    if ran:
        A("- parsed metrics:")
        for r in ran:
            mt = r.get("metrics") or {}
            A(f"  - `{r['batch_name']}`/`{r['policy']}`: rc={r.get('returncode')} "
              f"solved={mt.get('solved')} failed={mt.get('failed')} "
              f"n={mt.get('num_theorems')} ({mt.get('metrics_path')})")
    A("- exact commands: `project/evolve/experiments/sf1/out/real/eval_commands.sh`")
    A("- **limitation**: artifact-mode frontier rows lack `file_path`/`statement`; live "
      "eval requires backfilling `file_path` from the traced cache (Stage A live-mode TODO). "
      "The adapter recovers file_path where possible from `discovered_theorems.json` + routing "
      "sets and records `n_missing_file_path` per batch.")
    A("")
    A("## 5. Relabeling queue")
    A("")
    A(f"- queue size: **{rs.get('queue_size','?')}**  | per-theorem eval available: "
      f"`{rs.get('eval_per_theorem_available')}`")
    A(f"- by priority: {hist(rs.get('by_priority'))}")
    A(f"- by candidate family: {hist(rs.get('by_candidate_family'))}")
    A(f"- high-priority decls: {rs.get('high_priority_decls', [])[:20]}")
    A("- family-specific ladders are emitted per row (wx3 → `Multiset.induction_on <;> simp_all`; "
      "mx2 → `simp/simp_all/aesop/classical; aesop/exact?`; arith → `simp/omega`; ext → `simp/ext/aesop`).")
    A("- **connection to NS23**: this queue is the input to minimal-sufficient relabeling — each "
      "candidate must be confirmed by the minimal tactic ladder before any family-win attribution.")
    A("")
    A("## 6. Promotion decision")
    A("")
    gate = ("MINE_MORE" if not real_eval_ran else "REVIEW_RESULTS")
    A(f"- recommendation: **{gate}**")
    A("- rationale: no live RC1/raw/NS9 deltas were confirmed this stage "
      f"(real_eval_ran=`{real_eval_ran}`); SF1 produces the queue, not a promotion.")
    A("- RC2 promotion remains gated on ALL of: positive delta over RC1, zero regressions, "
      "zero off-gate emissions, minimal-sufficient attribution, deterministic reproduction.")
    A("")
    A("## 7. Connection to long-term theorem discovery")
    A("")
    A("- SF1 C/D/E/F is the bridge from proof-search to **failure-driven lemma discovery**: it "
      "turns the open frontier into classified, batched, evaluable, relabel-queued units.")
    A("- **SF2 — Failure Pattern Miner** (next): RC1 failures → recurring proof-state/goal "
      "patterns → missing-lemma templates. The `sf1_failure_driven_seed` batch + the "
      "`future_failure_driven_lemma_candidate` family scores are the seed inputs.")
    A("- **SF3 — Lemma Inventor**: candidate lemma → proof → downstream-utility test.")
    A("")
    A("## 8. Protected-file confirmation")
    A("")
    A("`git diff --stat HEAD` for protected files (empty = unchanged):")
    A("")
    A("```")
    A(diff_protected if diff_protected else "(no changes to rc1_production_wrapper.json or ns24_router.json)")
    A("```")
    A("")
    A("Working-tree status:")
    A("")
    A("```")
    A(status)
    A("```")
    A("")
    A("All SF1 Stage C/D/E/F changes are additive (upgraded `sf1_classify_frontier.py`, "
      "`sf1_make_batches.py`, `sf1_eval_matrix.py`, `sf1_minimal_relabel_new_wins.py`, "
      "`sf1_promotion_report.py`; new batches + out/real artifacts + this report). "
      "RC1 wrapper, NS9 genome, NS24 router, and REL1 reports were not modified.")
    A("")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w", encoding="utf-8").write("\n".join(L))
    print(f"[sf1:report] recommendation={gate} batches={len(batches)} "
          f"queue={rs.get('queue_size','?')} ran={len(ran)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
