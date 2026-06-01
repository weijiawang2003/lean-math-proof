# RC5S — Timeout-Safe Dynamic Stage Hardening Report

**Type:** engineering/safety hardening (no discovery, no release, no promotion, no wrapper change)
**Date:** 2026-06-01

---

## 1. Executive summary

RC5S hardens the RC5H dynamic retrieval stage against its three blockers (LeanDojo stalls,
off-policy grammar leakage, unbounded B10/B20).

| Metric | Result |
|---|---|
| strict grammar — RC5H programs removed | 523 of 1792 (366 stall-risk, 73 off-policy, 84 namespace) |
| 3 RC5H true-hybrid winners survive filtering | True |
| safe plan — off-policy programs | **0** |
| safe B5 — global stalls | **none** (max wall 12.3s, cap 60s) |
| safe B5 — bounded timeouts (process-kill) | 0 |
| RC5H winners reproduced under safe stage | **3/3** |
| safe new dynamic wins | 0 |
| B10 / B20 | 0 new wins → B5_ONLY / disabled |
| RC5S vs RC5H verdict | SAFETY_HARDENING_SUCCESS |

**Decision: RC5S_SAFETY_HARDENING_SUCCESS** (see §13).

---

## 2. Background

RC5H (`RC5H_DYNAMIC_STAGE_USEFUL_BUT_NOT_RELEASE_SAFE`) validated the hybrid concept (+3
TRUE_HYBRID_DELTA over RC4) but was not release-safe:

1. depth-2/3 `simp_all` / `<;> aesop` / depth-3 try chains **stall LeanDojo** at B10+; the
   per-tactic SIGALRM cannot interrupt them (22/88 hit the 150s cap; B20 unrunnable).
2. **74 off-policy programs** — the reused TR6 generator leaked a broad grammar.
3. RC5H ranker data hurts PR-AUC (3 positives) — do not retrain.

RC5S fixes (1) and (2) (engineering); (3) is out of scope (no retrain).

---

## 3. Strict policy

`rc5s_strict_policy.json` (`out/rc5s_policy_diff.*`).

- **Allowed grammar (low-risk, 8 patterns + simpa variant):** `exact L`, `simpa using L`,
  `simpa [L]`, `simp [L]`, `rw [L]`, `simp [L] <;> aesop`, `rw [L] <;> aesop`, `ext x <;> simp [L]`,
  `constructor <;> intro h <;> aesop`.
- **Removed by default:** anything containing `simp_all`; depth-3 try chains; any program with ≥2
  `<;>` except the single `constructor <;> intro h <;> aesop` pattern; bare tactics
  (aesop/omega/nlinarith/tauto with no lemma).
- **`<;> aesop` namespace-gated** to historically-safe {Set, Finset, List, Multiset} (not Nat).
- **Namespaces:** {Set,Finset,List,Multiset,Nat} allowed; Order disabled.
- **Budgets:** B5 default; B10 = safe NON-aesop families only; **B20 disabled**.
- **Timeout policy:** per-theorem wall cap (60s) enforced by **process-group SIGTERM→SIGKILL**
  (`run_with_timeout.py`), per-tactic SIGALRM (8s) best-effort, checkpoint each theorem,
  deterministic resume.

---

## 4. Existing-plan filter

`out/rc5s_filter_report.json`. RC5H plan 1792 programs → **1269 POLICY_ALLOWED**; removed
523 (366 REMOVED_STALL_RISK, 73 REMOVED_OFF_POLICY, 84
REMOVED_NAMESPACE_DISABLED). **All 3 RC5H true-hybrid winning programs survive** (True) —
they use `simp [L]` / `simp [L] <;> aesop`, all in the low-risk grammar.

---

## 5. Hardened benchmark set

`cases/rc5s_benchmark_summary.json` — **111 theorems** (safety/hardening, not coverage): 3
winners + 84 prior-stall cases + 2 off-policy cases + 12 Nat/Order hard negatives + 10 floor
smoke (+ TR7 dynamic-tail / eligible-no-win folded into the above via dedup).

---

## 6. Safe dynamic plan

`out/rc5s_safe_dynamic_plan.json` — **89 theorems with programs, 22 gated out** (namespace
disabled / not eligible). **B5 444 programs, B10 reserve 384, off-policy in final plan = 0.**
B5 patterns: simp_L 242, simp_L_aesop 168, exact_L 16, simpa_using_L 12, rw_L_aesop 4, ext_simp_L 1, simpa_L 1.

---

## 7. Timeout-safe runner

`rc5s_timeout_safe_runner.py` (reusable core).

- Each theorem runs in an **isolated subprocess** wrapped by `run_with_timeout.py`, which enforces
  a hard wall-clock cap via **process-group SIGTERM→SIGKILL** (exit 124 on timeout) — this bounds
  a stuck LeanDojo/aesop/simp_all even when it ignores the per-tactic SIGALRM (the exact RC5H
  failure).
- The inner Lean probe reuses the validated TR5 worker (one Dojo, runs the programs).
- Records `started_at / ended_at / wall_seconds / killed_by_timeout / exit_code / outcomes` per
  theorem; **per-theorem checkpoint + deterministic resume**. Reusable by RC5H-v2.

---

## 8. Safe B5 results

`out/rc5s_b5_results.json` — **clean success**.

| metric | value |
|---|---|
| theorems | 89 |
| dynamic successes | 3 |
| **global stalls** | **none** (max wall **12.3s**, cap 60s) |
| killed_by_timeout (bounded) | **0** |
| off-policy programs | **0** |
| **RC5H winners reproduced** | **3/3** (`Finset.biUnion_subset_iff_forall_subset`, `Finset.image_subset_iff`, `Multiset.add_bind`) |
| unknown-name (bounded, not stalls) | 28 |

Removing the stall-prone tactics (`simp_all`, depth-3 try chains) means **no theorem approaches
the 60s cap** — the worst wall time was 12.3s. The hardening eliminated the RC5H stalls outright.
Notably `Finset.image_subset_iff` (a B10 win in RC5H) now reproduces at **B5** because strict
filtering reordered its `simp [Finset.subset_iff]` into the top-5.

---

## 9. Optional safe B10 results

`out/rc5s_b10_results.json` — ranks 6–10, safe NON-aesop reserve only, on the 85 B5-unsolved.

- new wins: **0** | killed_by_timeout: **0** | total wall: 854.9s | no global stalls: **True**
- **recommendation: `B5_ONLY`** — B10 adds 0 marginal yield at material cost, and (unlike RC5H,
  where B10 stalled pervasively) it is now fully bounded.

---

## 10. Attribution and safety classification

`out/rc5s_attribution_and_safety.json`.

- **SAFE_TRUE_DYNAMIC_WIN: 3** (all 3 RC5H winners) · NO_WIN_SAFE_BUDGET: 86.
- recovered prior wins: **3/3** · lost: 0 · new: 0 · bounded timeouts: 0.
- off-policy blocked pre-execution: 73 · unsafe (stall-risk) quarantined: 366.
- remaining safety issues: none (0 unbounded stalls, 0 lost wins).

---

## 11. RC5S vs RC5H comparison

`out/rc5s_vs_rc5h_comparison.json` → **`SAFETY_HARDENING_SUCCESS`**.

| metric | RC5H | RC5S |
|---|---|---|
| programs | 1792 | 1269 (strict-filtered) |
| off-policy | ~74 | **0** |
| global stalls | pervasive @B10+ (22/88 hit 150s, manual kills) | **none** |
| timeout handling | SIGALRM-only (failed on simp_all/aesop) | **process-group kill**, 0 bounded kills needed |
| max wall | 150s (cap) | **12.3s** |
| true wins | 3 | 3 (**3/3 recovered**) |
| B10 / B20 | pervasive stalls / unrunnable | bounded, B5_ONLY / disabled |

---

## 12. Exported safety dataset

`data/rc5s_safe_dynamic_examples.jsonl` — **828 safe attempt rows** (one program = one row) with
policy status / timeout status / outcome / safety class. **off-policy examples: 0** · winning
programs: 3. Safety/audit data only — the ranker is **NOT** retrained (3 positives; RC5H showed
retrain hurts PR-AUC).

---

## 13. Decision

**RC5S_SAFETY_HARDENING_SUCCESS**

All success criteria are met: **0 off-policy programs** (strict grammar enforced before execution),
**no global stalls** (max wall 12.3s under a 60s process-group-kill cap — the depth-2/3
`simp_all`/aesop stallers were removed), **all timeouts bounded and recorded** (0 were even
needed), **all 3 prior true-hybrid wins preserved** (3/3, one even promoted from B10→B5), and a
clear **B5-only** budget recommendation. The RC5H blockers are fixed at the engineering level: the
process-group-kill watchdog bounds any residual stall regardless of SIGALRM, and the strict
grammar eliminates the leakage and the worst stallers. RC5H ranker-retrain (blocker 3) is
intentionally out of scope.

---

## 14. Next steps

Hardening succeeded → the dynamic stage is now safe, bounded, reproducible, and auditable.
1. **Run RC5H-v2** — a fresh out-of-sample benchmark using the RC5S strict policy + timeout-safe
   runner (B5-only), to measure fresh dynamic delta now that the stage is safe to run at scale.
2. Keep **B5-only** as the production-experiment budget (B10 adds 0 yield; B20 disabled).
3. Gather more fresh dynamic-win positives (TR8) before any ranker retrain.
The `rc5s_timeout_safe_runner.py` is the reusable safe-execution core for all of the above.
RC2 stays production; RC4 static stays the best static candidate; RC5H originals untouched.

---

## 15. Protected-file confirmation

- RC1 wrapper — untouched · RC2 release wrapper — untouched · NS24 router — untouched
- RC4R wrapper — untouched · **RC5H original artifacts — untouched** · NS9 · REL1/RC1/RC2 reports
  · TR1–7 datasets — untouched
- No production change · no RC5 release · no promotion · **no commit made**
- `git diff --stat HEAD` over protected wrappers + router + RC5H policy: **empty**.
