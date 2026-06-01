# RC5S Methodology

## Pipeline

1. **Strict policy** (`rc5s_strict_policy.json`) — allowed low-risk grammar (8 patterns), removed
   stall families (`simp_all`*, depth-3 try chains, bare aesop/omega/tauto), namespace gates
   ({Set,Finset,List,Multiset,Nat}; Order disabled), aesop-tail gated to historically-safe
   namespaces ({Set,Finset,List,Multiset}), budgets (B5 default, B10 safe-families-only, no B20),
   and a timeout policy (per-theorem wall cap + per-tactic SIGALRM + **process-group kill
   fallback** + per-theorem checkpoint + deterministic resume).
2. **Filter the existing RC5H plan** through the strict policy — classify every emitted program
   (POLICY_ALLOWED / REMOVED_STALL_RISK / REMOVED_OFF_POLICY / REMOVED_NAMESPACE_DISABLED /
   REMOVED_LOW_CONFIDENCE / REMOVED_DUPLICATE), confirm the 3 true-hybrid winners survive.
3. **Hardened benchmark set** — the 3 winners + B10 stall cases + off-policy-emission cases + TR7
   dynamic-tail + Nat/Order hard negatives + eligible-but-no-win fresh failures + a small floor smoke.
4. **Safe dynamic plan** — strict-grammar programs only (reject off-policy *before* scoring), TR4
   ranker scores, top-5 (B5) + a safe B10 reserve; 0 off-policy in the final plan.
5. **Timeout-safe runner** (core) — per-theorem subprocess under `run_with_timeout` (process-group
   SIGTERM→SIGKILL) with a tight wall cap; records started/ended/wall_seconds/killed_by_timeout/
   exit_code/outcomes; checkpoint each theorem; deterministic resume.
6. **Safe B5** — top-5 safe programs/theorem; assert 0 global stalls, all timeouts bounded.
7. **Safe B10** — ranks 6–10 on unsolved, safe families only; report marginal yield + timeout cost.
8. **Attribution + safety** — SAFE_TRUE_DYNAMIC_WIN / SAFE_NEW_DYNAMIC_WIN / LOST_WIN_DUE_TO_POLICY
   / TIMEOUT_BOUNDED / OFF_POLICY_BLOCKED / UNSAFE_PROGRAM_QUARANTINED / NO_WIN_SAFE_BUDGET.
9. **RC5S vs RC5H comparison** — program/off-policy/stall/timeout/win/cost deltas → hardening verdict.
10. **Export** safety dataset (one safe attempt = one row). No ranker retrain (safety data only).

## Core safety invariant

No single program or theorem can stall the run: the outer `run_with_timeout` watchdog kills the
whole process group at the per-theorem wall cap regardless of whether LeanDojo/aesop/simp_all
ignore SIGALRM. Every kill is recorded as a bounded `killed_by_timeout` event, not a hang.

## Determinism

Programs are ordered by ranker score then lexically; per-theorem checkpoint enables exact resume;
the strict grammar is a pure function of the tactic string + namespace.
