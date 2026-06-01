# RC5S — Timeout-Safe Dynamic Stage Hardening

RC5S is a **hardening task, not a discovery task**. It takes RC5H's useful-but-unsafe dynamic
retrieval stage and makes it **strict-policy, timeout-safe, B5-first, Dojo-stall-aware,
reproducible, and auditable**. It does **not** seek new candidate families and does **not** alter
any production wrapper.

## What RC5H broke (and RC5S fixes)

| RC5H blocker | RC5S fix |
|---|---|
| depth-2/3 `simp_all` / `<;> aesop` / depth-3 try chains stall LeanDojo at B10+; per-tactic SIGALRM can't interrupt them (22/88 hit the 150s cap; B20 unrunnable) | **strict low-risk grammar** removes `simp_all`/depth-3-try; **timeout-safe runner** wraps every theorem in a hard wall-clock cap enforced by **process-group kill** (not SIGALRM) so no program can stall the run |
| 74 off-policy programs (broad TR6 grammar leakage) | **strict grammar enforcement** — every program must match an allowed pattern exactly; off-policy programs are blocked before execution (target 0) |
| unclear safety metrics | every attempt records `wall_seconds`, `killed_by_timeout`, `exit_code`; bounded timeouts are first-class |

## The 3 RC5H true-hybrid winners (must survive)

All three use low-risk tactics, so the strict grammar preserves them:

- `Finset.biUnion_subset_iff_forall_subset` — `simp [Finset.biUnion_subset] <;> aesop` (Finset, aesop-safe ns)
- `Multiset.add_bind` — `simp [Multiset.bind]`
- `Finset.image_subset_iff` — `simp [Finset.subset_iff]`

## Scope

This is a **safety/hardening benchmark**, not a coverage benchmark. Success = 0 off-policy
programs, no global stalls, every timeout bounded and recorded, most prior true wins preserved,
and a safe budget recommendation. RC2 stays production; RC4 static stays the best static candidate;
RC5H originals are untouched.

## Protected

RC1/RC2/NS24, RC4R wrapper, NS9, REL1/RC1/RC2 reports, TR1–7 datasets, RC4A/B/C/D/R + **RC5H
original** artifacts — untouched. No README update, no routing change, no RC5 release, no
promotion, no commit.
