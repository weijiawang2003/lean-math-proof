# RC4R — RC4 Release Candidate Benchmark

RC4R prepares an **off-by-default RC4 release candidate wrapper** from the validated RC4D
composition and runs a formal **RC2-vs-RC4 benchmark** to determine whether RC4 is ready to be
*recommended* as the next production wrapper. **This is a release-candidate benchmark, not a
production promotion.** RC2 stays the production stack; RC4 remains release-candidate only unless
the owner approves.

## What RC4 is

    RC4 = RC2 ⊕ RC4A ⊕ RC4B ⊕ RC4C_residue   (= the validated RC4D composition)

The RC4 release wrapper is a **clean RC2-based wrapper**: RC2's actions preserved exactly, plus
the 15 validated RC4D deployable tactics prepended to `priority_templates["any"]` and
name-prefix-gated via `theorem_name_tactic_gates`. The wrapper is **purely additive** — on any
theorem whose name matches no RC4 gate prefix, every RC4 tactic is gate-denied and the search is
byte-identical to RC2 (so RC4 ≡ RC2 on all non-gate-firing theorems by construction).

## Recommendation criteria

RC4 can be recommended (`RC4_RELEASE_CANDIDATE_RECOMMENDED`) only if:

- positive **net delta** over RC2 (new wins − regressions > 0),
- **0 regressions** on canonical floors (hard fail otherwise),
- **0 off-gate** emissions,
- **deterministic** (or only explicit infra flakes not affecting wins/regressions),
- canonical floors preserved (demo_v1, nat_defs_medium, nat_defs_large_v5 ≥ RC2 release floor),
- clean wrapper diff (only the validated RC4 actions added; RC2 fields untouched),
- at least known-win reproduction, **preferably a fresh out-of-sample frontier delta**.

## Benchmark design / reuse

RC4D already benchmarked the functionally-identical RC4D wrapper on the canonical floors
(demo 12→12, medium 37→37, large 49→49) and the 23 attributed known wins (RC4 22/23, RC2 0/23).
Because the RC4 release wrapper has identical actions/gates (only the metadata key differs, which
the loader ignores), those results are **reused** with a demo_v1 spot-check re-run. The genuinely
new measurement is the **fresh out-of-sample frontier** — theorems from the TR6 fresh pool that
were NOT used in RC4D validation — run live for both wrappers (RC4 live only where its gate fires;
elsewhere RC4 ≡ RC2 is reused).

## Protected files

RC1/RC2 release wrappers, NS24 router, NS9 checkpoints, REL1/RC1/RC2 reports, TR1–TR6 datasets,
RC4A/B/C/D source artifacts — **untouched**. No README update, no production routing change, no RC2
replacement, no production release, no commit unless instructed.
