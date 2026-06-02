# FLI2 — Large-Scale Retrieval-Gap Lemma Deployment

**Decision: `FLI2_RETRIEVAL_GAP_RESCUES_FOUND`**

FLI2 scaled FLI1's retrieval-gap signal to a 217-failure pool and asked whether retrieved-but-
undeployed lemmas can rescue failed theorems when used as gated enabling actions. Result: **6
robust, non-vacuous downstream rescues** (≥3 threshold met) plus 30 partial-progress cases, all
discovered automatically from failure analysis. This advances the research-assistant goal: the
system can convert failure analysis into actionable lemma deployment. Discovery only — no
production change, no commit, nothing promoted.

## 1. Executive summary

- Pool: **217** retrieval-gap / high-signal failures (15 FLI1 retrieval-gap + 202 FLI0 high-signal).
- Actions: **1,472** gated deployment actions; **1,059 run live** over **161 theorems** (the 200-
  action prescribed batch was cheap, so we expanded; the slow Multiset/List `aesop` tail — ~51
  theorems — was truncated for tractability and is reported honestly, not hidden).
- **Candidate solves: 14 → 11 robust rescue-candidate actions over 6 distinct theorems.**
- **TRUE_RETRIEVAL_GAP_RESCUE: 6 theorems** (controls fail at position, candidate solves, robust,
  non-vacuous): `Finset.card_le_one_iff`, `Finset.mem_filterMap`, `Finset.card_subtype`,
  `Finset.mem_map`, `Finset.mem_preimage`, `List.bidirectionalRec_singleton`.
- **PARTIAL_PROGRESS: 30 theorems**; CONTROL_DUPLICATE 1; UNKNOWN_NAME_OR_IMPORT_GAP 30 (retrieved
  lemma not in scope at position); NO_RESCUE 94; 0 infra; 0 SELF_IMPORT_VACUOUS.
- Mined **25 deployment-rule families** (1 clean promotion-candidate, the rest needs-more-data);
  **all 11 rescue actions are NEW families beyond RC4B/RC4C** (overlap 0).

## 2. Project motivation (reminder)

The goal is a verifier-guided mathematical research assistant: use Lean as the verifier, treat
proof-search failures as signal, identify missing intermediate lemmas *or missing deployments*,
and test whether they rescue downstream theorems. FLI2's metric is downstream rescue and reusable
deployment rules — not solved count.

## 3. Why FLI2 follows FLI1

FLI1 found 1 robust rescue and 15 retrieval gaps: the relevant lemma often already exists and is
even retrieved, but the search never turns it into the right action. FLI2 tests this at scale.

## 4. Retrieval-gap pool construction

217 cases, deduped by theorem (FLI1 records precedence): A) 15 FLI1 confirmed retrieval gaps;
C) 202 FLI0 clean high-signal failures (nonempty retrieved lemmas; namespace ∈ {Finset, List,
Multiset, Set, Nat}; high-value bridge pattern; not unknown-name/infra). Patterns: MAP_FILTER_BIND
101, SUBSET 39, IFF 36, MEMBERSHIP 20, INDUCTION 11, SINGLETON 6, DISJOINT 3, EXT 1.

## 5. Deployment action generation

1,472 gated actions (avg ~6.8/theorem, cap 8). Templates: SIMPLE_SIMP 617, SIMP_AESOP 502,
EXACT_LEMMA 188, CONSTRUCTOR_SIMP 76, OMEGA_CLOSER 46, GCONGR_CLOSER 38, EXT_SIMP 3,
INTRO_SIMP_AESOP 2. Each gated by namespace + constant overlap + pattern. Banned: simp_all, bare
aesop as credited deploy, depth-3 chains, B20 search, unknown lemmas.

## 6. Live evaluation setup (vacuity-safe)

Theorem-centric: one LeanDojo Dojo per theorem **at its real file position** (target theorem &
downstream out of scope — no fresh `import Module`, avoiding FLI1's vacuity trap). Controls (simp,
aesop, classical<;>aesop, constructor<;>simp, ext<;>simp) run first; then every gated action from
the initial state. Process-group hard timeout + per-tactic SIGALRM + per-theorem checkpoint/resume.
Rescue candidates (solved + all controls fail, non-vacuous) re-run once for robustness.

## 7. Rescue attribution

| class | theorems |
|---|---|
| **TRUE_RETRIEVAL_GAP_RESCUE** | **6** |
| PARTIAL_PROGRESS | 30 |
| UNKNOWN_NAME_OR_IMPORT_GAP | 30 |
| CONTROL_DUPLICATE | 1 |
| NO_RESCUE | 94 |

**The 6 true rescues** (all verified: bare controls incl. `aesop` fail at position; deployment
closes; robust on re-run):

| theorem | deployed lemma/def | tactic | kind |
|---|---|---|---|
| `Finset.card_le_one_iff` | `Finset.card_le_one` | `simp [L] <;> aesop` | lemma-bridge (FLI1 ↔ RC4C-style) |
| `Finset.mem_filterMap` | `Finset.filterMap` | `simp [L]` | def-unfold (RC4A-style) |
| `Finset.card_subtype` | `Finset.subtype` | `simp [L]` | def-unfold |
| `Finset.mem_map` | `Finset.map` | `simp [L]` | def-unfold |
| `Finset.mem_preimage` | `Finset.preimage` | `simp [L]` | def-unfold |
| `List.bidirectionalRec_singleton` | `List.bidirectionalRec` | `simp [L]` | def-unfold |

Mechanistic reading: at the theorem's position the plain controls don't fire, but unfolding the
relevant definition (or deploying the exact characterization lemma) via a gated `simp [L]` exposes
the structure and closes the goal. This is precisely the RC4A/RC4B/RC4C deployment pattern —
**discovered automatically from failure analysis rather than hand-curated.**

## 8. Deployment rule mining

25 candidate families (by namespace × lemma-family × template). 6 carry rescue support
(FINSET_MAP / FINSET_IMAGE / FINSET_SUBTYPE / FINSET_CARD / FINSET_MEM / LIST_* bridges); 1 reaches
clean `candidate` promotion_status, the rest are `needs_more_data` (1 rescue amid many same-family
non-rescues → high false-positive rate, so the gate would need tightening before any validation).
This is the honest state: the *families* are real, but most need a sharper trigger than
"same namespace + same lemma-family" before they could be RC-candidates.

## 9. Comparison to RC4B/RC4C

RC4B/RC4C were **manually** built lemma-enabling static wrappers (disjoint_left bridge; selected
`simp [L]` enablers) validated via literal-RC2 additive eval. FLI2 **discovers** the same *kind* of
object — a small gated `simp [L]` / closer deploying an existing lemma — but sourced automatically
from failure analysis. **Family overlap with RC4B/RC4C = 0; all 11 rescue actions are new
families** (membership/def-unfold over Finset map/filterMap/preimage/subtype/card, List rec).
Notably several are **def-unfold (RC4A-family)** deployments, so FLI2 spans the full RC4A–RC4C
deployment space. FLI2 therefore looks like a **scalable RC-candidate *generator*** — but it only
emits candidates; each would still need the full literal-RC2 additive validation
(off-gate/floors/determinism) before becoming an RC candidate.

## 10. Main findings

1. Failure analysis **can** convert retrieved-but-undeployed lemmas into actionable deployments
   that rescue failed theorems — 6 robust, non-vacuous rescues at scale.
2. The deployable signal is concentrated and honest: 30 retrieval gaps were *availability* gaps
   (lemma not in scope at position, UNKNOWN_NAME), 30 produced partial progress, and most actions
   simply don't help — the wins are where a control genuinely fails but a specific lemma/def closes.
3. FLI2 rediscovered the RC4A (def-unfold) and RC4B/RC4C (lemma-bridge) patterns automatically,
   suggesting an automated RC-candidate generator is feasible.

## 11. Limitations

- **Coverage:** 161/217 theorems evaluated; the slow Multiset/List `aesop` tail (~51) was truncated
  for tractability (each hit the per-theorem cap). Reported, not hidden; those are slow precisely
  because no quick close exists, so additional rescues there are unlikely but unmeasured.
- **Rule maturity:** only 1 of 25 families is a clean promotion-candidate; most have high
  false-positive rates under the coarse trigger and need sharper gates.
- **Not validated/promoted:** rescues are at-position LeanDojo solves, not RC4-style literal-RC2
  additive validations. No off-gate/floors/determinism check was run; nothing is production-bound.
- Robustness used a tight re-run (winning tactic + controls); one card_le_one action was flaky
  (its sibling action is robust, so the theorem still counts).

## 12. Recommended FLI3

1. Take the 6 rescues (+ best partials) into the **RC4-style literal-RC2 additive validation
   harness** (off-gate/floors/determinism) — turn discovery into validated RC candidates.
2. **Tighten deployment-rule triggers** (constant-level, not just namespace+family) to cut the
   false-positive rate and lift more families to `candidate`.
3. For the 30 **availability-gap** cases (UNKNOWN_NAME), add the missing import/scope so the
   retrieved lemma is deployable, then re-test.
4. Feed the 30 **partial-progress** residuals into FLI1-style multi-step lemma invention.

## 13. Protected-file confirmation

`git diff --stat HEAD` over RC1/RC2/RC4-release/RC5S-policy wrappers + NS24 router = **empty**. No
RC*/TR*/FLI0/FLI1 committed artifact, production wrapper, routing config, or README modified. FLI2
wrote only under `project/evolve/experiments/fli2/`, `project/evolve/reports/fli/`, and
`scripts/fli2_*.py`. Nothing promoted, ranker not retrained, **no commit**.
