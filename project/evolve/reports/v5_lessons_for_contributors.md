# v5 — lessons for the next contributor

A short document capturing the meta-learnings about *how to do this
research*, not just *what the results were*. Read this if you're
about to spend a few hours on v6.

## 1. Run a no-op cycle first

Before any code changes, the first thing the autonomous loop did
was reproduce v4.7's 26/38 on `nat_defs_medium`. Verifying the
baseline is unchanged costs one eval (~5 min) and prevents wasting
hours chasing phantom regressions later.

Concrete: every v5 followup loop starts with `v5-00-baseline-repro`.
Every wave's first cycle should be the previous wave's best,
re-evaluated.

## 2. Trace first, hypothesize second

When 12 first-pass variants all failed at 26/38, the temptation was
to keep adding templates. The actual breakthrough came from spending
~10 minutes reading the trace for a single failing theorem and
noticing that generative_topk's `simp [Nat.one_mul]` advanced
state at step 1, shadowing every downstream template.

Concrete: when a direction plateaus, stop adding variants. Read
one failing trace end-to-end. The mechanism is usually there.

## 3. The wrapper is the production artifact, not the model

`gen_v5` raw on `nat_defs_medium` proves 3 / 38. With the v5 wrapper:
31 / 38. The wrapper does ~10× the work. This is true for the
nat_defs Lean domain at T5-small scale.

Concrete: when iterating, optimize the wrapper. The model is fine.

## 4. Domain transfer is real but narrow

The 5 newly-proved priority_templates transferred from
`nat_defs_medium` to `nat_defs_large_v5`: same omega-omega template
closes 5 unseen-iff theorems on the new set. But on `demo_v1`
(Set-heavy), the wrapper adds only +1 over gen_v5 raw — because the
templates name Nat-specific lemmas.

Concrete: priority_templates is *per-domain*. For v6, treat domain
as a first-class shape gate.

## 5. Within-slot ordering is structural too

Wave 5's v5-31 reordering test showed that putting omega-omega
generic templates BEFORE specific div/mul templates inside the
iff slot regresses 4 wins. The wrapper's first-non-erroring-wins
semantics applies *within* a slot, not just between slots.

Concrete: hand-curate the order from most specific to most generic.
For v6, encode specificity in the slot type, not by ordering.

## 6. Diminishing returns are obvious in the scoreboard

After v5-27 hit 31/38 (wave 4), every subsequent variant in waves
5 and 6 also hit 31/38 — including v5-38-w6-combined which tried
4 new lemma forms. That's a saturation signal.

Concrete: when N variants in a row hit the same number, the
problem isn't more variants. Pivot to writing or to a structural
change.

## 7. The 30s-vs-3min cost difference of polling traces matters

The autonomous loop subprocess writes a `traces.jsonl` that grows
during the eval. You can read it mid-eval to see the current
theorem and step. This is faster than waiting for the eval to
finish; useful when a cycle seems hung.

Concrete: `python3 -c "import json,glob; lines=open(glob.glob('.../traces.jsonl')[0]).readlines(); last=json.loads(lines[-1]); print(last['full_name'], last['step'])"`.

## 8. Commit early; commit reports as separate artifacts

This branch has 11 commits, each focused. The reports are tracked;
the run artifacts (autonomous_runs/) are not. The reports point
to run-artifact paths so the audit trail is complete even without
committing artifacts. The user spec asked for this and the discipline
paid off — the working state is reproducible from git alone.

Concrete: never commit run artifacts. Reports cite paths and key
findings.

## 9. The eval is deterministic if seeds are unfixed

Re-running v5-27 master twice produced 31/38 both times. The
underlying LeanDojo + T5 generative beam search is deterministic
on the same input. Reproducibility is built in.

Concrete: re-run only matters if you change the input (model,
genome, theorem set). One eval per config is enough.

## 10. The mutator was deterministic too — and that's the bug

The v3 → v4 mutator was deterministic and finite-vocabulary. It
could explore *content* of existing slots but not *add new slots*.
v5's `priority_templates` slot had to be added by hand. This is
the gap a v6 LLM mutator could close.

Concrete: if your mutator is deterministic and your scoreboard
plateaus, the architecture is the bottleneck, not the search budget.
