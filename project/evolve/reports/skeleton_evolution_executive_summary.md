# Skeleton evolution — executive summary

## The result

Without retraining `gen_v5` (a t5-small Lean tactic generator), an
AlphaEvolve-style outer loop evolved a 17-skeleton proof-search
configuration that proves:

  - **37/38** on `nat_defs_medium` (97.4%)
  - **49/65** on `nat_defs_large_v5` (75.4%)

Same checkpoint, no fine-tune, no architecture changes.

## Raw model vs. evolved wrapper

| policy | medium | large |
|---|---:|---:|
| `gen_v5` plain (model alone) | 3/38 (7.9%) | not run |
| `hybrid_evolved` (NS9 best, 17 skel) | **37/38 (97.4%)** | **49/65 (75.4%)** |

The wrapper closes the gap by composing the model's beam-search
top-K with a small evolved layer of priority templates, family
tactics, deny-lists, and shape-aware premise retrieval. Lean is
the only evaluator throughout — every accepted candidate proves at
least as many theorems as the previous best on real Mathlib
rollouts.

## Compression

48 enabled skeletons (raw NS3-combined) → **17 enabled skeletons** —
**65% smaller**, same proved counts. The compaction was driven by
five iterations of safe pruning (NS5–NS9), each adding a new
diagnostic that turned a previously-rejected mutation class into a
provably-safe one.

## Why this is AlphaEvolve-style

AlphaEvolve doesn't search for *the* mathematical object — it
searches for a *program* that generates the object, using a
deterministic evaluator (LLM grading, etc) as the fitness function.

This project's analogue:

  - **Object** = a Lean proof.
  - **Program** = a strategy genome (`SkeletonBag` + wrapper
    config).
  - **Evaluator** = Lean / Dojo (strict, no partial credit, no
    proxies).
  - **Mutations** = small, archive-guided, scoped edits to the bag.
  - **Pre-flight safety** = a cached-model rank simulator (NS8)
    that rejects mutations *before* paying for a Lean roundtrip.

The genome is treated as data, the evaluator never changes, and
every iteration is a real Mathlib eval. No proxy metrics anywhere.

## What's left

The residual failure (`Nat.AM_GM` on medium; ~16 unproved on
large) is a *model-capability* ceiling. The skeleton bag has been
exhausted — no further configuration of the existing emission
machinery closes it. The next attack vector is a targeted fine-
tune of `gen_v5` on multi-step inductive arguments (NS10),
leveraging the per-step trace corpus the skeleton-evolution work
produced as supervision signal.

## Best genome on disk

  - Pointer: `project/evolve/best/ns9_best_genome.json`
  - Reproduce: `python eval_rollout_all.py --theorem-set
    nat_defs_medium --policy-type hybrid_evolved --ckpt-dir
    project/models/gen_v5 --top-k 8 --max-steps 8 --strategy-config
    project/evolve/best/ns9_best_genome.json` (~2.5 min)

## Reports index

- `skeleton_evolution_final_report.md` — full v3→NS9 progression.
- `ns5_skeleton_evolution_report.md` ... `ns9_retrieval_gate_decoupling.md`
  — per-iteration deep dives.
- `nat_defs_medium_summary.md` — chronological log.

## Commit trail

`aed695c` (NS9), `04b38bb` (NS8), `709ec70` (NS7), `2b6044b` (NS6),
`9e546f0` (NS5), `4a61ea1` (NS4.2), `28a3b0c` (NS4.1),
`88a739b` (NS4) on branch `ns5-skeleton-evolution`.
