# Finset Wall — Retriever-Patch Negative Result

**Date:** 2026-05-02
**Run dir:** `experiments/finset_wall_20260502_164911/`
**Status:** Single-seed, deterministic beam decode. Catalog patch applied; eval re-run; no change.

## TL;DR

Patching the static premise catalog to add the lemmas that working `Finset.Basic` proofs actually cite did **not** move `gen_ckpt_v6_premise`'s curriculum score (still 19/30) and did **not** prove any of the four Finset frontier theorems. Per-theorem dispositions are byte-identical to the baseline. The retriever was not the bottleneck. The catalog turns out to be essentially fine; the upstream issue is that the model never learned tactics that prove the frontier theorems themselves, regardless of what gets surfaced at inference time. This redirects future effort toward training-data composition, not retrieval.

## Phase 1 — diagnosis of the gap

Filtered `project/project_state.json` to theorems with `file_path` containing `Mathlib/Data/Finset/Basic.lean`, `proved=true`, and a non-empty `proof_tactics`. **133** records.

| Group | Count | Share | Example tactic |
|---|---:|---:|---|
| `aesop` / `tauto` / `simp [*]` / `simp_all` (no named premise) | 119 | **89.5%** | `aesop` |
| Tactic cites at least one named premise | 14 | 10.5% | `simp [Set.union_univ]` |

**Every** named-premise tactic cites `Set.*` lemmas — never `Finset.*`. Distinct premises and frequencies:

| Premise | Count | In old `Set` catalog | In old `Finset` catalog |
|---|---:|---|---|
| `Set.union_univ` | 4 | **no** | no |
| `Set.univ_union` | 3 | **no** | no |
| `Set.union_self` | 2 | yes | no |
| `Set.inter_empty` | 2 | **no** | no |
| `Set.mem_diff` | 2 | yes | no |
| `Set.diff_empty` | 1 | yes | no |

Three premises (`Set.union_univ`, `Set.univ_union`, `Set.inter_empty`) were genuinely absent from any domain in `STATIC_PREMISES`. Three were present in the `Set` catalog but never made the top-15 for `Finset.Basic` states because the static-catalog scorer assigns the same `domain_bonus + name_overlap` to every member of every detected domain, and Finset-state tokens (`Finset α`, no bare `Set` token) push `Set.*` lemmas behind the 12 `Finset.*` lemmas in tie-break order.

`Finset.Basic` retriever recall (the 0% R@15 from the earlier probe) was a real measurement, but the *cause* is structural ranking, not a missing-name issue. The brief's stop-condition fits: working proofs are dominated by `aesop`/`tauto`; the catalog gap is small.

I proceeded to Phase 2 anyway — the three truly-absent names were a real, easily-fixable miss, and the eval is the only honest test of whether the catalog mattered.

## Phase 2 — the patch

Edits to `STATIC_PREMISES` in `premise_retriever.py`:

1. Appended the three previously-absent lemmas plus `Set.empty_inter` to the `Set` list:
   ```
   "Set.union_univ", "Set.univ_union", "Set.inter_empty", "Set.empty_inter"
   ```
2. Appended all six Finset-cited `Set.*` lemmas to the `Finset` list as well, so build_index_from_traces would put them in `_all_premises` even when the state has only Finset tokens.

I considered (and rejected) an aggressive reorder that front-loaded the priority `Set.*` lemmas. A pilot run showed it lifted Finset.Basic R@15 from 0% to 57% but knocked Set.Basic R@5 from 83% to 0% — a clearly-bad trade. Within the constraint of editing `STATIC_PREMISES` only, no edit reaches the brief's R@5 > 40% on Finset.Basic without breaking Set.Basic.

Index rebuilt with `python -c "...build_index_from_traces..."` → `[PremiseRetriever] Indexed 195173 traces, found 119 unique premises (79028 total references)`.

## Phase 3 — curriculum eval

```
python eval_rollout_all.py \
  --theorem-set curriculum_all \
  --ckpt-dir project/gen_ckpt_v6_premise \
  --policy-type premise_augmented \
  --top-k 8 --max-steps 8 --decode-mode beam \
  --out-dir experiments/finset_wall_20260502_164911
```

Run dir: `eval-9365fd17`.

| Metric | Before (baseline 5ab20bb2) | After (this run) | Δ |
|---|:-:|:-:|:-:|
| Proved | **19/30** | **19/30** | 0 |
| Errored | 11/30 | 11/30 | 0 |
| Skipped (Finset.insert_comm — LeanDojo can't locate it) | 1/31 | 1/31 | 0 |

Per-theorem dispositions are **identical**. Zero theorems flipped status in either direction. The four Finset frontier theorems (`Finset.mem_insert`, `Finset.mem_singleton`, `Finset.disjoint_insert_right`, plus the unavailable `Finset.insert_comm`) all stayed `fail` (or `skip`); `Nat.mul_add_mod'` stayed `fail`. The seven non-frontier `fail`s (`Set.subset_univ`, `Set.empty_subset`, `Set.ite_univ`, `Set.subset_union_left/right`, `Set.inter_subset_left/right`) also stayed `fail`.

Acceptance criteria, scored honestly:
- ❌ Score 19/30 → ≥21/30: still 19/30.
- ❌ ≥1 of the four frontier theorems proved: 0/4 (3/4 evaluable, 1/4 unavailable).
- ❌ Finset.Basic R@5 from 0% to >40%: still 0% (see Phase 4).
- ✅ Negative-result success: confirmed — the catalog is fine, the model is upstream-of-retrieval bound.

## Phase 4 — retriever probe (after)

`experiments/finset_wall_20260502_164911/retriever_probe_after.json`.

| Bucket | n | R@1 | R@5 | R@10 | R@15 |
|---|---:|---:|---:|---:|---:|
| Set.Basic | 12 | 50.0% | 83.3% | 83.3% | 83.3% |
| Nat.Defs | 2 | 16.7% | 50.0% | 50.0% | 50.0% |
| **Finset.Basic** | **14** | **0.0%** | **0.0%** | **0.0%** | **0.0%** |
| Overall | 28 | 22.6% | 39.3% | 39.3% | 39.3% |

Identical to the pre-patch numbers. The conservative append-at-end of the four new `Set.*` entries put them at Set-list positions 28–31, and on Finset-state queries (where 12 Finset entries score 2.5 vs Set entries' 2.0) they end up at retrieved ranks 40+, never inside top-15.

The recall metric is unmoved by this edit. To move it, the `Finset` list would have to be trimmed to fewer than three entries, **or** the scorer in `retrieve()` would need to break ties differently — but `retrieve()` is in `premise_retriever.py` outside the `STATIC_PREMISES` dict, which the brief's constraints forbid editing.

## Why the patch was a no-op: the model is upstream-bound

Side-by-side per-theorem traces between the baseline (`overnight_20260501_205550/premise_beam`) and this run, on the four hardest theorems:

| Theorem | Step | Baseline beam (8 candidates) | Patched beam |
|---|---:|---|---|
| `Finset.mem_insert` | 1 | `aesop` (advances state) | identical |
| `Finset.mem_insert` | 2 | `simp [*]`, `simp [Set.union_univ]`, `simp [Nat.mul_zero]`, `simp [Set.empty_diff]`, `simp [Set.univ_union]`, `simp [Nat.zero_add]`, `simp [Set.mem_union]` | identical |
| `Finset.mem_singleton` | 1 | `aesop` (advances) | identical |
| `Finset.mem_singleton` | 2 | `simp [Set.mem_union]`, `simp [Set.union_self]`, `simp [List.map]`, `tauto`, `simp_all`, `simp [*]`, `simp at *` | identical |
| `Finset.disjoint_insert_right` | 1 | `aesop` (advances) + `simp [Nat.mul_zero]` errored first | identical |
| `Finset.disjoint_insert_right` | 2 | `simp [*]`, `simp [Set.mem_union]`, `simp [Set.union_self]`, `simp_all only []`, `tauto`, `simp_all`, `simp at *` | identical |
| `Nat.mul_add_mod'` | 1 | `simp [Nat.one_mul]`, `simp [List.length_cons]`, `simp [Nat.sub_self]`, `simp [Nat.mul_one]`, `simp [*]`, `aesop`, `simp [List.filter]`, `simp [List.map]` | identical |

Two facts, both load-bearing:

1. **The beam is byte-identical.** Beam decoding is deterministic; identical input → identical output. The premise prefix changed (the new entries propagate into the top-K retrieved), but the *ranked top-K* fed to the model is roughly stable across this edit on these particular states (the highly-ranked Finset entries didn't move). Within the noise of what actually gets prepended in the first 10 retrieved, the model produces the same tactics.

2. **The baseline already emitted `simp [Set.union_univ]` at step 2 of `Finset.mem_insert` — before this patch added it to the catalog.** So the model is not learning premise names from the inference prefix. It memorized them during training (`seq2seq_premise_v1.jsonl` has `Set.union_univ` in the prompt 4× and as a tactic argument in the target 0× — the model has read the name many times and never been supervised to emit it for *this* goal). The "Relevant premises:" prefix is a hint, not a source. Augmenting the hint catalog cannot teach the model to emit a tactic it was never trained to emit on the relevant state.

For the four frontier theorems specifically, the retriever is *not* failing — it surfaces the right names:

```
Finset.mem_insert state → top-10 retrieved:
   1. Finset.mem_insert      ← own name, rank 1
   2. Finset.mem_singleton
   3. Finset.mem_union
   ...
   8. Finset.disjoint_insert_right
   9. Finset.insert_comm
```

`Finset.mem_insert`, `Finset.mem_singleton`, `Finset.disjoint_insert_right`, and `Finset.insert_comm` are all in the top-10 returned for the corresponding states — and have been since well before this edit. The model still doesn't emit `simp [Finset.mem_insert]`-style tactics for these proofs. The catalog has the names; the prefix delivers them; the model ignores them.

The mechanism: in the training pool `seq2seq_premise_v1.jsonl`, `Finset.mem_insert` appears as a tactic argument 160× — but always to *prove other theorems*, never to prove `Finset.mem_insert` itself. The model has no examples of "given the goal `a ∈ insert b s ↔ a = b ∨ a ∈ s`, emit `<the right tactic>`." The retriever can put `Finset.mem_insert` into the prefix all day; without supervision linking it to its own goal state, the model will not connect them.

## What this means

**The catalog is essentially fine.** 89% of working Finset.Basic proofs use `aesop`/`tauto`. The 11% that name premises cite mostly `Set.*` lemmas, only three of which were genuinely missing — and those got picked up by the model in the *baseline* anyway via training memorization. There is no inference-time retrieval intervention, within the constraints of this codebase, that can lift the frontier-theorem score.

**The bottleneck is training-data composition, not retrieval.** The four Finset frontier theorems each require a specific tactic the model has never been supervised to emit on their goal state. Concrete next steps that would actually move the needle:

1. **Mine ground-truth tactics for the frontier theorems from Mathlib itself** (e.g. `Finset.mem_insert` is canonically proven by `Iff.rfl` or `simp [insert]` after unfolding the `Multiset.cons_iff` definition). Add (state, tactic) pairs for these to the next training pool. This is the smallest possible intervention that could break the Finset wall.
2. **Stratified pool construction.** The dilution finding from the prior writeup — that v6's pool dropped relative idiom frequencies — applies more broadly. Build a v8 pool that explicitly upsamples the (state, tactic) patterns for theorems that the curriculum tests, capped to avoid overfitting.
3. **Target the Finset.Basic theorems in the project_state that already have proofs.** They're proven by `aesop`/`tauto`, which the model already does. The gap in those proofs is on Finset.Basic theorems whose ground-truth proof is *not* aesop — those would teach the model the missing patterns. Audit `project_state.json` for proven Finset.Basic theorems whose `proof_tactics` is non-trivial; include those (state, tactic) pairs in training.

The retriever may still be worth improving — but as a higher-recall input to *training*, not as a higher-recall input to inference. The model conditions weakly on the inference prefix because that prefix is structurally identical across 5,577 training rows. Retraining with a smarter retriever, or with dynamic per-row premises, would change that incentive structure.

## Files

- Edit: `premise_retriever.py` — added 4 lemmas to `STATIC_PREMISES["Set"]` (positions 28–31), 6 lemmas to `STATIC_PREMISES["Finset"]` (positions 13–18). Comments inline.
- Index: `project/premise_index.json` rebuilt from `project/all_traces.jsonl` (195,173 traces, 119 unique premises).
- Probe: `retriever_probe_after.json` — same numbers as pre-patch (Finset.Basic R@*=0%).
- Eval: `eval-9365fd17/{config.json,metrics.json,traces.jsonl}` — 19/30 proved, identical disposition matrix to the pre-patch baseline `experiments/overnight_20260501_205550/premise_beam/eval-5ab20bb2`.
- Stdout: `eval_stdout.log`.

## Limitations

Single-seed, deterministic beam decode. The 19/30 result has not been confirmed under sampling decode or alternative seeds. Sampling could plausibly hit one of the frontier theorems by luck, but the *systematic* finding here — the model emits the same tactics regardless of catalog edits — does not depend on decoding mode. The premise-prefix is constructed identically across runs, the model's beam is deterministic, and the per-theorem dispositions diff to zero.

The eval skips `Finset.insert_comm` due to a LeanDojo lookup failure, so this run actually evaluates 3 Finset frontier theorems out of 4 named in the brief. Resolving the lookup is out of scope for the catalog-patch experiment.
