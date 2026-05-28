# Finset Wall — Brief for Claude Code

> You are Claude Code, running in `~/dev/dojo_sandbox`. This task fixes the
> highest-leverage open problem in the project. Read this whole brief before
> doing anything. It builds on the FINDINGS document at
> `experiments/capacity_isolation_20260502_002705/FINDINGS.md` — read that
> first if you have not already.

## The problem in one paragraph

Five theorems in `curriculum_all` are unsolved by *every* checkpoint trained
to date — `gen_v5`, `gen_ckpt_v6_premise`, `gen_v6` (t5-base on v6 data),
and `gen_v7` (t5-base on v5 data). Four of them are `Finset.Basic`:
`Finset.mem_insert`, `Finset.mem_singleton`, `Finset.disjoint_insert_right`,
`Finset.insert_comm`. The fifth is `Nat.mul_add_mod'`.

Capacity scaling and training-data growth do not move any of these. The
likely upstream cause: the premise retriever (`premise_retriever.py`,
`project/premise_index.json`) measured 0% recall@15 across 14 evaluable
`Finset.Basic` proofs in an earlier probe — meaning the lemmas that
`Finset.Basic` proofs actually cite are absent from the retriever's
catalog. If the retriever can't surface the right premises, the
premise-augmented model can't condition on them, so it can't learn to
emit `simp [Finset.<lemma>]` tactics for these theorems.

## Goal

Lift `gen_ckpt_v6_premise`'s curriculum score above 19/30 by fixing the
Finset coverage gap in the premise retriever. Best-case outcome: one or
more of the four Finset frontier theorems gets proved for the first time
in this project's history. Realistic outcome: a measurable score lift on
Finset-adjacent theorems even if the four hardest stay unsolved.

## Concrete plan

### Phase 1 — diagnose the gap

1. Read `project/project_state.json`. Filter to theorems with
   `file_path` containing `Mathlib/Data/Finset/Basic.lean` and `proved: true`
   and a non-empty `proof_tactics`.
2. For each, run `extract_premises_from_tactic` (in `premise_retriever.py`)
   on `proof_tactics`. Collect every distinct premise name that appears.
3. Compare to `STATIC_PREMISES["Finset"]` in `premise_retriever.py`.
   Report which premises are *cited by working proofs but missing from
   the catalog*.

If the result is "the catalog is fine, working proofs use mostly `aesop` /
`tauto` / no named premises," the retriever isn't the bottleneck — stop
and report that finding instead. Don't proceed to Phase 2.

### Phase 2 — patch the catalog

1. Add the missing premise names to `STATIC_PREMISES["Finset"]` in
   `premise_retriever.py`.
2. Rebuild the premise index by running:
   ```bash
   python -c "from premise_retriever import PremiseRetriever; \
     r = PremiseRetriever(); \
     r.build_index_from_traces('project/all_traces.jsonl'); \
     r.save_index('project/premise_index.json')"
   ```

### Phase 3 — re-evaluate

Run `gen_ckpt_v6_premise` on `curriculum_all` with the patched retriever:

```bash
python eval_rollout_all.py \
  --theorem-set curriculum_all \
  --ckpt-dir project/gen_ckpt_v6_premise \
  --policy-type premise_augmented \
  --top-k 8 --max-steps 8 \
  --decode-mode beam \
  --out-dir experiments/finset_wall_$(date +%Y%m%d_%H%M%S)
```

Should take ~5–15 minutes on this Mac. Save the run dir path.

### Phase 4 — also re-probe the retriever

Re-run the retriever-quality probe to confirm Finset recall actually
improved:

```bash
python experiments/retriever_probe.py \
  --state project/project_state.json \
  --traces project/all_traces.jsonl \
  --premise-index project/premise_index.json \
  --out experiments/finset_wall_<TIMESTAMP>/retriever_probe_after.json
```

Compare against the previous probe (Set.Basic 83% R@5, Finset 0% R@15).

### Phase 5 — write findings

Produce `experiments/finset_wall_<TIMESTAMP>/FINDINGS.md` with:

- The list of premises that were missing and were added.
- Before / after curriculum score for `gen_ckpt_v6_premise`.
- For each of the four Finset frontier theorems: did it get solved?
  If yes, with what tactic? If no, what did the model try?
- Before / after retriever recall@5 / @15 on Finset.Basic.
- Honest assessment: did the gap fix matter, partially matter, or
  not move the needle?

If the curriculum score did NOT move, dig into the per-theorem traces in
the new run dir and check: is the model emitting the new premises at all?
Is it emitting them in correctly-formed tactics? If the model still
ignores them, the bottleneck is upstream of retrieval (the model never
learned to use these premises during training, regardless of what gets
injected at inference). That itself is a useful finding.

## Constraints

You may:
- Edit `premise_retriever.py` (the `STATIC_PREMISES` dict only).
- Read any file in the repo.
- Run the eval and the rebuild commands above.
- Write new files under `experiments/finset_wall_*/`.

You should NOT:
- Edit `eval_rollout_all.py`, `generative_policy.py`, `policy.py`,
  `model_rollout.py`, `env.py`, `core_types.py`, `tasks.py`, or any
  `train_*.py` script.
- Re-train any model.
- Run rollouts with anything other than the existing checkpoints.
- Modify `project/project_state.json` or `project/all_traces.jsonl`.
- Edit `MEMORY.md` or any file under `.claude/`.

If the static catalog patch turns out to be the wrong fix, do not
freelance into deeper changes. Write up what you found, propose what the
right fix would be, and stop.

## Acceptance criteria

This task is a clear success if any of these hold:
- `gen_ckpt_v6_premise` score moves from 19/30 to 21/30 or higher.
- At least one of the four Finset frontier theorems becomes proved.
- The retriever recall@5 on Finset.Basic moves from 0% to >40%.

This task is also a useful (negative-result) success if:
- The catalog turns out to be fine and the upstream issue is the model
  never learned Finset tactics. That redirects future effort toward
  training-data composition, not retrieval.

## Style

- Concrete > ceremonial. Cite theorem names and tactic strings.
- Numbers from the actual run, not from this brief.
- If results contradict the prior, say so loudly.
- Skip apologies, skip "as a language model" framing.

When you finish, print the path to `FINDINGS.md` and a 3-line summary of
the top finding. That's it.
