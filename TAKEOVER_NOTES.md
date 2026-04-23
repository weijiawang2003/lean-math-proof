# Takeover Notes — lean-supervised

## If I were taking over this repo tomorrow

### First 3 files to open
1. `policy.py` — the import-time bomb; must understand before touching anything else
2. `clf_ckpt/action_space.json` — verify it matches `search_v2` (42 actions); it does currently
3. `model_rollout.py` — the integration point between policy + Dojo; shows what breaks first

### First 3 fixes to implement
1. **Refactor `policy.py` to lazy-load** — unblocks safe imports, testing, multi-checkpoint experiments
2. **Add action-space validation** — assert `num_labels == len(actions)` at load time; remove silent `idx=0` fallback
3. **Add train/val split** — even 90/10 with eval accuracy logged per epoch; needed to judge whether training is doing anything

### First end-to-end experiment to rerun
```bash
python run_pipeline.py --pipeline classifier \
  --theorem-set nat_single \
  --action-space search_v2 \
  --auto-eval
```
Then manually inspect:
- `sft_dataset.jsonl` — how many rows, label distribution
- `clf_ckpt/action_space.json` — matches `search_v2`
- rollout metrics — finished=True or error?

### What to postpone
- Beam search unification (works, just duplicated)
- Char-LM archival (not hurting anything, just cluttering)
- Moving defaults into run dirs (cosmetic)
- README consolidation (operational docs exist, just scattered)
- Benchmark specs integration (currently advisory only)
- Extended theorem sets beyond `mixed_easy_v2` (fix infra first)

## Current project status

The pipeline mechanically works end-to-end on `nat_single` with `search_v2`. The checkpoint in `clf_ckpt/` was trained on `search_v2` (42 actions). Out of 8 rollout runs saved, 5 finished the proof in 1 step, 2 errored in 1 step, 1 ran all 5 steps without finishing. This is a small enough theorem set that the signal is dominated by memorization, not generalization.

The project is **not yet ready for meaningful proof-rate optimization**. The training data is 133 SFT rows from 152 search transitions — essentially a single theorem's search tree. Before optimizing accuracy, you need: (1) train/val split to know if the model generalizes at all, (2) multi-theorem training data from `mixed_easy_v2` or larger, (3) confidence that action-space consistency is enforced, not just hoped for.

## What success looks like in the next 2 weeks
1. Policy loading is safe and testable
2. Training reports eval accuracy each epoch
3. You can run `mixed_easy_v2` end-to-end without manual checkpoint babysitting
4. You have a clear number for "episode success rate on held-out theorems from search data"
