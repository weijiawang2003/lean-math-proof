# Legacy / Archived Scripts

These files were part of earlier exploration and are **not** integrated with the
current classifier pipeline.  They are kept here for reference.

| File | What it was |
|------|-------------|
| `train_sft_char_lm.py` | Character-level GRU language model for tactic generation. |
| `collect_trace_nat_mul_add_mod.py` | Ad-hoc single-theorem trace script (bypasses shared modules). |
| `mini_dojo_test.py` | Minimal LeanDojo smoke test. |
| `char_lm_ckpt.pt` | Saved char-LM checkpoint. |
| `toy_trace.jsonl` | Early trace artifact (incomplete schema). |
| `toy_dataset.jsonl` | Early SFT dataset artifact (incomplete schema). |

If you want to revive generative tactic prediction, consider a transformer-based
approach integrated with the shared `core_types`, `trace_io`, and `experiment_io`
modules rather than extending the char-LM path.
