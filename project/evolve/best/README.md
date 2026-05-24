# Best evolved genomes

## `ns9_best_genome.json` — current best (NS9, 2026-05-24)

Strategy-config JSON consumed by `eval_rollout_all.py
--strategy-config`. Reproduces:

  - **`nat_defs_medium`**: **37/38** (97.4%) in ~2.5 min
  - **`nat_defs_large_v5`**: **49/65** (75.4%) in ~5 min

against the unchanged `project/models/gen_v5` t5-small checkpoint.

### Composition (17 enabled skeletons)

  - 12 priority_templates: `pt_iff_{0..7}`, `pt_any_{9,10}`,
    `pt_eq_11`, `pt_le_12`, `pt_lt_8`
  - 3 family_tactic: `fam_mod_{13,14,15}` (no `fam_div_*` — pruned
    under NS9 retrieval-gate decoupling)
  - 1 fallback_tactic: `fb_16`
  - 1 retrieval gate (dynamic):
    `retrieval_family_gates=["div", "mod", "pow"]`,
    `retrieval_requires_family=False`

### Reproduce

```bash
# Medium (~2.5 min, expect 37/38)
python eval_rollout_all.py \
  --theorem-set nat_defs_medium \
  --policy-type hybrid_evolved \
  --ckpt-dir project/models/gen_v5 \
  --top-k 8 --max-steps 8 \
  --strategy-config project/evolve/best/ns9_best_genome.json \
  --out-dir /tmp/ns9_repro_medium

# Large (~5 min, expect 49/65)
python eval_rollout_all.py \
  --theorem-set nat_defs_large_v5 \
  --policy-type hybrid_evolved \
  --ckpt-dir project/models/gen_v5 \
  --top-k 8 --max-steps 8 \
  --strategy-config project/evolve/best/ns9_best_genome.json \
  --out-dir /tmp/ns9_repro_large
```

Metrics land at `<out-dir>/eval-*/metrics.json`. Proved count is
the `proved` field.

### Provenance

Promoted at cycle 3 of the NS9 sweep
(`project/evolve/ns9_runs/ns8-20260523-234547-fc1277/` — the
`ns8-` prefix is from the NS8 runner code; this is the NS9 result).
Two mutations from the NS8 seed (also 37/49 at 20 enabled):

  - Cycle 2: `disable_dead_skeleton` removed `fb_19, fam_div_14`
    (the exact pair NS6 cycle 4 tried; NS7/NS8 always rejected).
  - Cycle 3: `disable_dead_skeleton` removed `pt_iff_2`.

Both pairs were previously rank-coupled to the critical
`retrieved:Nat.div_lt_iff_lt_mul:rw` skeleton; NS9's
`retrieval_requires_family=False` gate makes them safe.

See `project/evolve/reports/ns9_retrieval_gate_decoupling.md` for
the detailed simulator replay and sweep table, and
`project/evolve/reports/skeleton_evolution_final_report.md` for the
full v3→NS9 progression.
