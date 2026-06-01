# TR4 ablation study (OOF PR-AUC, group=theorem)

| feature group | cols | OOF PR-AUC |
|---|---|---|
| theorem_name_only | 1117 | 0.017 |
| goal_only | 495 | 0.0159 |
| lemma_name_only | 913 | 0.0153 |
| program_tactic_only | 2053 | 0.391 |
| symbolic_and_interactions_only | 247 | 0.2983 |
| all_features | 4825 | 0.5216 |

- all-features 0.5216 vs best-single 0.391 (interaction gain 0.1306)

all-features materially exceeds the best single block (+0.1306 PR-AUC) → the ranker uses retrieval/program INTERACTIONS, not a single memorized cue.
