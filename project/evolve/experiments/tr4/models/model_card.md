# TR4 model card

Program-level success rankers. EXPLORATORY — not production routing, not promoted.
- training rows: 4737 ((theorem,program) pairs)
- success positives: 23 (0.49%)
- models: heuristic / logistic / sgd / hgb
- headline metric: leakage-free OOF (GroupKFold by theorem)
- intended use: rank candidate programs to cut probe budget in future TR3-style search
- NOT for: proof generation, production routing, automatic promotion
